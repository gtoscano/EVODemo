"""Docker contexts and private bind-mount staging for rootless Docker.

OpenCode and candidates always run as a non-root container user. Two small,
trusted Python helpers prepare/recover ownership of disposable copies only.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import shutil
import stat
import subprocess
import tempfile
import uuid

IMAGE = "mini-ea-opencode:1.4.3"
OUTPUT_LIMIT = 1024 * 1024


class CleanupError(Exception):
    """Stop if a container may still run or private staging cannot be recovered."""


def limit_output():
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT, OUTPUT_LIMIT))


class Docker:
    def __init__(self, context="default", rootless=False, image=IMAGE):
        if not isinstance(context, str) or not context.strip() or context.startswith("-"):
            raise ValueError("Choose a nonempty Docker context name")
        self.context, self.rootless, self.image = context, rootless, image

    def command(self, *parts):
        return ["docker", "--context", self.context, *parts]

    def check(self):
        if not shutil.which("docker"):
            raise RuntimeError("Docker is required; install it and start its daemon")
        if os.getuid() == 0:
            raise RuntimeError("Run the demo as a non-root user with Docker access")
        for parts, message in (
            (("version", "--format", "{{.Server.Version}}"),
             f"Docker is not accessible. Run 'docker --context {self.context} version' and check daemon/socket permissions."),
            (("image", "inspect", self.image),
             f"Docker image {self.image} is missing or unreadable in context {self.context}. Build the image using this project's README."),
        ):
            if subprocess.run(self.command(*parts), capture_output=True, timeout=15).returncode:
                raise RuntimeError(message)
        info = subprocess.run(self.command("info", "--format", "{{json .}}"),
                              capture_output=True, text=True, timeout=15)
        try:
            settings = json.loads(info.stdout)
            security = settings["SecurityOptions"]
            if info.returncode or not isinstance(security, list):
                raise ValueError()
            self.rootless = any(item in ("rootless", "name=rootless") for item in security)
            self.server_version = str(settings.get("ServerVersion", ""))
        except (ValueError, TypeError, KeyError):
            raise RuntimeError("Cannot determine Docker daemon security options") from None
        print(f"Docker context: {self.context} ({'rootless' if self.rootless else 'regular'})", flush=True)
        if self.rootless:
            if str(settings.get("CgroupVersion")) != "2" or settings.get("CgroupDriver") != "systemd":
                raise RuntimeError("Rootless Docker requires cgroup v2 with systemd; ask the lab administrator")
            probe = self.run(["--network=none"], ["python", "-I", "-c",
                'from pathlib import Path; import json, os; '
                'print(json.dumps({"uid": os.getuid(), **{n: Path("/sys/fs/cgroup", n).read_text().strip() '
                'for n in ("cpu.max", "memory.max", "memory.swap.max", "pids.max")}}))'],
                timeout=15, memory_mb=128)
            try:
                limits = json.loads(probe)
                quota, period = map(int, limits["cpu.max"].split())
                if (limits["uid"] != 1000 or quota != period or period <= 0
                        or limits["memory.max"] != str(128 * 1024 * 1024)
                        or limits["memory.swap.max"] != "0" or limits["pids.max"] != "64"):
                    raise ValueError()
            except (ValueError, TypeError, KeyError):
                raise RuntimeError("Rootless Docker resource limits are not enforced; check cgroup delegation") from None
            self.check_mounts()  # Detect ownership problems before any model request.

    def check_mounts(self):
        """Trusted fixtures only: verify writes and read-only access, without OpenCode."""
        with tempfile.TemporaryDirectory(prefix="mini-ea-mount-check-") as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            candidate.mkdir()
            context = root / "context.txt"
            context.write_text("trusted fixture")
            context.chmod(0o600)
            self.run(["--network=none",
                "--mount", f"type=bind,src={candidate},dst=/candidate",
                "--mount", f"type=bind,src={context},dst=/context.txt,readonly"],
                ["python", "-I", "-c", '''from pathlib import Path
import errno
assert Path("/context.txt").read_text() == "trusted fixture"
try:
    Path("/context.txt").write_text("forbidden")
except OSError as error:
    assert error.errno in (errno.EROFS, errno.EACCES, errno.EPERM)
else:
    raise AssertionError("Trusted mount was writable")
Path("/candidate/created.txt").write_text("write passed")
'''], timeout=15, memory_mb=128)
            if ((candidate / "created.txt").read_text() != "write passed"
                    or context.read_text() != "trusted fixture"):
                raise RuntimeError("Docker mount self-check failed")

    def run(self, options, command, **kwargs):
        user = "1000:1000" if self.rootless else f"{os.getuid()}:{os.getgid()}"
        with self.mounts(options) as staged:
            return self.execute(staged, command, user=user, **kwargs)

    def execute(self, options, command, *, user, timeout, input_text=None, environment=None,
                memory_mb=512, diagnostics_path=None, allow_failure=False,
                summarize=lambda output: "", output_limiter=limit_output):
        if os.getuid() == 0:
            raise RuntimeError("Run the demo as a non-root user")
        name = "mini-ea-" + uuid.uuid4().hex
        args = self.command("run", "--pull=never", "--name", name,
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user", user, "--cpus=1", f"--memory={memory_mb}m",
            f"--memory-swap={memory_mb}m", "--pids-limit=64", "--log-driver=none",
            *options, self.image, *command)
        with tempfile.TemporaryFile() as output:
            try:
                result = subprocess.run(args, input=input_text, text=True, stdout=output,
                    stderr=subprocess.STDOUT, timeout=timeout, preexec_fn=output_limiter, env=environment)
                output.seek(0)
                message = output.read(OUTPUT_LIMIT).decode("utf-8", errors="replace")
                if result.returncode and not allow_failure:
                    try:
                        state = subprocess.run(self.command("inspect", "--format", "{{.State.OOMKilled}}", name),
                                               capture_output=True, text=True, timeout=10)
                        oom = state.stdout.strip() if state.returncode == 0 else "unknown"
                    except (OSError, subprocess.TimeoutExpired):
                        oom = "unknown"
                    if oom == "true":
                        raise RuntimeError(f"Container ran out of memory (limit {memory_mb} MiB; Docker OOMKilled=true)")
                    if result.returncode == 137:
                        raise RuntimeError(f"Container exited with status 137 (SIGKILL; Docker OOMKilled={oom})")
                    raise RuntimeError(f"Container exited with status {result.returncode}")
                return message
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Container timed out after {timeout} seconds") from None
            finally:
                try:
                    if diagnostics_path is not None:
                        output.seek(0)
                        diagnostics_path.write_text(summarize(output.read(OUTPUT_LIMIT).decode("utf-8", errors="replace")))
                finally:
                    try:
                        cleanup = subprocess.run(self.command("rm", "-f", name), capture_output=True, timeout=10)
                    except (OSError, subprocess.TimeoutExpired):
                        raise CleanupError(f"Cannot confirm container cleanup in context {self.context}: {name}") from None
                    if cleanup.returncode and b"No such container" not in cleanup.stderr:
                        raise CleanupError(f"Cannot confirm container cleanup in context {self.context}: {name}")

    def ownership(self, directory, recover=False):
        # Fixed trusted code: never import, execute or follow files in /stage.
        script = '''import os, stat, sys
recover = sys.argv[1] == "recover"
for parent, directories, files in os.walk("/stage", followlinks=False):
    for path in [parent, *(os.path.join(parent, n) for n in directories + files)]:
        if path == "/stage":
            continue
        mode = os.lstat(path).st_mode
        os.lchown(path, 0 if recover else 1000, 0 if recover else 1000)
        if recover and not stat.S_ISLNK(mode):
            os.chmod(path, 0o700 if stat.S_ISDIR(mode) else 0o600)
'''
        self.execute(["--network=none", "--cap-add=CHOWN", "--cap-add=DAC_OVERRIDE", "--cap-add=FOWNER",
            "--mount", f"type=bind,src={directory},dst=/stage"],
            ["python", "-I", "-c", script, "recover" if recover else "prepare"],
            user="0:0", timeout=15, memory_mb=128)

    @contextmanager
    def mounts(self, options):
        if not self.rootless or "--mount" not in options:
            yield options
            return
        # A private 0700 ancestor protects all staged material from other host users.
        directory = Path(tempfile.mkdtemp(prefix="mini-ea-rootless-"))
        staged, writable = list(options), []
        prepared = False
        uncertain_cleanup = False
        try:
            for i, option in enumerate(options):
                if option != "--mount":
                    continue
                fields = dict(part.split("=", 1) if "=" in part else (part, "") for part in options[i + 1].split(","))
                if fields.get("type") != "bind" or set(fields) - {"type", "src", "dst", "readonly"}:
                    raise ValueError("Only explicit bind mounts are supported")
                source = Path(fields["src"])
                target = directory / str(i)
                copy_files(source, target)
                fields["src"] = str(target)
                staged[i + 1] = ",".join(key + ("=" + value if value else "") for key, value in fields.items())
                if "readonly" not in fields:
                    writable.append((target, source))
            prepared = True
            self.ownership(directory)
            yield staged
        except CleanupError:
            uncertain_cleanup = True
            raise
        finally:
            if not uncertain_cleanup:
                if prepared:
                    try:
                        self.ownership(directory, recover=True)
                    except (RuntimeError, OSError, CleanupError):
                        raise CleanupError(f"Cannot recover private rootless staging: {directory}") from None
                try:
                    for staged_source, destination in writable:
                        copy_files(staged_source, destination, replace=True, bounded=True)
                finally:
                    shutil.rmtree(directory)
            # If cleanup is uncertain, retain the private directory and stop the EA.


def copy_files(source, destination, *, replace=False, bounded=False):
    """Copy regular files only; validate output before replacing a disposable workspace."""
    paths = [source]
    if source.is_dir() and not source.is_symlink():
        paths.extend(source.rglob("*"))
    size = 0
    if bounded and len(paths) > 200:
        raise ValueError("Rootless workspace has too many files or directories")
    for path in paths:
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("Rootless mounts require regular files and directories, without symlinks")
        if stat.S_ISREG(mode):
            size += path.stat().st_size
    if bounded and size > 256 * 1024:
        raise ValueError("Rootless workspace exceeds 256 KiB")
    if replace and destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)

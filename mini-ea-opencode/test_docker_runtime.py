"""Offline Docker tests. All daemon/container responses are scripted."""

import contextlib
import ast
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import docker_runtime as runtime


class DockerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_cli_defaults_and_explicit_context(self):
        custom = (Path(__file__).parent / "evolve.py").exists()
        controller = importlib.import_module("evolve" if custom else "demo")
        command = ["evolve.py", "projects/task_manager.md"] if custom else ["demo.py"]
        with patch("sys.argv", command):
            self.assertEqual(controller.arguments().docker_context, "default")
            self.assertEqual(controller.arguments().generations, 1)
        with patch("sys.argv", command + ["--docker-context", "student-engine"]):
            self.assertEqual(controller.arguments().docker_context, "student-engine")

    def test_docker_only_check_does_not_require_opencode_or_credentials(self):
        custom = (Path(__file__).parent / "evolve.py").exists()
        controller = importlib.import_module("opencode_agent" if custom else "demo")
        from argparse import Namespace
        args = Namespace(docker_context="student-engine", check_docker=True)
        with patch.object(runtime.Docker, "check"), patch.object(runtime.Docker, "check_mounts") as mounts, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            controller.prerequisites(args)  # No model/auth/opencode attributes exist.
        mounts.assert_called_once()
        self.assertEqual(args.docker.context, "student-engine")
        self.assertIn("No OpenCode", output.getvalue())

    def test_mount_probe_uses_only_trusted_fixtures(self):
        docker = runtime.Docker()

        def fake(options, command, **kwargs):
            self.assertIn("--network=none", options)
            self.assertEqual(command[:3], ["python", "-I", "-c"])
            ast.parse(command[-1])
            self.assertNotIn("OPENAI_API_KEY", " ".join(options))
            mounts = [option for option in options if option.startswith("type=bind")]
            self.assertTrue(next(m for m in mounts if "dst=/context.txt" in m).endswith(",readonly"))
            candidate = Path(next(m for m in mounts if "dst=/candidate" in m).split("src=", 1)[1].split(",", 1)[0])
            (candidate / "created.txt").write_text("write passed")

        with patch.object(docker, "run", side_effect=fake) as run:
            docker.check_mounts()
        run.assert_called_once()

    def test_context_is_explicit_for_execution_inspection_and_cleanup(self):
        calls = []

        def fake(command, **kwargs):
            calls.append(command)
            self.assertEqual(command[:3], ["docker", "--context", "classroom"])
            if command[3] == "run":
                self.assertEqual(command[command.index("--user") + 1], "123456:234567")
                return subprocess.CompletedProcess(command, 137)
            return subprocess.CompletedProcess(command, 0, "false\n", b"")

        with patch.object(runtime.os, "getuid", return_value=123456), \
             patch.object(runtime.os, "getgid", return_value=234567), \
             patch.object(runtime.subprocess, "run", side_effect=fake):
            with self.assertRaisesRegex(RuntimeError, "SIGKILL"):
                runtime.Docker("classroom").run([], ["python", "--version"], timeout=1)
        self.assertEqual([call[3] for call in calls], ["run", "inspect", "rm"])

    def test_rootless_stages_private_copies_and_keeps_candidate_nonroot(self):
        workspace = self.root / "candidate"
        workspace.mkdir()
        (workspace / "main.py").write_text("raise AssertionError('NEVER EXECUTE ON HOST')\n")
        auth = self.root / "auth.json"
        auth.write_text("PRIVATE_FIXTURE_AUTH")
        auth.chmod(0o600)
        calls, stage_roots = [], []

        def fake(command, **kwargs):
            calls.append(command)
            self.assertEqual(command[:3], ["docker", "--context", "rootless"])
            if command[3] != "run":
                return subprocess.CompletedProcess(command, 0, b"", b"")
            user = command[command.index("--user") + 1]
            mounts = [value for value in command if value.startswith("type=bind")]
            self.assertIn("--read-only", command)
            self.assertIn("--security-opt=no-new-privileges", command)
            if user == "0:0":
                self.assertIn("--network=none", command)
                self.assertNotIn("--env", command)
                self.assertIn("--cap-add=CHOWN", command)
                self.assertEqual(command[-4:-2], ["-I", "-c"])
                self.assertIn('os.walk("/stage", followlinks=False)', command[-2])
                ast.parse(command[-2])
                stage_roots.append(Path(mounts[0].split("src=", 1)[1].split(",", 1)[0]))
            else:
                self.assertEqual(user, "1000:1000")
                self.assertNotIn("--cap-add=CHOWN", command)
                self.assertNotIn("--privileged", command)
                paths = {value.split("dst=", 1)[1].split(",", 1)[0]:
                         Path(value.split("src=", 1)[1].split(",", 1)[0]) for value in mounts}
                staged = paths["/candidate"]
                self.assertNotEqual(staged, workspace)
                self.assertEqual(staged.parent.stat().st_mode & 0o777, 0o700)
                self.assertEqual((staged / "main.py").read_text(), (workspace / "main.py").read_text())
                self.assertTrue(next(m for m in mounts if "dst=/auth" in m).endswith(",readonly"))
                self.assertEqual(paths["/auth"].read_text(), "PRIVATE_FIXTURE_AUTH")
                (staged / "main.py").write_text("# Generated output saved as text, never imported\n")
                kwargs["stdout"].write(b"ok")
            return subprocess.CompletedProcess(command, 0)

        with patch.object(runtime.subprocess, "run", side_effect=fake), \
             patch.object(runtime.os, "getuid", return_value=987654321):
            result = runtime.Docker("rootless", rootless=True).run([
                "--mount", f"type=bind,src={workspace},dst=/candidate",
                "--mount", f"type=bind,src={auth},dst=/auth,readonly"], ["opencode", "fake"], timeout=1)
        self.assertEqual(result, "ok")
        self.assertIn("Generated output", (workspace / "main.py").read_text())
        self.assertEqual(auth.read_text(), "PRIVATE_FIXTURE_AUTH")
        self.assertEqual(auth.stat().st_mode & 0o777, 0o600)
        self.assertEqual(len(stage_roots), 2)
        self.assertTrue(all(not path.exists() for path in stage_roots))

    def test_readonly_mount_and_source_survive_staging(self):
        source = self.root / "candidate"
        source.mkdir()
        (source / "main.py").write_text("# Parent\n")
        docker = runtime.Docker("lab", rootless=True)
        with patch.object(docker, "ownership") as helper:
            with docker.mounts(["--mount", f"type=bind,src={source},dst=/candidate,readonly"]) as options:
                staged = Path(options[1].split("src=", 1)[1].split(",", 1)[0])
                self.assertTrue(options[1].endswith(",readonly"))
                (staged / "main.py").write_text("# Fake modification to disposable copy\n")
            self.assertEqual(helper.call_count, 2)
        self.assertEqual((source / "main.py").read_text(), "# Parent\n")

    def test_timeout_recovers_partial_artifacts_without_replacing_parent(self):
        source = self.root / "workspace"
        source.mkdir()
        parent = self.root / "parent.py"
        parent.write_text("# Saved parent\n")
        docker = runtime.Docker("lab", rootless=True)
        with patch.object(docker, "ownership") as helper:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                with docker.mounts(["--mount", f"type=bind,src={source},dst=/candidate"]) as options:
                    staged = Path(options[1].split("src=", 1)[1].split(",", 1)[0])
                    (staged / "partial.py").write_text("# Partial generated text\n")
                    raise RuntimeError("timed out")
            self.assertEqual(helper.call_count, 2)
        self.assertTrue((source / "partial.py").exists())
        self.assertEqual(parent.read_text(), "# Saved parent\n")

    def test_unsafe_outputs_do_not_replace_workspace_or_follow_links(self):
        for unsafe in ("symlink", "oversized"):
            with self.subTest(unsafe=unsafe):
                source = self.root / unsafe
                source.mkdir()
                (source / "original.py").write_text("# Parent copy\n")
                docker = runtime.Docker("lab", rootless=True)
                with patch.object(docker, "ownership"), self.assertRaises(ValueError):
                    with docker.mounts(["--mount", f"type=bind,src={source},dst=/candidate"]) as options:
                        staged = Path(options[1].split("src=", 1)[1].split(",", 1)[0])
                        if unsafe == "symlink":
                            (staged / "escape").symlink_to(self.root)
                        else:
                            (staged / "large.py").write_bytes(b"x" * (256 * 1024 + 1))
                self.assertEqual((source / "original.py").read_text(), "# Parent copy\n")
                self.assertFalse(staged.parent.exists())

    def test_uncertain_cleanup_stops_and_keeps_private_staging(self):
        source = self.root / "workspace"
        source.mkdir()
        docker = runtime.Docker("lab", rootless=True)
        with patch.object(docker, "ownership") as helper:
            with self.assertRaises(runtime.CleanupError):
                with docker.mounts(["--mount", f"type=bind,src={source},dst=/candidate"]) as options:
                    staged = Path(options[1].split("src=", 1)[1].split(",", 1)[0])
                    retained = staged.parent
                    self.addCleanup(runtime.shutil.rmtree, retained)
                    raise runtime.CleanupError("Cannot confirm Docker cleanup")
            helper.assert_called_once()
        self.assertTrue(retained.exists())
        self.assertEqual(retained.stat().st_mode & 0o777, 0o700)

    def check_daemon(self, *, rootless, limits=None, context="lab", driver="systemd"):
        calls = []
        settings = {"SecurityOptions": ["name=rootless"] if rootless else ["name=seccomp"],
                    "CgroupVersion": "2", "CgroupDriver": driver}
        default_limits = {"uid": 1000, "cpu.max": "100000 100000", "memory.max": "134217728",
                          "memory.swap.max": "0", "pids.max": "64"}

        def fake(command, **kwargs):
            calls.append(command)
            self.assertEqual(command[:3], ["docker", "--context", context])
            operation = command[3]
            if operation == "info":
                return subprocess.CompletedProcess(command, 0, json.dumps(settings), "")
            if operation == "run":
                self.assertIn("--network=none", command)
                self.assertEqual(command[command.index("--user") + 1], "1000:1000")
                self.assertNotIn("--env", command)
                self.assertNotIn("--mount", command)
                ast.parse(command[-1])
                kwargs["stdout"].write(json.dumps(limits or default_limits).encode())
            return subprocess.CompletedProcess(command, 0, b"", b"")

        docker = runtime.Docker(context)
        with patch.object(runtime.shutil, "which", return_value="/usr/bin/docker"), \
             patch.object(runtime.os, "getuid", return_value=123456), \
             patch.object(docker, "check_mounts"), \
             patch.object(runtime.subprocess, "run", side_effect=fake), contextlib.redirect_stdout(io.StringIO()):
            docker.check()
        return docker, calls

    def test_detects_actual_daemon_not_context_name(self):
        docker, calls = self.check_daemon(rootless=True, context="arbitrary-name")
        self.assertTrue(docker.rootless)
        self.assertEqual([call[3] for call in calls], ["version", "image", "info", "run", "rm"])
        docker, calls = self.check_daemon(rootless=False, context="rootless")
        self.assertFalse(docker.rootless)
        self.assertEqual([call[3] for call in calls], ["version", "image", "info"])

    def test_rootless_rejects_missing_delegation_and_ignored_limits(self):
        with self.assertRaisesRegex(RuntimeError, "cgroup v2"):
            self.check_daemon(rootless=True, driver="none")
        for name in ("cpu.max", "memory.max", "memory.swap.max", "pids.max", "uid"):
            limits = {"uid": 1000, "cpu.max": "100000 100000", "memory.max": "134217728",
                      "memory.swap.max": "0", "pids.max": "64"}
            limits[name] = "max"
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "not enforced"):
                self.check_daemon(rootless=True, limits=limits)

    def test_invalid_daemon_info_is_rejected(self):
        for output in ("not JSON", "{}", "[]", '{"SecurityOptions": null}'):
            responses = [subprocess.CompletedProcess([], 0, b"", b"")] * 2 + [
                subprocess.CompletedProcess([], 0, output, "")]
            with self.subTest(output=output), \
                 patch.object(runtime.shutil, "which", return_value="/usr/bin/docker"), \
                 patch.object(runtime.os, "getuid", return_value=123456), \
                 patch.object(runtime.subprocess, "run", side_effect=responses), \
                 self.assertRaisesRegex(RuntimeError, "security options"):
                runtime.Docker().check()


if __name__ == "__main__":
    unittest.main()

"""Offline container packaging checks: no daemon, provider or downloaded code execution."""

import argparse
import contextlib
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import docker_runtime as runtime
import demo as agent

HERE = Path(__file__).resolve().parent


class ContainerOpenCodeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(
            model="fixture/model", opencode_config=self.root / "no-config",
            opencode_auth=self.root / "no-auth", opencode_memory_mb=2048)

    HELP = "run --pure --dir --model --agent --format"

    def test_startup_uses_only_network_free_container_cli_probes(self):
        self.args.evaluation_image = "classroom/python-opencode:1"
        self.args.check_opencode = True
        calls = []

        def probe(docker, options, command, **kwargs):
            calls.append(command)
            self.assertEqual(docker.image, self.args.evaluation_image)
            self.assertEqual(docker.context, "default")
            self.assertIn("--network=none", options)
            self.assertNotIn("--mount", options)
            self.assertNotIn("OPENAI_API_KEY", " ".join(options))
            self.assertNotIn("PRIVATE_TEST_KEY", " ".join(options))
            self.assertEqual(kwargs, {"timeout": 30, "memory_mb": 2048})
            return "1.4.3\n" if command == ["opencode", "--version"] else self.HELP

        with patch.object(agent.Docker, "check"), \
             patch.object(agent.Docker, "run", autospec=True, side_effect=probe), \
             patch.object(subprocess, "run", side_effect=AssertionError("No host CLI execution")), \
             patch.object(agent, "openai_key", side_effect=AssertionError("No credential reads")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            agent.prerequisites(self.args)
        self.assertEqual(calls, [["opencode", "--version"], ["opencode", "run", "--help"]])
        self.assertIn("container image: classroom/python-opencode:1", output.getvalue())
        self.assertIsNone(self.args.api_key)
        self.assertFalse(hasattr(self.args, "opencode"))

    def test_incompatible_container_cli_is_rejected_before_credentials(self):
        for version, help_text in (("2.0.0", self.HELP), ("1.4.3", "run --model")):
            with self.subTest(version=version, help=help_text), \
                 patch.object(agent.Docker, "check"), \
                 patch.object(agent.Docker, "run", side_effect=[version, help_text]), \
                 patch.object(agent, "openai_key") as key:
                with self.assertRaisesRegex(RuntimeError, "Container OpenCode must be 1.x"):
                    agent.prerequisites(self.args)
                key.assert_not_called()

    def test_missing_executable_reports_rebuild_and_never_falls_back_to_host(self):
        with patch.object(agent.Docker, "check"), \
             patch.object(agent.Docker, "run", side_effect=RuntimeError("Container status 127")), \
             patch.object(subprocess, "run", side_effect=AssertionError("No host CLI execution")), \
             patch.object(agent, "openai_key") as key:
            with self.assertRaisesRegex(RuntimeError, "Rebuild the image"):
                agent.prerequisites(self.args)
            key.assert_not_called()

    def test_cleanup_failure_still_stops_startup(self):
        with patch.object(agent.Docker, "check"), \
             patch.object(agent.Docker, "run", side_effect=agent.CleanupError("Uncertain cleanup")):
            with self.assertRaises(agent.CleanupError):
                agent.prerequisites(self.args)

    def test_build_installs_pinned_linux_architecture_without_executing_download(self):
        # Execute only the trusted Dockerfile's installer against a fake archive.
        # The bytes are never an executed program; there is no network or Docker.
        dockerfile = (HERE / "Dockerfile").read_text()
        installer = dockerfile.split("RUN python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        archive = io.BytesIO()
        executable = b"TRUSTED FIXTURE: NEVER EXECUTE THIS"
        with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
            entry = tarfile.TarInfo("opencode")
            entry.size = len(executable)
            bundle.addfile(entry, io.BytesIO(executable))
        for machine, asset in (("x86_64", "x64-baseline"), ("aarch64", "arm64")):
            with self.subTest(machine=machine), \
                 patch("platform.machine", return_value=machine), \
                 patch("urllib.request.urlopen", return_value=io.BytesIO(archive.getvalue())) as download, \
                 patch.object(Path, "write_bytes", autospec=True) as write, \
                 patch.object(Path, "chmod", autospec=True) as chmod, \
                 patch.object(subprocess, "run", side_effect=AssertionError("No executable runs")):
                exec(compile(installer, "trusted-Dockerfile-installer", "exec"), {})
                download.assert_called_once_with(
                    "https://github.com/anomalyco/opencode/releases/download/v1.4.3/"
                    f"opencode-linux-{asset}.tar.gz", timeout=120)
                write.assert_called_once_with(Path("/usr/local/bin/opencode"), executable)
                chmod.assert_called_once_with(Path("/usr/local/bin/opencode"), 0o755)
        self.assertEqual(runtime.IMAGE, "mini-ea-opencode:1.4.3")
        ignore = (HERE / ".dockerignore").read_text().splitlines()
        self.assertIn("**", ignore)
        self.assertIn("!Dockerfile", ignore)


if __name__ == "__main__":
    unittest.main()

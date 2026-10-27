"""Offline local-server checks: no Docker, network or inference."""
import argparse
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

MINI = not (Path(__file__).parent / "evolve.py").is_file()
agent = importlib.import_module("demo" if MINI else "opencode_agent")


class LocalProviderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = {
            "OPENCODE_MODEL": "local/qwen3.8-27b",
            "OPENCODE_BASE_URL": "http://127.0.0.1:10000/v1",
            "OPENCODE_NETWORK": "host",
        }
        self.args = argparse.Namespace(
            model="local/qwen3.8-27b", config={"provider": {"cloud": {}}},
            docker=SimpleNamespace(rootless=False), opencode_timeout=5,
            opencode_memory_mb=2048, opencode_auth=self.root / "auth.json",
            opencode_config=self.root / "no-config.json", api_key=None)
        self.addCleanup(patch.stopall)
        patch.object(agent, "env_value", side_effect=self.settings.get).start()
        patch.dict(os.environ, {"OPENAI_API_KEY": "PRIVATE_CLOUD_FIXTURE"}).start()

    def configure(self):
        with contextlib.redirect_stdout(io.StringIO()):
            agent.configure_local_provider(self.args)

    def test_local_config_replaces_inherited_provider_and_uses_exact_id(self):
        self.configure()
        self.assertEqual(set(self.args.config["provider"]), {"local"})
        provider = self.args.config["provider"]["local"]
        self.assertEqual(provider["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(provider["options"]["baseURL"], self.settings["OPENCODE_BASE_URL"])
        self.assertEqual(set(provider["models"]), {"qwen3.8-27b"})

    def test_mismatched_model_is_rejected_instead_of_using_cloud(self):
        self.args.model = "openai/gpt-4.1-mini"
        with self.assertRaisesRegex(RuntimeError, "local/SERVER_MODEL_ID"):
            self.configure()

    def test_missing_endpoint_and_invalid_network_or_url_are_rejected(self):
        for changes in (
            {"OPENCODE_BASE_URL": None},
            {"OPENCODE_NETWORK": "none"},
            {"OPENCODE_BASE_URL": "file:///private"},
            {"OPENCODE_BASE_URL": "http://secret:password@localhost:10000/v1"},
        ):
            with self.subTest(changes=changes):
                for key in ("local_llm_url", "opencode_network"):
                    if hasattr(self.args, key):
                        delattr(self.args, key)
                previous = self.settings.copy()
                self.settings.update(changes)
                with self.assertRaises(RuntimeError):
                    self.configure()
                self.settings = previous
                # The mock is bound to the original dictionary.
                agent.env_value.side_effect = self.settings.get

    def test_rootless_host_network_requires_recent_engine(self):
        for version, succeeds in (("29.4.0", False), ("29.5.0", True), ("29.6.2", True)):
            self.args.docker = SimpleNamespace(rootless=True, server_version=version)
            if succeeds:
                self.configure()
            else:
                with self.assertRaisesRegex(RuntimeError, "29.5"):
                    self.configure()

    def test_prerequisites_need_no_cloud_credentials_and_only_probe_cli(self):
        with patch.object(agent.Docker, "check"), \
             patch.object(agent.Docker, "run", side_effect=["1.4.3", "--model --dir --pure --agent --format"]) as run, \
             patch.object(agent, "openai_key", side_effect=AssertionError("No cloud credential read")), \
             contextlib.redirect_stdout(io.StringIO()):
            agent.prerequisites(self.args)
        self.assertIsNone(self.args.api_key)
        for call in run.call_args_list:
            self.assertIn("--network=none", call.args[0])

    def test_coder_and_reviewer_use_local_model_without_cloud_auth_mount(self):
        self.configure()
        self.args.opencode_auth.write_text("PRIVATE_AUTH_FIXTURE")
        candidate, context = self.root / "candidate", self.root / "context"
        candidate.mkdir()
        context.mkdir()
        calls = []

        def transport(options, command, **kwargs):
            calls.append((options, command))
            self.assertNotIn("PRIVATE_CLOUD_FIXTURE", str(options))
            self.assertNotIn("OPENAI_API_KEY", kwargs["environment"])
            self.assertFalse(any("dst=/provider-auth.json" in item for item in options))
            config = json.loads(next(item.split("=", 1)[1] for item in options
                                     if item.startswith("OPENCODE_CONFIG_CONTENT=")))
            self.assertEqual(config["model"], self.args.model)
            self.assertEqual(config["small_model"], self.args.model)
            self.assertEqual(set(config["provider"]), {"local"})
            if MINI:
                mount = next(item for item in options if item.startswith("type=bind,src=") and "dst=/exercise" in item)
                workspace = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
                (workspace / "solution.py").write_text("def solve(numbers, target):\n    return []\n")
            return "{}"

        with patch.object(agent, "run_container", side_effect=transport):
            if MINI:
                agent.generate(self.args, self.root / "candidate.py", "Fixture problem")
            else:
                agent.invoke(self.args, candidate, context, "Fixture coder")
                agent.invoke(self.args, candidate, context, "Fixture reviewer", reviewer=True)
                agent.invoke(self.args, candidate, context, "", reviewer=True,
                             debug_tool=("read", {"filePath": "/candidate/main.py"}))
        for options, _ in calls[:1 if MINI else 2]:
            self.assertIn("--network=host", options)
        if not MINI:
            options = calls[-1][0]
            self.assertIn("--network=none", options)
            self.assertNotIn("--network=host", options)
            self.assertTrue(any("dst=/candidate,readonly" in item for item in calls[1][0]))

    def test_objective_execution_keeps_network_disabled(self):
        import evaluator
        source = self.root / ("solution.py" if MINI else "main.py")
        source.write_text("raise AssertionError('NEVER EXECUTE ON HOST')\n")
        context = self.root / "trusted"
        context.mkdir()
        if MINI:
            with patch.object(evaluator, "run_container", return_value="[]") as run:
                evaluator.evaluate(source, docker=self.args.docker)
        else:
            project = {"rubric": {"required_files": ["main.py"], "standard_library_only": True},
                       "smoke_test": "Fixture smoke"}
            with patch.object(agent, "run_container", return_value="") as run:
                result = evaluator.objective_checks(self.root, project, context, docker=self.args.docker)
                self.assertTrue(result["feasible"])
        self.assertIn("--network=none", run.call_args.args[0])
        self.assertNotIn("--network=host", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()

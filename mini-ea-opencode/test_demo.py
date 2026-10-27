"""Offline checks. The fake OpenCode is trusted test code; candidates never run."""

import argparse
import contextlib
import io
import json
import os
import random
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import demo
import evaluator


class FitnessTests(unittest.TestCase):
    def test_valid_fitness_and_public_example(self):
        answers = [[1, 2, 3], [1, 2], [1, 2], [2, 3], [2, 3], [], []]
        answers += [[] for _ in evaluator.CASES[7:]]
        result = evaluator.score_answers(answers)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["fitness"], 70)
        self.assertEqual(result["public"], {"indices": [1, 2, 3], "values": [7, 15, 8], "sum": 30, "target": 30})
        self.assertEqual(evaluator.score_answers([[] for _ in evaluator.CASES])["fitness"], 0)

    def test_invalid_answers_have_no_fitness(self):
        for answer in (None, {}, (1, 2), [True], [False], [1.0], ["1"], [1, 1], [-1], [6], [0, 4]):
            with self.subTest(answer=answer):
                result = evaluator.score_answers([answer] + [[] for _ in evaluator.CASES[1:]])
                self.assertFalse(result["feasible"])
                self.assertIsNone(result["fitness"])
        for answers in (None, {}, [], [[]]):
            self.assertFalse(evaluator.score_answers(answers)["feasible"])

    def test_expanded_suite_ceiling_is_attainable(self):
        answers = [[1, 2, 3], [1, 2], [1, 2], [2, 3], [2, 3], [], [],
                   [2], [0, 1, 2], [2, 6], [], list(range(17)), [0, 2, 9, 15],
                   list(range(12)), [1, 2], [0, 1, 4, 8], [6, 13, 15]]
        info = evaluator.benchmark_info()
        result = evaluator.score_answers(answers)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["fitness"], 784)
        self.assertEqual(info["fitness_ceiling"], result["fitness"])
        self.assertEqual(info["case_count"], 17)
        self.assertEqual(len(info["evaluator_sha256"]), 64)
        self.assertEqual(max(len(numbers) for numbers, _ in evaluator.CASES), 32)
        self.assertEqual(max(target for _, target in evaluator.CASES), 1_000_000_000)

    def test_ceiling_matches_independent_exhaustive_small_case_oracle(self):
        rng = random.Random(1234)
        cases = [([rng.randrange(31) for _ in range(rng.randrange(9))], rng.randrange(1, 81))
                 for _ in range(40)]
        expected = 0
        for numbers, target in cases:
            sums = [sum(value for i, value in enumerate(numbers) if mask & (1 << i))
                    for mask in range(1 << len(numbers))]
            expected += max(total for total in sums if total <= target)
        with patch.object(evaluator, "CASES", cases):
            self.assertEqual(evaluator.benchmark_info()["fitness_ceiling"], expected)

    def test_simple_greedy_witness_has_room_for_improvement(self):
        # Fixed answer witnesses, not a supplied implementation or fallback parent.
        answers = [[4, 3], [0], [0], [2, 3], [2, 3], [], [], [2], [0, 1, 2],
                   [2, 6], [], list(range(17)), [18, 0, 9, 25], list(range(12)),
                   [2, 1], [0, 1, 2], [15, 14, 5]]
        result = evaluator.score_answers(answers)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["fitness"], 771)
        self.assertLess(result["fitness"], evaluator.benchmark_info()["fitness_ceiling"])

    def test_input_mutation_protocol_is_infeasible(self):
        result = evaluator.score_answers([{"error": "input_modified"}] + [[] for _ in evaluator.CASES[1:]])
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["fitness"])
        self.assertIn("input numbers were modified", result["reason"])

    def test_selection_is_strict_and_feasibility_first(self):
        good = lambda value: {"feasible": True, "fitness": value}
        bad = {"feasible": False, "fitness": None}
        for child, parent, expected in ((good(11), good(10), True), (good(10), good(10), False),
                                        (good(9), good(10), False), (bad, good(0), False),
                                        (good(0), bad, True), (bad, bad, False)):
            self.assertEqual(demo.better(child, parent), expected)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prompts = []
        # A real executable fake CLI, invoked in a fresh exercise workspace.
        self.fake = self.root / "fake-opencode"
        self.fake.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\nfrom pathlib import Path\n"
            "assert sys.argv[1] == 'run'\n"
            "n = int(os.environ['FAKE_ATTEMPT'])\n"
            "p = Path('solution.py')\n"
            "log = Path(os.environ['FAKE_LOG'])\n"
            "with log.open('a') as f: f.write(repr(p.read_text() if p.exists() else None) + '\\n')\n"
            "if str(n) == os.environ.get('FAKE_FAIL'): sys.exit(1)\n"
            # This source would raise immediately if executed or imported on host.
            "p.write_text(\"raise RuntimeError('CANDIDATE EXECUTED ON HOST')\\n\" + f'def solve(numbers, target): return [{n}]\\n')\n"
        )
        self.fake.chmod(0o755)
        self.args = argparse.Namespace(
            generations=4, results=self.root / "results",
            opencode_timeout=5, opencode_memory_mb=2048, model="fake/model", opencode_auth=self.root / "no-auth",
            config={}, api_key="test-secret-not-real", show_source=True,
        )

    def exercise(self, *, fail=None, unchanged=False, answers=None):
        attempts = []
        selected_docker = getattr(self.args, "docker", None)
        scores = iter(answers or ([], [0], [], [0]))
        evaluated = []

        def fake_generation(options, command, *, timeout, environment, memory_mb, diagnostics_path, docker=None):
            self.assertIs(docker, selected_docker)
            self.assertEqual(command[:2], ["opencode", "run"])
            self.assertEqual(memory_mb, 2048)
            self.assertIn("--pure", command)
            self.assertIn("fake/model", command)
            self.assertIn("OPENCODE_DISABLE_MODELS_FETCH=true", options)
            config_text = next(option.split("=", 1)[1] for option in options
                               if option.startswith("OPENCODE_CONFIG_CONTENT="))
            config = json.loads(config_text)
            permissions = config["agent"]["exercise"]["permission"]
            self.assertEqual(permissions["edit"], {
                "*": "deny", "/exercise/solution.py": "allow",
                "exercise/solution.py": "allow", "solution.py": "allow",
            })
            self.assertEqual(permissions["*"], "deny")
            self.assertEqual(permissions["external_directory"], "deny")
            self.assertTrue(diagnostics_path.name.endswith(".diagnostics.txt"))
            self.assertNotIn("test-secret-not-real", " ".join(options + command))
            self.assertNotIn("/trusted", " ".join(options))
            self.assertNotIn("evaluator.py", command[-1])
            self.prompts.append(command[-1])
            self.assertNotIn("fitness ceiling", command[-1])
            self.assertNotIn(str(evaluator.CASES[12][0]), command[-1])
            mounts = [value for value in options if value.startswith("type=bind")]
            self.assertEqual(len(mounts), 1)
            self.assertFalse(any("dst=/usr/local/bin/opencode" in mount for mount in mounts))
            mount = next(value for value in mounts if "dst=/exercise" in value)
            workspace = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
            self.assertEqual(sorted(p.name for p in workspace.iterdir()),
                             ["problem.md", "solution.py"] if attempts else ["problem.md"])
            before = (workspace / "solution.py").read_text() if attempts else None
            attempts.append(before)
            env = {**environment, "FAKE_ATTEMPT": str(len(attempts) - 1), "FAKE_LOG": str(self.root / "fake.log")}
            if fail is not None:
                env["FAKE_FAIL"] = str(fail)
            result = subprocess.run([str(self.fake), *command[1:]], cwd=workspace, env=env, timeout=timeout)
            if result.returncode:
                raise RuntimeError("Fake OpenCode failed")
            if unchanged and before:
                (workspace / "solution.py").write_text(before + "# only formatting changed\n")
            return ""

        def fake_evaluation(options, command, *, timeout, input_text, docker=None):
            self.assertIs(docker, selected_docker)
            self.assertIn("--network=none", options)
            self.assertIn("--interactive", options)
            self.assertEqual(command, ["python", "-I", "-B", "/trusted/evaluator.py", "--worker"])
            self.assertEqual(json.loads(input_text), [[n, t] for n, t in evaluator.CASES])
            self.assertEqual(timeout, evaluator.EVALUATION_SECONDS)
            mounts = [value for value in options if value.startswith("type=bind")]
            self.assertEqual(len(mounts), 2)
            self.assertFalse(any("dst=/usr/local/bin/opencode" in mount for mount in mounts))
            self.assertTrue(all(value.endswith(",readonly") for value in mounts))
            self.assertNotIn("OPENAI_API_KEY", " ".join(options))
            evaluated.append(mounts)
            return json.dumps([next(scores)] + [[] for _ in evaluator.CASES[1:]])

        output = io.StringIO()
        with patch.object(demo, "prerequisites"), patch.object(demo, "run_container", side_effect=fake_generation), \
             patch.object(evaluator, "run_container", side_effect=fake_evaluation), contextlib.redirect_stdout(output):
            run_dir = demo.run_demo(self.args)
        return run_dir, json.loads((run_dir / "results.json").read_text()), attempts, evaluated, output.getvalue()

    def test_rootless_context_reaches_generation_and_evaluation(self):
        self.args.docker = demo.Docker("rootless", rootless=True)
        _, result, _, evaluated, _ = self.exercise(fail=4)
        self.assertTrue(evaluated)
        self.assertEqual(result["selected_candidate"], "offspring_1")

    def test_current_parent_selection_saving_and_no_host_execution(self):
        run_dir, result, parents, evaluated, output = self.exercise(fail=4)
        self.assertEqual([row["status"] for row in result["history"]],
                         ["initial", "accepted", "rejected", "rejected", "generation_failed"])
        self.assertIsNone(parents[0])
        self.assertEqual(parents[1], (run_dir / "parent_0.py").read_text())
        self.assertEqual(parents[2:], [(run_dir / "offspring_1.py").read_text()] * 3)
        self.assertEqual(result["selected_candidate"], "offspring_1")
        self.assertEqual(result["best"]["fitness"], 12)
        self.assertEqual((run_dir / "best_solution.py").read_text(), parents[2])
        self.assertFalse((run_dir / "offspring_4.py").exists())
        self.assertEqual(len(evaluated), 4)
        self.assertIn("Public example: indices=[0], values=[12], sum=12, target=30", output)
        self.assertIn("OFFSPRING ACCEPTED!", output)
        self.assertNotIn("test-secret-not-real", (run_dir / "results.json").read_text() + output)
        self.assertEqual(result["settings"]["case_count"], 17)
        self.assertEqual(result["settings"]["fitness_ceiling"], 784)
        self.assertEqual(result["settings"]["initial_strategy"], "auto")
        self.assertFalse(result["fitness_ceiling_reached"])
        # Fresh directories: identical settings cannot silently reuse old artifacts.
        second, *_ = self.exercise(fail=4)
        self.assertNotEqual(run_dir, second)

    def test_initial_failure_saves_history_without_a_fake_best(self):
        with self.assertRaisesRegex(RuntimeError, "Initial generation failed"):
            self.exercise(fail=0)
        run_dir = next(self.args.results.iterdir())
        self.assertFalse((run_dir / "best_solution.py").exists())
        self.assertIsNone(json.loads((run_dir / "results.json").read_text())["best"])

    def test_unchanged_offspring_does_not_replace_parent(self):
        self.args.generations = 1
        run_dir, result, _, evaluated, _ = self.exercise(unchanged=True)
        self.assertEqual(result["selected_candidate"], "parent_0")
        self.assertEqual(result["history"][1]["status"], "generation_failed")
        self.assertEqual(len(evaluated), 1)
        self.assertEqual((run_dir / "best_solution.py").read_bytes(), (run_dir / "parent_0.py").read_bytes())

    def test_greedy_initialization_restricts_only_the_initial_prompt(self):
        self.args.initial_strategy = "greedy"
        _, result, _, _, _ = self.exercise(fail=4)
        self.assertIn("For classroom initialization ONLY", self.prompts[0])
        self.assertIn("largest-first greedy heuristic", self.prompts[0])
        self.assertTrue(all("For classroom initialization ONLY" not in prompt for prompt in self.prompts[1:]))
        self.assertEqual(result["settings"]["initial_strategy"], "greedy")

    def test_initial_ceiling_is_reported_without_skipping_generations(self):
        info = {**evaluator.benchmark_info(), "fitness_ceiling": 12}
        with patch.object(demo, "benchmark_info", return_value=info):
            _, result, _, _, output = self.exercise(answers=([0], [], [], [], []))
        self.assertTrue(result["fitness_ceiling_reached"])
        self.assertEqual(len(result["history"]), 5)
        self.assertIn("further offspring can only tie or score lower", output)

    def test_accepted_offspring_ceiling_is_reported_and_saved(self):
        info = {**evaluator.benchmark_info(), "fitness_ceiling": 12}
        with patch.object(demo, "benchmark_info", return_value=info):
            _, result, _, _, output = self.exercise(fail=4)
        self.assertTrue(result["fitness_ceiling_reached"])
        self.assertEqual(result["selected_candidate"], "offspring_1")
        self.assertIn("Fitness ceiling reached", output)

    def test_env_loader_never_sources_shell_code(self):
        (self.root / ".env").write_text("IGNORED=private\nexport OPENAI_API_KEY='literal-$(touch hacked)'\n")
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "cwd", return_value=self.root):
            self.assertEqual(demo.openai_key(), "literal-$(touch hacked)")
        self.assertFalse((self.root / "hacked").exists())

    def test_model_loads_from_dotenv_and_environment_takes_priority(self):
        (self.root / ".env").write_text("OPENCODE_MODEL=openai/gpt-4.1-mini\n")
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "cwd", return_value=self.root):
            self.assertEqual(demo.env_value("OPENCODE_MODEL"), "openai/gpt-4.1-mini")
            with patch.dict(os.environ, {"OPENCODE_MODEL": "fake/environment"}):
                self.assertEqual(demo.env_value("OPENCODE_MODEL"), "fake/environment")

    def test_container_prerequisites_preserve_model_priority(self):
        self.args.opencode_config = self.root / "no-config.json"
        flags = "--model --dir --pure --agent --format"
        for cli_model, env_model, configured_model, expected in (
            ("fake/override", "fake/environment", "fake/configured", "fake/override"),
            (None, "fake/environment", "fake/configured", "fake/environment"),
            (None, None, "fake/configured", "fake/configured"),
            (None, None, None, "openai/gpt-4.1-mini"),
        ):
            with self.subTest(expected=expected):
                self.args.model = cli_model
                self.args.opencode_config.write_text(json.dumps({"model": configured_model} if configured_model else {}))
                with patch.object(demo.Docker, "check"), \
                     patch.object(demo.Docker, "run", side_effect=["1.4.3\n", flags]) as probes, \
                     patch.object(demo.subprocess, "run", side_effect=AssertionError("No host CLI")), \
                     patch.object(demo, "openai_key", return_value="test-secret-not-real"), \
                     patch.object(demo, "env_value", side_effect=lambda name: env_model if name == "OPENCODE_MODEL" else None), \
                     contextlib.redirect_stdout(io.StringIO()):
                    demo.prerequisites(self.args)
                self.assertEqual(probes.call_count, 2)
                self.assertFalse(hasattr(self.args, "opencode"))
                self.assertEqual(self.args.model, expected)


class SandboxTests(unittest.TestCase):
    def test_evaluator_check_needs_no_opencode_provider_or_generation(self):
        args = argparse.Namespace(check_evaluator=True, docker_context="classroom")
        with patch.object(demo.Docker, "check"), \
             patch.object(demo.Docker, "run", side_effect=AssertionError("No OpenCode probes")), \
             patch.object(demo, "check_evaluator") as check, \
             patch.object(demo, "generate") as generate, contextlib.redirect_stdout(io.StringIO()):
            demo.run_demo(args)
        check.assert_called_once_with(docker=args.docker)
        generate.assert_not_called()

    def test_check_evaluator_routes_each_fixture_through_the_sandbox(self):
        sentinel = object()
        sources = []

        def fake(candidate, *, docker):
            self.assertIs(docker, sentinel)
            source = candidate.read_text()  # Never execute the fixture here.
            sources.append(source)
            if "append" in source:
                return evaluator.infeasible("Instance 0: input numbers were modified")
            if "True" in source:
                return evaluator.infeasible("Instance 0: return a list")
            return {"feasible": True, "fitness": 0, "reason": ""}

        with patch.object(evaluator, "evaluate", side_effect=fake), contextlib.redirect_stdout(io.StringIO()):
            evaluator.check_evaluator(docker=sentinel)
        self.assertEqual(len(sources), 3)

    def test_check_evaluator_does_not_treat_any_container_failure_as_a_pass(self):
        with patch.object(evaluator, "evaluate", return_value=evaluator.infeasible("Evaluation failed: status 127")), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Evaluator check failed"):
            evaluator.check_evaluator(docker=object())

    def test_diagnostics_recognize_opencode_permission_rule_rejection(self):
        summary = evaluator.diagnostic_summary(
            'PermissionDeniedError: The user has specified a rule which prevents you from using this specific tool call'
        )
        self.assertIn("Permission error observed", summary)

    def test_diagnostic_summary_never_copies_raw_log_content(self):
        output = 'INFO service=config token=PRIVATE_KEY\nERROR service=llm 401 unauthorized Bearer PRIVATE_TOKEN'
        summary = evaluator.diagnostic_summary(output)
        self.assertIn("Configuration initialization observed", summary)
        self.assertIn("Model request observed", summary)
        self.assertIn("Authentication or authorization error observed", summary)
        self.assertNotIn("PRIVATE", summary)
        self.assertNotIn("Bearer", summary)

    def test_killed_container_reports_oom_state_before_cleanup(self):
        for oom in ("true", "false", "unknown"):
            with self.subTest(oom=oom):
                calls = []

                def fake_run(command, **kwargs):
                    calls.append(command)
                    if command[3] == "run":
                        self.assertIn("--memory=2048m", command)
                        self.assertIn("--memory-swap=2048m", command)
                        kwargs["stdout"].write(b"private provider log must not be printed")
                        return subprocess.CompletedProcess(command, 137)
                    if command[3] == "inspect":
                        self.assertIn("{{.State.OOMKilled}}", command)
                        if oom == "unknown":
                            raise subprocess.TimeoutExpired(command, 10)
                        return subprocess.CompletedProcess(command, 0, oom + "\n", "")
                    return subprocess.CompletedProcess(command, 0, b"", b"")

                with patch.object(evaluator.os, "getuid", return_value=1000), \
                     patch.object(evaluator.subprocess, "run", side_effect=fake_run):
                    with self.assertRaises(RuntimeError) as error:
                        evaluator.run_container([], ["opencode", "run", "fake"], timeout=5, memory_mb=2048)
                message = str(error.exception)
                self.assertIn(f"Docker OOMKilled={oom}", message)
                self.assertIn("2048 MiB" if oom == "true" else "SIGKILL", message)
                self.assertNotIn("private provider", message)
                self.assertEqual([call[3] for call in calls], ["run", "inspect", "rm"])

    def test_timeouts_make_candidates_infeasible(self):
        with patch.object(evaluator, "run_container", side_effect=RuntimeError("timed out")):
            self.assertFalse(evaluator.evaluate(Path("untrusted.py"))["feasible"])

    def test_container_limits_and_cleanup_after_timeout(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if command[3] == "run":
                for flag in ("--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                             "--cpus=1", "--memory=512m", "--pids-limit=64", "--pull=never"):
                    self.assertIn(flag, command)
                self.assertEqual(command[command.index("--user") + 1], "1000:1000")
                self.assertIs(kwargs["preexec_fn"], evaluator.limit_output)
                kwargs["stdout"].write(b"INFO service=provider PRIVATE_KEY\n")
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return subprocess.CompletedProcess(command, 0, b"", b"")

        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(evaluator.os, "getuid", return_value=1000), \
             patch.object(evaluator.os, "getgid", return_value=1000), \
             patch.object(evaluator.subprocess, "run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            diagnostics = Path(temporary) / "parent_0.diagnostics.txt"
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                evaluator.run_container(["--network=none"], ["python", "--version"], timeout=1,
                                        diagnostics_path=diagnostics)
            self.assertIn("Provider initialization observed", diagnostics.read_text())
            self.assertNotIn("PRIVATE_KEY", diagnostics.read_text())
        self.assertEqual(calls[1][:5], ["docker", "--context", "default", "rm", "-f"])
        self.assertEqual(calls[0][calls[0].index("--name") + 1], calls[1][5])

    def test_missing_docker_fails_before_generation(self):
        args = argparse.Namespace()
        with patch.object(demo.shutil, "which", return_value=None), patch.object(demo, "generate") as generation:
            with self.assertRaisesRegex(RuntimeError, "Docker is required"):
                demo.prerequisites(args)
            generation.assert_not_called()

    def test_docker_permission_error_is_distinct_from_missing_image(self):
        failed = subprocess.CompletedProcess([], 1, b"", b"permission denied")
        with patch.object(demo.shutil, "which", return_value="/usr/bin/docker"), \
             patch.object(demo.os, "getuid", return_value=1000), \
             patch.object(demo.subprocess, "run", return_value=failed) as command:
            with self.assertRaisesRegex(RuntimeError, "Docker is not accessible"):
                demo.prerequisites(argparse.Namespace())
            self.assertEqual(command.call_count, 1)

    def test_missing_image_gives_build_instructions(self):
        success = subprocess.CompletedProcess([], 0, b"29.8.1", b"")
        missing = subprocess.CompletedProcess([], 1, b"", b"No such image")
        with patch.object(demo.shutil, "which", return_value="/usr/bin/docker"), \
             patch.object(demo.os, "getuid", return_value=1000), \
             patch.object(demo.subprocess, "run", side_effect=[success, missing]):
            with self.assertRaisesRegex(RuntimeError, "Docker image mini-ea-opencode:1.4.3.*Build the image"):
                demo.prerequisites(argparse.Namespace())


class WorkerProtocolTests(unittest.TestCase):
    def run_trusted_fixture(self, solve):
        """Only a hand-written trusted fake module runs here, never candidate source."""
        spec = SimpleNamespace(loader=Mock())
        with patch.object(evaluator.Path, "exists", return_value=True), \
             patch.object(evaluator.importlib.util, "spec_from_file_location", return_value=spec), \
             patch.object(evaluator.importlib.util, "module_from_spec", return_value=SimpleNamespace(solve=solve)), \
             patch.object(sys, "stdin", io.StringIO('[[[1, 2], 3]]')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            evaluator.worker()
        return json.loads(output.getvalue())

    def test_worker_checks_input_contents_and_types(self):
        def mutate(numbers, target, mode):
            if mode == "append":
                numbers.append(0)
            elif mode == "clear":
                numbers.clear()
            elif mode == "reverse":
                numbers.reverse()
            else:
                numbers[0] = True if mode == "bool" else 1.0
            return []

        for mode in ("append", "clear", "reverse", "bool", "float"):
            with self.subTest(mode=mode):
                result = self.run_trusted_fixture(lambda numbers, target: mutate(numbers, target, mode))
                self.assertEqual(result, [{"error": "input_modified"}])

    def test_worker_preserves_type_validation_and_suppresses_prints(self):
        for answer, expected in (([], [[]]), ([0], [[0]]), ((0,), [None]), ([True], [None])):
            def fixture(numbers, target):
                print("Trusted fixture debug output")
                return answer
            with self.subTest(answer=answer):
                self.assertEqual(self.run_trusted_fixture(fixture), expected)

    def test_worker_requires_docker_before_attempting_candidate_import(self):
        with patch.object(evaluator.Path, "exists", return_value=False), \
             patch.object(evaluator.importlib.util, "spec_from_file_location") as load:
            with self.assertRaisesRegex(RuntimeError, "must run inside Docker"):
                evaluator.worker()
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Instructor-owned scoring. Candidate Python runs ONLY in a Docker worker."""

import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
# The sandbox worker needs only this evaluator, never the host Docker adapter.
if __name__ != "__main__":
    from docker_runtime import Docker, CleanupError

IMAGE = "mini-ea-opencode:1.4.3"
EVALUATION_SECONDS = 15
OUTPUT_LIMIT = 1024 * 1024

# INSTRUCTOR: change these fixed instances and score_answers for another exercise.
# The first case is public. The remaining cases never go to OpenCode.
CASES = [
    ([12, 7, 15, 8, 19, 6], 30),
    ([8, 7, 6], 13),  # Largest-first greedy gets 8; 7 + 6 gets 13.
    ([6, 4, 3], 7),
    ([0, 0, 5, 5, 11], 10),
    ([40, 2, 9, 1], 10),
    ([], 5),
    ([0, 0, 0], 1),
    ([2, 4, 6], 7),  # The target is not always attainable.
    ([5, 5, 5, 5, 0, 0], 15),
    ([0, 0, 8, 0, 8, 0, 3, 3], 11),
    ([31, 32, 33, 34], 30),  # Every item is too large.
    ([1] * 32, 17),  # Larger n exposes unbounded exhaustive search.
    ([91, 17, 65, 34, 8, 72, 56, 49, 23, 87, 12, 61, 39, 44, 5, 78,
      29, 53, 96, 14, 68, 37, 81, 21, 59, 46, 3, 75, 32, 63, 19, 84], 321),
    ([3, 0, 7, 11, 5, 2, 0, 19, 23, 13, 17, 29], 1_000_000_000),
    ([1_000_000_000, 2, 18, 11, 0, 7], 20),
    ([13, 13, 12, 12, 9, 9, 7, 7, 6, 6, 0, 0], 41),
    ([6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66], 155),
]


def benchmark_info():
    """Trusted exact ceiling for this bounded suite, never a candidate fallback.

    Reachable sums stay small for these instructor-owned cases. Keep that true
    when changing the suite. No reference solver or private inputs go to OpenCode.
    """
    ceiling = 0
    for numbers, target in CASES:
        reachable = {0}
        for number in numbers:
            reachable |= {total + number for total in reachable if total + number <= target}
        ceiling += max(reachable)
    return {"case_count": len(CASES), "fitness_ceiling": ceiling,
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluation_timeout_seconds": EVALUATION_SECONDS, "cpu_limit_seconds": 10}


def infeasible(reason):
    return {"feasible": False, "fitness": None, "reason": reason, "public": None}


def score_answers(answers):
    """Pure trusted calculation: never import or execute candidate code here."""
    if type(answers) is not list or len(answers) != len(CASES):
        return infeasible("Expected one answer per instance")
    fitness = 0
    public = None
    for case, ((numbers, target), indices) in enumerate(zip(CASES, answers)):
        if type(indices) is dict and indices == {"error": "input_modified"}:
            return infeasible(f"Instance {case}: input numbers were modified")
        if type(indices) is not list:
            return infeasible(f"Instance {case}: return a list")
        if any(type(i) is not int for i in indices):  # bool is NOT an index.
            return infeasible(f"Instance {case}: indices must be integers, not booleans")
        if len(set(indices)) != len(indices):
            return infeasible(f"Instance {case}: duplicate index")
        if any(i < 0 or i >= len(numbers) for i in indices):
            return infeasible(f"Instance {case}: index out of range")
        values = [numbers[i] for i in indices]
        total = sum(values)
        if total > target:
            return infeasible(f"Instance {case}: target exceeded")
        fitness += total
        if case == 0:
            public = {"indices": indices, "values": values, "sum": total, "target": target}
    return {"feasible": True, "fitness": fitness, "reason": "", "public": public}


def limit_output():
    # Limit the Docker client's output file too, not just files inside Docker.
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT, OUTPUT_LIMIT))


def diagnostic_summary(output):
    """Only fixed labels leave the raw log: no keys, URLs, prompts, or tool content."""
    output = output.lower()
    markers = [
        ("Configuration initialization observed", ("service=config",)),
        ("Dependency/package installation observed", ("service=npm", "service=bun", "installing")),
        ("Provider initialization observed", ("service=provider",)),
        ("Model request observed", ("service=llm",)),
        ("Tool activity observed", ("service=tool",)),
        ("Authentication or authorization error observed", ("unauthorized", "invalid api key", "authentication", "401", "403")),
        ("Quota or rate-limit error observed", ("insufficient_quota", "rate limit", "429")),
        ("Network/TLS error observed", ("fetch failed", "econn", "enotfound", "certificate", "connection refused")),
        ("Permission error observed", ("permission denied", "permission rejected", "permission requested",
                                       "permissiondeniederror", "rule which prevents")),
        ("Model unavailable error observed", ("modelnotfound", "model_not_found", "model not found")),
        ("An error was logged", ("error",)),
    ]
    labels = [label for label, terms in markers if any(term in output for term in terms)]
    return "OpenCode diagnostic summary (raw logs omitted):\n" + "\n".join(
        "- " + label for label in labels or ["No recognized startup or error messages were captured"]
    ) + "\n"


def run_container(options, command, *, docker=None, **kwargs):
    """One bounded container, using the explicitly selected Docker context."""
    return (docker or Docker()).run(options, command, summarize=diagnostic_summary,
                                    output_limiter=limit_output, **kwargs)


def evaluate(candidate, *, docker=None):
    """Send inputs to the sandbox; validate its answers on the trusted host."""
    try:
        output = run_container([
            "--network=none", "--interactive", "--ulimit", "cpu=10:10",
            "--ulimit", "fsize=1048576:1048576",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m,mode=1777",
            "--mount", f"type=bind,src={candidate.resolve()},dst=/candidate/solution.py,readonly",
            "--mount", f"type=bind,src={Path(__file__).resolve()},dst=/trusted/evaluator.py,readonly",
        ], ["python", "-I", "-B", "/trusted/evaluator.py", "--worker"],
            timeout=EVALUATION_SECONDS, input_text=json.dumps(CASES), docker=docker)
        return score_answers(json.loads(output))
    except (RuntimeError, ValueError, OSError) as error:
        # Error text never enters the variation prompt; only feasible/fitness do.
        return infeasible(f"Evaluation failed: {error}")


def check_evaluator(*, docker):
    """Three explicit trusted fixtures, executed only in Docker; no OpenCode."""
    fixtures = [
        ("valid empty selection", "return []", True),
        ("input mutation rejected", "numbers.append(0); return []", False),
        ("boolean index rejected", "return [True]", False),
    ]
    with tempfile.TemporaryDirectory(prefix="mini-ea-evaluator-check-") as temporary:
        candidate = Path(temporary) / "solution.py"
        for label, body, feasible in fixtures:
            candidate.write_text("def solve(numbers, target):\n    " + body + "\n")
            result = evaluate(candidate, docker=docker)
            if (result["feasible"] is not feasible
                    or (feasible and result["fitness"] != 0)
                    or (label == "input mutation rejected" and "input numbers were modified" not in result["reason"])
                    or (label == "boolean index rejected" and "return a list" not in result["reason"])):
                raise RuntimeError(f"Evaluator check failed: {label}: {result['reason']}")
            print(f"  PASS: {label}", flush=True)
    print("Evaluator checks passed. Trusted fixtures only; no OpenCode, credentials or model inference.")


def worker():
    """Untrusted side of the protocol. This function runs inside Docker only."""
    if not Path("/.dockerenv").exists():
        raise RuntimeError("The worker must run inside Docker")
    cases = json.load(sys.stdin)
    answers = []
    # Ordinary print/debug output must not corrupt the answer protocol.
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        spec = importlib.util.spec_from_file_location("solution", "/candidate/solution.py")
        solution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution)
        for original, target in cases:
            numbers = original.copy()
            answer = solution.solve(numbers, target)
            # Check final values AND types: 1 -> True or 1.0 is still mutation.
            if numbers != original or any(type(number) is not int for number in numbers):
                answers.append({"error": "input_modified"})
            # Preserve Python's list/int semantics across JSON (tuples are invalid).
            elif type(answer) is not list or any(type(i) is not int for i in answer):
                answers.append(None)
            else:
                answers.append(answer.copy())
    print(json.dumps(answers))


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        sys.exit("Use python demo.py; this file belongs to the instructor.")
    worker()

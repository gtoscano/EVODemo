"""A small (1+1) evolutionary loop. No third-party Python packages."""

import argparse
import ast
from datetime import datetime
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit

from evaluator import CleanupError, IMAGE, benchmark_info, check_evaluator, evaluate, run_container
from docker_runtime import Docker

HERE = Path(__file__).resolve().parent
MAX_GENERATIONS = 20


def env_value(name):
    """Read one setting without sourcing .env or evaluating shell expressions."""
    if os.environ.get(name):
        return os.environ[name]
    # Accept .env alongside the project or in the directory it was launched from.
    for path in dict.fromkeys([Path.cwd() / ".env", HERE / ".env", HERE.parent / ".env"]):
        if path.is_file():
            for line in path.read_text().splitlines():
                key, sep, value = line.removeprefix("export ").partition("=")
                if sep and key.strip() == name:
                    words = shlex.split(value, comments=True)
                    if len(words) != 1:
                        raise RuntimeError(f"{name} in .env must have one value")
                    return words[0]
    return None


def openai_key():
    return env_value("OPENAI_API_KEY")


def provider_config(path):
    # Copy only provider/model settings, never user plugins, tools, MCP or rules.
    if not path.is_file():
        if path.with_suffix(".jsonc").is_file():
            raise RuntimeError("Use --opencode-config with a plain JSON copy of your JSONC provider/model settings")
        return {}
    try:
        config = json.loads(path.read_text())
        if type(config) is not dict:
            raise ValueError("Not an object")
        return {key: config[key] for key in ("model", "small_model", "provider") if key in config}
    except (ValueError, TypeError):
        raise RuntimeError("OpenCode config must be a JSON object (JSONC is not supported)") from None


def configure_local_provider(args):
    """An explicit local endpoint needs no cloud credentials."""
    args.local_llm_url = (getattr(args, "local_llm_url", None)
                          or env_value("OPENCODE_BASE_URL"))
    args.opencode_network = (getattr(args, "opencode_network", None)
                              or env_value("OPENCODE_NETWORK") or "bridge")
    if args.opencode_network not in ("bridge", "host"):
        raise RuntimeError("OPENCODE_NETWORK must be bridge or host")
    if not args.local_llm_url:
        if args.model.startswith("local/"):
            raise RuntimeError("Set OPENCODE_BASE_URL for a local/model")
        return
    url = urlsplit(args.local_llm_url)
    if (url.scheme not in ("http", "https") or not url.hostname
            or url.username or url.password or url.query or url.fragment):
        raise RuntimeError("OPENCODE_BASE_URL must be an HTTP(S) endpoint without credentials/query/fragment")
    if not args.model.startswith("local/") or not args.model.removeprefix("local/"):
        raise RuntimeError("With OPENCODE_BASE_URL, set OPENCODE_MODEL=local/SERVER_MODEL_ID")
    model = args.model.split("/", 1)[1]
    # Replace inherited providers so all model calls use this endpoint.
    args.config = {"provider": {"local": {
        "npm": "@ai-sdk/openai-compatible", "name": "Local model server",
        "options": {"baseURL": args.local_llm_url.rstrip("/"), "apiKey": "not-required"},
        "models": {model: {"name": model}},
    }}}
    if (args.opencode_network == "host" and args.docker.rootless
            and not getattr(args, "check_opencode", False)):
        version = getattr(args.docker, "server_version", "")
        match = re.match(r"(\d+)\.(\d+)", version)
        if not match or tuple(map(int, match.groups())) < (29, 5):
            raise RuntimeError("Rootless host networking needs Docker Engine 29.5+; "
                               "upgrade or use a server reachable with bridge networking")
    print(f"Local provider: {args.local_llm_url}; OpenCode network: {args.opencode_network}", flush=True)


def prerequisites(args):
    args.docker = Docker(getattr(args, "docker_context", "default"), image=getattr(args, "evaluation_image", IMAGE))
    args.docker.check()
    if getattr(args, "check_evaluator", False):
        return  # Only trusted fixtures in Docker; no OpenCode/config/auth needed.
    if getattr(args, "check_docker", False):
        if not args.docker.rootless:
            args.docker.check_mounts()
        print("Docker checks passed. No OpenCode, credentials, or model inference.", flush=True)
        return
    # Probe the exact executable used for coding/review, never a host installation.
    # These calls have no mounts, provider settings, credentials or network.
    options = ["--network=none", "--tmpfs", "/tmp:rw,nosuid,size=256m,mode=1777",
               "--workdir=/tmp", "--env", "HOME=/tmp/home",
               "--env", "XDG_DATA_HOME=/tmp/data", "--env", "XDG_CONFIG_HOME=/tmp/config",
               "--env", "XDG_CACHE_HOME=/tmp/cache", "--env", "OPENCODE_DISABLE_MODELS_FETCH=true",
               "--env", "OPENCODE_DISABLE_PROJECT_CONFIG=true",
               "--env", 'OPENCODE_CONFIG_CONTENT={"autoupdate":false}']
    try:
        version, help_text = [args.docker.run(options, command, timeout=30,
                                            memory_mb=getattr(args, "opencode_memory_mb", 2048))
                              for command in (["opencode", "--version"], ["opencode", "run", "--help"])]
    except (RuntimeError, OSError):
        raise RuntimeError(f"OpenCode could not run inside {args.docker.image}. Rebuild the image using README.md; no host OpenCode installation is needed.") from None
    if not all(
        flag in help_text for flag in ("--model", "--dir", "--pure", "--agent", "--format")
    ) or not re.fullmatch(r"1\.[0-9]+\.[0-9]+", version.strip()):
        raise RuntimeError("Container OpenCode must be 1.x with run --pure/--dir/--model/--agent/--format; rebuild the documented image")
    print(f"OpenCode version: {version.strip()} (container image: {args.docker.image})")
    args.config = provider_config(args.opencode_config)
    args.model = (args.model or env_value("OPENCODE_MODEL") or args.config.get("model")
                  or provider_config(HERE / "opencode.json").get("model"))
    if not args.model or "/" not in args.model:
        raise RuntimeError("Choose a provider/model with --model or in opencode.json")
    print(f"Model: {args.model}", flush=True)
    configure_local_provider(args)
    args.api_key = (None if args.local_llm_url or getattr(args, "check_opencode", False)
                    else openai_key())
    if getattr(args, "check_opencode", False):
        return  # Debug tool checks do not need provider credentials or inference.
    if not args.local_llm_url and not args.api_key and not args.opencode_auth.is_file():
        raise RuntimeError("Configure OpenCode auth or set OPENAI_API_KEY in .env (see README)")



def generate(args, destination, problem, parent=None, parent_result=None):
    """OpenCode sees only a fresh exercise directory, never the archive/evaluator."""
    with tempfile.TemporaryDirectory(prefix="mini-ea-work-") as temporary:
        workspace = Path(temporary)
        workspace.chmod(0o755)
        (workspace / "problem.md").write_text(problem)
        if parent is not None:
            shutil.copyfile(parent, workspace / "solution.py")  # ACTUAL current parent.
        permissions = {
            "*": "deny", "external_directory": "deny",
            "read": {"*": "deny", "/exercise/problem.md": "allow", "/exercise/solution.py": "allow"},
            # OpenCode checks writes relative to its worktree ("/" without Git).
            "edit": {"*": "deny", "/exercise/solution.py": "allow",
                     "exercise/solution.py": "allow", "solution.py": "allow"},
        }
        config = {
            **args.config, "model": args.model, "small_model": args.model,
            "autoupdate": False, "share": "disabled", "snapshot": False, "formatter": False,
            "lsp": False, "permission": permissions,
            "agent": {"exercise": {"mode": "primary", "description": "Implement this exercise", "steps": 6,
                                   "permission": permissions}},
        }
        options = [
            "--network=" + getattr(args, "opencode_network", "bridge"),
            # OpenCode caches and its session database need more room than a candidate.
            "--ulimit", "fsize=67108864:67108864",
            "--tmpfs", "/tmp:rw,nosuid,size=256m,mode=1777", "--workdir=/exercise",
            "--mount", f"type=bind,src={workspace},dst=/exercise",
            "--env", "HOME=/tmp/home", "--env", "XDG_DATA_HOME=/tmp/data",
            "--env", "XDG_CONFIG_HOME=/tmp/config", "--env", "XDG_CACHE_HOME=/tmp/cache",
            # Use the CLI's bundled model catalog, avoiding another startup network request.
            "--env", "OPENCODE_DISABLE_MODELS_FETCH=true",
            "--env", "OPENCODE_CONFIG_CONTENT=" + json.dumps(config),
        ]
        if args.api_key:
            # Forward by name; keep the secret out of subprocess command arguments.
            options += ["--env", "OPENAI_API_KEY"]
        if not getattr(args, "local_llm_url", None) and args.opencode_auth.is_file():
            # Only the provider auth file, not the user's home or OpenCode sessions.
            options += ["--mount", f"type=bind,src={args.opencode_auth.resolve()},dst=/provider-auth.json,readonly"]
        prompt = (
            "Read /exercise/problem.md. Implement /exercise/solution.py, using only the standard library. "
            "Save the file with the write/edit tools; do not merely print code. "
            "Do not run code or use shell commands. Include a short comment explaining your strategy. "
        )
        if parent is None:
            prompt += "Create the initial solution from the specification and public example."
            if getattr(args, "initial_strategy", "auto") == "greedy":
                prompt += (
                    " For classroom initialization ONLY, use a largest-first greedy heuristic: "
                    "consider items in descending value order and select each item if it fits. "
                    "Do not use exact search, dynamic programming or repair/swap improvements for this "
                    "initial parent. Later offspring may use any algorithm. This deliberate starting "
                    "strategy does not change the objective or fitness rules."
                )
        else:
            prompt += (
                "solution.py contains the actual current parent. Read it first. "
                "Revise it into a meaningfully different algorithmic strategy that attempts to improve "
                "fitness, not just formatting or renaming. Keep the required function interface. "
                f"Parent public summary: feasible={parent_result['feasible']}, "
                f"aggregate fitness={parent_result['fitness']}. "
                "Private instances and evaluation rules are unavailable. Do not invent test results."
            )
        command = ["opencode", "run", "--pure", "--dir", "/exercise", "--agent", "exercise",
                   "--print-logs", "--log-level", "INFO", "--title", "Subset-sum exercise"]
        if args.model:
            command += ["--model", args.model]
        if not getattr(args, "local_llm_url", None) and args.opencode_auth.is_file():
            # Copy to a writable, disposable data directory; never write host auth.
            command = ["sh", "-c", "mkdir -p /tmp/data/opencode && cp /provider-auth.json /tmp/data/opencode/auth.json && exec \"$@\"",
                       "opencode-start", *command]
        # Docker inherits only this explicit provider key; no other host environment is mounted.
        environment = dict(os.environ)
        if getattr(args, "local_llm_url", None):
            environment.pop("OPENAI_API_KEY", None)
        if args.api_key:
            environment["OPENAI_API_KEY"] = args.api_key
        run_container(options, command + [prompt], timeout=args.opencode_timeout, environment=environment,
                      docker=getattr(args, "docker", None), memory_mb=args.opencode_memory_mb, diagnostics_path=destination.with_suffix(".diagnostics.txt"))
        solution = workspace / "solution.py"
        if not solution.exists():
            raise RuntimeError("OpenCode did not create solution.py; see this attempt's diagnostics file")
        if solution.is_symlink() or not solution.is_file() or solution.stat().st_size > 65536:
            raise RuntimeError("OpenCode must write a regular solution.py of at most 64 KiB")
        source = solution.read_text()
        destination.write_text(source)
        destination.chmod(0o644)
        if parent is not None:
            try:
                unchanged = ast.dump(ast.parse(source)) == ast.dump(ast.parse(parent.read_text()))
            except SyntaxError:
                unchanged = source == parent.read_text()
            if unchanged:
                raise RuntimeError("OpenCode left the implementation unchanged")


def better(offspring, parent):
    return offspring["feasible"] and (
        not parent["feasible"] or offspring["fitness"] > parent["fitness"]
    )


def show(label, result):
    print(f"\n{label}\n  Feasible: {'Yes' if result['feasible'] else 'No'}\n  Fitness: {result['fitness']}")
    if result["reason"]:
        print(f"  {result['reason']}")


def save(run_dir, history, parent, parent_result, requested, settings=None):
    if parent is not None:
        shutil.copyfile(parent, run_dir / "best_solution.py")
    (run_dir / "results.json").write_text(json.dumps({
        "generations_requested": requested, "generations_attempted": max(0, len(history) - 1),
        "selected_candidate": parent.stem if parent else None,
        "settings": settings,
        "fitness_ceiling_reached": bool(settings and parent_result and parent_result["feasible"]
                                       and parent_result["fitness"] == settings["fitness_ceiling"]),
        "best": parent_result, "history": history,
    }, indent=2) + "\n")


def run_demo(args):
    print("=" * 40 + "\n       (1+1) EVOLUTIONARY CODER\n" + "=" * 40)
    print("Objective: Maximize subset sum without exceeding target", flush=True)
    prerequisites(args)
    if getattr(args, "check_docker", False):
        return
    if getattr(args, "check_evaluator", False):
        return check_evaluator(docker=args.docker)
    settings = {**benchmark_info(), "initial_strategy": getattr(args, "initial_strategy", "auto")}
    problem = (HERE / "problem.md").read_text()  # INSTRUCTOR: public task specification.
    args.results.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=args.results))
    print(f"Results: {run_dir}")
    print(f"Evaluation: {settings['case_count']} fixed instances; fitness ceiling {settings['fitness_ceiling']}")
    print(f"Initial strategy: {settings['initial_strategy']}\n\nCreating initial solution with OpenCode...", flush=True)
    history = []
    parent = run_dir / "parent_0.py"
    try:
        generate(args, parent, problem)
    except (RuntimeError, OSError, ValueError) as error:
        history.append({"candidate": "parent_0", "status": "generation_failed", "reason": str(error),
                        "fitness": None, "feasible": None, "accepted": False})
        save(run_dir, history, None, None, args.generations, settings)
        raise RuntimeError(f"Initial generation failed: {error}. History: {run_dir / 'results.json'}") from None
    parent_fitness = evaluate(parent, docker=getattr(args, "docker", None))
    show("PARENT", parent_fitness)
    if parent_fitness["feasible"] and parent_fitness["fitness"] == settings["fitness_ceiling"]:
        print("Fitness ceiling reached for this suite; further offspring can only tie or score lower.")
    history.append({"candidate": parent.stem, "parent": None, "status": "initial", "accepted": True, **parent_fitness})
    save(run_dir, history, parent, parent_fitness, args.generations, settings)

    # The (1+1) EA: exactly one incumbent and one new variation at a time.
    for generation in range(1, args.generations + 1):
        print(f"\nGeneration {generation}\nCreating a variation of the parent...", flush=True)
        offspring = run_dir / f"offspring_{generation}.py"
        try:
            generate(args, offspring, problem, parent, parent_fitness)
        except (RuntimeError, OSError, ValueError) as error:
            print(f"Generation failed: {error}\nParent survives!")
            history.append({"candidate": offspring.stem, "parent": parent.stem, "status": "generation_failed",
                            "reason": str(error), "feasible": None, "fitness": None, "accepted": False})
            save(run_dir, history, parent, parent_fitness, args.generations, settings)
            continue
        offspring_fitness = evaluate(offspring, docker=getattr(args, "docker", None))
        show("OFFSPRING", offspring_fitness)
        accepted = better(offspring_fitness, parent_fitness)
        print(f"\nComparison:\n  Parent: {parent_fitness['fitness']}\n  Offspring: {offspring_fitness['fitness']}")
        history.append({"candidate": offspring.stem, "parent": parent.stem, "accepted": accepted,
                        "status": "accepted" if accepted else "rejected", **offspring_fitness})
        if accepted:
            parent = offspring
            parent_fitness = offspring_fitness
            print("Selection: OFFSPRING ACCEPTED!")
            if parent_fitness["fitness"] == settings["fitness_ceiling"]:
                print("Fitness ceiling reached for this suite; further offspring can only tie or score lower.")
        else:
            print("Selection: Parent survives! (ties retain the parent)")
        save(run_dir, history, parent, parent_fitness, args.generations, settings)

    print("\nAttempt          Fitness   Feasible   Outcome")
    for entry in history:
        feasible = {True: "Yes", False: "No", None: "N/A"}[entry["feasible"]]
        print(f"{entry['candidate']:<17}{str(entry['fitness']):<10}{feasible:<11}{entry['status']}")
    print("\n" + "-" * 40 + "\nFINAL BEST SOLUTION\n" + "-" * 40)
    print(f"Feasible: {parent_fitness['feasible']}\nFitness: {parent_fitness['fitness']}\n"
          f"Fitness ceiling: {settings['fitness_ceiling']}\n"
          f"Generations: {args.generations}\nSelected candidate: {parent.stem}\nSource: {run_dir / 'best_solution.py'}")
    if parent_fitness["public"]:
        public = parent_fitness["public"]
        print(f"Public example: indices={public['indices']}, values={public['values']}, "
              f"sum={public['sum']}, target={public['target']}")
    if not parent_fitness["feasible"]:
        print("No feasible solution was found. The saved source is the final infeasible incumbent.")
    if args.show_source:
        print("\n" + parent.read_text())  # Display text only; NEVER execute it on the host.
    return run_dir


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, choices=range(MAX_GENERATIONS + 1), default=1, metavar="0..20")
    parser.add_argument("--docker-context", default="default", help="Docker context (default: default; use rootless for your rootless daemon)")
    parser.add_argument("--model", help="Optional configured provider/model")
    parser.add_argument("--initial-strategy", choices=("auto", "greedy"), default="auto",
                        help="Request a greedy initial parent for teaching; offspring remain unrestricted")
    parser.add_argument("--opencode-timeout", type=int, choices=range(1, 601), default=180, metavar="1..600")
    parser.add_argument("--opencode-memory-mb", type=int, choices=range(512, 8193), default=2048,
                        metavar="512..8192", help="OpenCode container memory limit in MiB (default: 2048)")
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    parser.add_argument("--opencode-config", type=Path, default=config_home / "opencode/opencode.json")
    parser.add_argument("--opencode-auth", type=Path, default=data_home / "opencode/auth.json")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check-docker", action="store_true", help="Check Docker mounts and rootless limits without OpenCode or inference")
    checks.add_argument("--check-evaluator", action="store_true", help="Run trusted evaluator fixtures in Docker without OpenCode or inference")
    parser.add_argument("--show-source", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_demo(arguments())
    except (CleanupError, RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"Error: {error}") from None

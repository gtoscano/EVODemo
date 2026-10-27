# A tiny (1+1) evolutionary coder

The objective is to select indices whose values have the largest sum without
exceeding a target. Fitness is the sum across fixed instances, **only if every
answer is valid**. The evaluator, not the LLM, decides fitness.

The strengthened suite has **17 instances and maximum aggregate fitness 784**.
It covers zeros, duplicates, unattainable targets, 32-item lists, and a large
target with a small attainable sum. The original seven-case suite had a maximum
of 70; scores from those older runs are not directly comparable to new runs.

The **parent** is the current solution. The **offspring** is a variation of that
actual parent's source. OpenCode is the variation operator: it proposes a different
algorithm. A feasible offspring survives only if its fitness is strictly larger,
or the parent is infeasible. Ties and two infeasible candidates retain the parent.
Variation can lose useful behavior and make a solution worse. The best fitness
cannot decrease once feasible, but a generation need not improve it, and finding
the optimum is not guaranteed. An initially optimal parent is a valid demonstration.

```text
OpenCode creates parent → evaluate
                            ↓
               copy current parent → OpenCode varies it
                            ↑                  ↓
                     select survivor ← evaluate offspring
                            ↓ (after the last generation)
                       save best source
```

## Setup

Use Python 3.10+ and Docker accessible as your normal user. uv is optional. No pip packages
or host OpenCode executable are needed. The supplied image contains **OpenCode
1.4.3** and Python 3.12. Startup checks version/help inside that image, with no
network or credentials. OpenCode 2 is not supported by the current tool policy.

Linux/WSL is the existing target. Container packaging prepares for macOS with
[Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/), but
**macOS is untested**. Do not treat it as validated classroom support yet.

On a fresh Ubuntu installation (if Docker is already installed, keep that setup):

```bash
sudo apt-get update
sudo apt-get install -y python3 python-is-python3 docker.io curl
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Log out and back in so the group change applies. Run the demo as your normal user.
Alternatively follow [Docker's installation instructions](https://docs.docker.com/engine/install/).

```bash
# From mini-ea-opencode/:
docker --context default version
docker --context default build -t mini-ea-opencode:1.4.3 .
```

The shared Python/OpenCode image is identical in all three coding demos: build
it once per Docker daemon/context. The Dockerfile selects Linux arm64 or
x64-baseline according to the build architecture. `.dockerignore` excludes
credentials, results and source from the build context. Image construction checks
the CLI without model inference. It downloads tools once; candidate execution
remains network-free. Rebuild after updating the Dockerfile.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Keep an already configured provider/model. Existing `opencode auth login`
credentials can be reused if you have them; host OpenCode is optional. Otherwise
put `OPENAI_API_KEY=your-key` in `.env` in this
directory, its parent, or the launch directory. Set the default model alongside it:

```dotenv
OPENAI_API_KEY=your-key
OPENCODE_MODEL=openai/gpt-4.1-mini
```

Existing environment variables take priority over `.env` values. Only
`OPENAI_API_KEY`, `OPENCODE_MODEL`, `OPENCODE_BASE_URL`, and
`OPENCODE_NETWORK` are read; `.env` is never sourced as shell code.
Never commit `.env`. See the [OpenCode CLI documentation](https://opencode.ai/docs/cli/).

Only `model`, `small_model`, and `provider` from `~/.config/opencode/opencode.json`
and provider auth are copied. Model selection uses `--model` first, then
`OPENCODE_MODEL`, then a configured global model, then this project's small
`opencode.json`. The chosen model is printed
before generation. An API key supplies credentials, not a model selection.

The project default is **`openai/gpt-4.1-mini`**, selected for this simple exercise.
Actual run cost depends on the provider's current rates and total token use.
See the [official model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
Change `OPENCODE_MODEL` in `.env` or pass `--model` to use something else. JSONC, plugins, MCP,
credential commands, other environment
variables, and host file references are not forwarded. `--opencode-config` accepts
another plain JSON file. Provider URLs must be reachable **from Docker**.

## Local llama.cpp server on a Linux Spark

Create your own `.env.local` beside `demo.py` with the following settings.
Environment files are excluded from GitHub, along with virtual environments,
uv files/caches, and saved results. The repository contains no API keys.

```dotenv
OPENCODE_MODEL=local/qwen3.8-27b
OPENCODE_BASE_URL=http://127.0.0.1:10000/v1
OPENCODE_NETWORK=host
```

`local/` is the demo's provider prefix; the exact model ID sent to the server
is `qwen3.8-27b`. No OpenAI key is needed. With this endpoint configured, the
controller replaces inherited provider settings and does not read/forward the
OpenAI key or mount the OpenCode authentication file. An incompatible cloud
model override causes an error rather than falling back to a paid provider.
The existing `.env` file is unchanged.

The demo itself automatically reads `.env`, not `.env.local`; the explicit
`uv run --env-file .env.local` option loads the latter. Already exported environment variables take
priority, so clear conflicting `OPENCODE_MODEL`, `OPENCODE_BASE_URL`, or
`OPENCODE_NETWORK` exports if the printed settings differ.

The loopback URL assumes llama-server runs on the **same workstation**.
`OPENCODE_NETWORK=host` shares the host network with OpenCode's coding/review
containers, allowing them to reach that loopback listener without exposing the
model server on the LAN. It also allows those containers to reach other host
network services. This setting is explicit and applies only to OpenCode:
CLI/tool self-checks and candidate execution retain `--network=none`.

For rootless Docker, real host networking requires **Docker Engine 29.5+**;
the demo rejects older rootless versions for this configuration. See
[Docker's rootless networking notes](https://docs.docker.com/engine/security/rootless/troubleshoot/#historical-limitations).
An existing `mini-ea-opencode:1.4.3` image can be reused. Pull the updated source
from GitHub as described below and create your local settings file on the Spark.
Updating only `.env.local` is insufficient if the Python controller is still an
older copy. No image rebuild is needed for this provider update. A fresh Docker
daemon still needs the image built once:

```bash
docker --context rootless build -t mini-ea-opencode:1.4.3 .
```

Verify Docker and the evaluator without inference:

```bash
python3 -B -m unittest -q
python3 demo.py --docker-context rootless --check-docker
python3 demo.py --docker-context rootless --check-evaluator
```

Before inference, test connectivity using the already built image:

```bash
docker --context rootless run --rm --pull=never --network=host \
  --read-only --cap-drop=ALL --security-opt=no-new-privileges --user 1000:1000 \
  mini-ea-opencode:1.4.3 python -I -c \
  'import json, urllib.request; data=json.load(urllib.request.urlopen("http://127.0.0.1:10000/v1/models", timeout=10)); print([m["id"] for m in data["data"]])'
```

This trusted probe lists model metadata; it does not generate text. Expect the
list to contain `qwen3.8-27b`. If it fails, check the Docker version, rootless
context, and that the server is running on this workstation on port 10000.
The successful host-side `/v1/models` request alone does not check Docker access.

Run the demo after those checks pass:

```bash
uv run --env-file .env.local python demo.py \
  --docker-context rootless --initial-strategy greedy \
  --generations 1 --show-source --opencode-timeout 600
```

Without uv, pass the same settings directly to Python:

```bash
OPENCODE_MODEL=local/qwen3.8-27b \
OPENCODE_BASE_URL=http://127.0.0.1:10000/v1 \
OPENCODE_NETWORK=host \
python3 demo.py --docker-context rootless --initial-strategy greedy \
  --generations 1 --show-source --opencode-timeout 600
```

The evolution commands perform real local inference. Local routing and
permissions are covered by offline tests; live inference on your Spark has not
been tested during development.

Fitness is still calculated by the deterministic Python evaluator; it does not use an LLM.

The terminal prints `Model: local/qwen3.8-27b`, the endpoint/network settings,
each candidate's measured fitness, and the survivor decision. An improvement is
not guaranteed. The final output names the selected candidate and the exact
fresh run directory under `results/`. That directory contains
`best_solution.py` (the winning source), `results.json` (fitness and selection
history), and the parent/offspring source files, including rejected candidates.
`--show-source` prints the winner as well. Inspect saved source with `cat` or
an editor; execute generated code only through the Docker evaluator.

## Get or update the code from GitHub

Clone `gtoscano/EVODemo` on the Spark (the command names the local folder `EvoDemo`):

```bash
git clone https://github.com/gtoscano/EVODemo.git EvoDemo
cd EvoDemo/mini-ea-opencode
```

An existing Git clone can be updated from that directory:

```bash
git pull --ff-only
```

A folder copied with `cp` or `scp` is not a Git clone. Keep that folder and
clone into a separate location; transfer any needed local settings yourself.
No API keys or virtual environments are downloaded. Python's standard library
is enough to run this project. If you choose to use uv, install it separately;
do not copy an x86 virtual environment onto the ARM64 Spark.

Because `/home` is shared through NFS, a clone in your home is visible from the
other lab workstations. Each workstation/account still needs its own working
rootless Docker daemon and images. The loopback model URL always refers to the
workstation where the command runs. Private repositories also require GitHub
read access for each student.

## Run

Run from the demo directory:

```bash
cd mini-ea-opencode
uv run python demo.py --generations 1
uv run python demo.py --generations 1 --show-source
uv run python demo.py --generations 5
uv run python demo.py --generations 3 --model provider/model --show-source
```

### Demonstrate improvement from a simple starting point

The default `--initial-strategy auto` lets OpenCode choose the initial algorithm.
It may immediately reach the maximum. For an explicit teaching initialization:

```bash
uv run python demo.py --initial-strategy greedy --generations 3 --show-source
```

This asks OpenCode to write a largest-first greedy initial parent. Later offspring
may use any strategy and still start from the actual surviving source. The same
17 cases, fitness calculation and strict selection apply throughout the run.
The controller supplies no handwritten solution and never changes a score to
show improvement. The initialization is a request to the model, not proof it
used that method; inspect the saved source. Improvement remains uncertain.

The controller computes the suite's exact fitness ceiling using trusted code
outside the candidate. When a survivor reaches 784 it explains that further
offspring can only tie or score lower. All requested generations still run.
The ceiling and private instances are not included in OpenCode's prompts.

### Regular or rootless Docker

Omitting `--docker-context` selects the context named **`default`**, normally the
regular system daemon. To use your rootless daemon, select its context explicitly:

```bash
docker --context rootless build -t mini-ea-opencode:1.4.3 .
uv run python demo.py --docker-context rootless --check-docker
uv run python demo.py --docker-context rootless --generations 1 --show-source
```

For regular Docker, the inference-free check is:

```bash
uv run python demo.py --check-docker
uv run python demo.py --check-evaluator
```

`--check-docker` needs no OpenCode installation or provider credentials. It runs
trusted fixtures to verify writable exercise mounts and read-only context mounts.
Rootless mode also verifies cgroup v2/systemd and actual CPU, memory, swap and
process limits before any model request. Missing delegation is an error.

`--check-evaluator` runs three explicitly trusted fixtures inside Docker: an
empty selection must be feasible with fitness zero, an input-mutating function
must be rejected, and boolean indices must be rejected. No OpenCode, API key or
model inference is involved. It uses the same worker, validation and limits as
normal evaluation. These fixtures are diagnostic code, never evolutionary parents
or fallback solutions. For rootless Docker add `--docker-context rootless`.

The program prints the selected context and detects the daemon's actual mode;
the context name alone does not determine it. Every Docker call, including
inspection and cleanup, explicitly uses that context, regardless of the CLI's
currently selected context or `DOCKER_HOST`. Custom context names work too.

`docker_runtime.py` keeps execution details outside the evolutionary loop.
Regular Docker retains the host UID/GID. Rootless Docker runs OpenCode and
candidates as container UID/GID 1000, using private copies of mounted files under
a host directory with mode 0700. Bounded, network-free trusted helpers prepare
and recover ownership of those copies; only those helpers use container UID 0,
which is unprivileged on the rootless host. They never execute candidate code.
Safe writable results are copied back after container cleanup. Saved parents,
provider auth files, evaluator source and their host permissions are not changed.
Evaluator mounts remain read-only. Uncertain cleanup stops the run.

See [Docker's rootless setup guide](https://docs.docker.com/engine/security/rootless/)
for lab prerequisites. No Docker socket or host home is mounted inside a container.
On ARM64, build the image natively with that machine's daemon; do not force amd64
images. No ARM64 OpenCode installation on the host is required.

For a future macOS check, use `docker context ls` to find Docker Desktop's context
(often `desktop-linux`). Build with `docker --context NAME build ...` and pass
`--docker-context NAME` to the demo. Docker Desktop does not require the Linux
rootless Docker setup. This update includes no macOS testing.

Offline tests cover both modes. Live rootless operation still needs verification
on the lab machines; this development session cannot access the Docker socket.

### Python environment

**A virtual environment is optional.** This demo uses only Python's standard
library, with no Python dependencies to install or lock. It has no `pyproject.toml`,
so `uv run` does not automatically create a project `.venv`. It uses an available
virtual environment in this directory or a parent directory; otherwise it uses
the discovered Python interpreter. See [uv run](https://docs.astral.sh/uv/reference/cli/#uv-run).

To create a separate environment for this demo, run this once from this directory
(skip it if `.venv` already exists):

```bash
uv venv --python python3
```

Afterward, `uv run` uses `.venv` without manual activation. Keep it if convenient;
removing `.venv` leaves the source and saved results intact. Exclude `.venv` when
sharing the project, and create a new one on the destination machine rather than
copying it. Each demo can have its own environment.

You can also run `python3 demo.py --generations 1` directly with Python 3.10+.
Creating or activating a Python environment does not load `.env`; the demo itself
loads the supported variables as described above.

Running the demo makes real provider calls, which may incur charges. By default
there is one generation; at most 20 are allowed.

| Container | Memory limit | Execution timeout |
| --- | --- | --- |
| OpenCode generation | **2048 MiB (2 GiB)** by default | 180 seconds per call |
| Candidate evaluation | **512 MiB** | 15 seconds per candidate |

All 17 instances share the candidate's 10-second CPU limit and 15-second Docker
wall-time limit. A timeout or out-of-memory failure makes that candidate infeasible.
No points are awarded for speed. Increasing `--opencode-timeout` gives the coding
agent more time; it does not change candidate evaluation limits or the ceiling.

Use `--opencode-memory-mb` (512–8192 MiB) or `--opencode-timeout` (1–600 seconds)
to change the generation limits. These options do not change evaluation limits.

The terminal shows feasibility, numerical fitness, comparisons, selection, an
attempt table, the final source path, and recorded indices/values/sum/target for
the public example. `--show-source` prints the source without executing it.
Fitness values depend on generated code; improvement is never fabricated.
Each variation is limited to six agent steps, and the same chosen model handles
auxiliary requests. The CLI uses its bundled model catalog without downloading a
new catalog at startup.

Each fresh `results/run-<timestamp>-<unique>/` contains `parent_0.py`, generated
`offspring_1.py`, further offspring, `best_solution.py`, and `results.json` with
fitness, public-example results, parent identities, and selection history.
New results also record `settings` with the initial strategy, case count, exact
fitness ceiling, evaluator source SHA-256 and evaluation limits, plus
`fitness_ceiling_reached`. Existing run files are never rewritten when the suite
changes. Do not compare aggregate scores across different evaluator fingerprints.
Each attempt also saves `<candidate>.diagnostics.txt`, even after a timeout.
These contain fixed progress/error labels; raw logs, credentials, and prompts are
not copied into them.
Failed generation has no fitness and leaves the parent intact; missing files
are not replaced with handwritten code. Initial generation failure stops the run.
If all candidates are infeasible, the final report says so explicitly.

## Troubleshooting

### Every offspring ties or loses

First compare the parent fitness with the printed ceiling. A parent already at
784 cannot improve its sum on this suite. Different algorithms may tie, and the
parent correctly survives. This demonstrates elitism, rather than a failure of
variation. Use the greedy-initialization command above when you want room for
improvement. Broader tests do not guarantee that an unrestricted parent will
start below the ceiling.

### Docker access or missing image

If the demo stops at the Docker check, no OpenCode/model call has been made.
First run these commands in the same terminal where you run the demo:

```bash
docker --context default version
docker --context default image inspect mini-ea-opencode:1.4.3
```

`docker version` must show a **Server** section. If the daemon is stopped, run
`sudo systemctl start docker`. On Docker Desktop, start the application instead.
For socket `permission denied` on a normal Linux installation, follow
[Docker's user-access setup](https://docs.docker.com/engine/install/linux-postinstall/):
`sudo usermod -aG docker "$USER"`, then log out and back in. Do not run the demo
with sudo or make the socket world-writable. In a restricted IDE/agent terminal,
Docker may remain blocked; use a normal host terminal with working Docker access.

If Docker works but the image is missing, run
`docker --context default build -t mini-ea-opencode:1.4.3 .` from this directory.
For rootless or another context, use its name in both the build and demo commands.
Then retry `uv run python demo.py --generations 1 --show-source`. The Python virtual
environment does not supply Docker, and the demo never falls back to host execution.

### OpenCode compatibility

OpenCode **1.4.3** is pinned in the Dockerfile. The compatibility check runs
`opencode --version` and `opencode run --help` inside the image and captures both
output streams. If the executable is missing or incompatible, rebuild the image.
The controller never falls back to host OpenCode. The old `--opencode PATH`
option has been removed; `--opencode-config` and `--opencode-auth` still select
provider configuration and credentials.

### Container exits with status 137

Status **137** usually indicates a SIGKILL. Memory exhaustion is a possible cause,
but the exit status alone does not confirm it. The demo now checks Docker's
`OOMKilled` state before removing the failed container.

The previous 512 MiB generation limit may have been too small for OpenCode.
With the current **2048 MiB default**, first retry the normal command:

```bash
uv run python demo.py --generations 1 --show-source
```

If the new error reports **`OOMKilled=true`**, retry with 4096 MiB on a host with
sufficient available memory:

```bash
uv run python demo.py --generations 1 --show-source --opencode-memory-mb 4096
```

Evaluation remains limited to **512 MiB**. If `OOMKilled=false` or `unknown`, memory
exhaustion is unconfirmed; check host/Docker diagnostics before increasing the limit.
See [Docker's memory-limit documentation](https://docs.docker.com/engine/containers/resource_constraints/).

### OpenCode reaches its time limit

A timeout does not establish whether startup, networking, or the model stalled.
Check the printed model and the failed attempt's `parent_0.diagnostics.txt` (or the
offspring's corresponding file) in its results directory. The summary can identify
configuration/dependency/provider initialization, a model request, tool activity,
and common authentication, quota, or network errors. Labels are hints, not a full trace.

The current demo explicitly selects a model and avoids the startup catalog fetch.
Retry normally first. If a model request is progressing but needs more time, use
`--opencode-timeout 300` (maximum 600 seconds). Increasing the timeout does not fix
an invalid key, insufficient quota, or a Docker network problem. A timed-out attempt
may already have incurred provider charges; the demo does not retry it automatically.

### OpenCode exits without creating solution.py

OpenCode can exit successfully even when an agent tool failed. Check the attempt's
diagnostic summary. The current policy allows the exact absolute and worktree-relative
forms of `solution.py`: OpenCode 1.4.3 checks reads with absolute paths but checks
writes relative to its worktree (which is `/` in this disposable directory without Git).
An earlier policy allowed only the absolute form and could block the write tool.
Use the current `demo.py` and retry; no model or credential change is needed for that fix.

## Isolation and changing the exercise

OpenCode sees only a disposable task/current-parent workspace. It can edit only
`solution.py`; shell, subagents, and external file tools are denied. Only generation
has networking/credentials. Saved parents and scoring are not mounted there.
Unchanged syntax trees are rejected; algorithmic difference is requested, not proven.

Evaluation uses non-root Docker with no network/credentials, read-only candidate
and evaluator mounts, a read-only root filesystem, resource/output/time limits,
and cleanup. Only this container imports candidate Python. Trusted host code
validates indices and computes fitness. There is no host execution fallback.
The worker also checks that each input list has the same final values and integer
types after `solve`; mutation makes the candidate infeasible. This catches, for
example, clearing/reordering the input or replacing an integer with a boolean.

Instructors change `problem.md` (the public specification) and `CASES` /
`score_answers` in `evaluator.py` (the private objective). For a different function
interface, also adjust `worker`, `benchmark_info` and the final public-example display. Freeze those
rules for an entire run. The evolutionary loop is the short `for` loop in `demo.py`.

## Offline tests

```bash
uv run python -m unittest -v
```

Tests invoke a fake OpenCode executable and fake Docker responses. They cover
validation, strict selection, current-parent copying, failure preservation, fresh
archives, best-source saving, sandbox flags and cleanup, and absence of host
candidate execution. They need neither Docker nor a provider and make no LLM calls.
Development: **48 offline tests passed** using
`python3 -B -m unittest -q`, including confirmed OOM kills,
other status-137 exits, and unavailable OOM diagnostics.
The installed OpenCode write tool was also checked with a trusted, comment-only
fixture: the original absolute-only policy denied the write; the corrected policy
allowed it. This check made no inference call and executed no candidate Python.
Actual container execution was not tested because this development session could
not access the Docker socket; no real model inference was launched.
Packaging tests also cover network-free in-image CLI probes, incompatible/missing
executables, no host fallback, pinned Linux archive selection, and exclusion of
the host executable mount. The new image build and macOS execution remain untested.

The Docker-context tests cover the default and explicit context, routing to coder
and evaluator, daemon-mode detection, rootless mount staging, credential/source
preservation, effective-limit rejection and cleanup failures. All Docker replies
in these tests are scripted; no generated code executes on the host.

The added checks verify an attainable ceiling with explicit optimal witnesses
and an independent exhaustive oracle on small fixtures, input contents/types,
initial-only greedy prompting, ceiling reporting without early termination, and
the model-free evaluator self-check. Worker-protocol unit tests use an explicitly
trusted fake module; generated source is never imported on the host. The live
Docker self-check still needs to run in your normal terminal. This Python-only
update reuses the existing image; no image rebuild is required.

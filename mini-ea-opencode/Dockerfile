# syntax=docker/dockerfile:1
# Shared runtime: build on the target architecture, without a host OpenCode install.
FROM python:3.12-slim-bookworm

# The release and archive naming follow the official OpenCode 1 installer.
# Use the x64 baseline build so older lab CPUs do not need AVX2.
RUN python - <<'PY'
import io
from pathlib import Path
import platform
import tarfile
import urllib.request

architecture = {"x86_64": "x64-baseline", "aarch64": "arm64"}.get(platform.machine())
if architecture is None:
    raise SystemExit("This image supports Linux amd64 and arm64 only")
url = ("https://github.com/anomalyco/opencode/releases/download/v1.4.3/"
       f"opencode-linux-{architecture}.tar.gz")
with urllib.request.urlopen(url, timeout=120) as response:
    archive_bytes = response.read()
with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
    members = [entry for entry in archive.getmembers() if entry.name in ("opencode", "./opencode")]
    if len(members) != 1 or not members[0].isfile():
        raise SystemExit("The OpenCode archive must contain a regular executable")
    member = members[0]
    destination = Path("/usr/local/bin/opencode")
    destination.write_bytes(archive.extractfile(member).read())
    destination.chmod(0o755)
PY

ENV OPENCODE_DISABLE_MODELS_FETCH=true OPENCODE_DISABLE_PROJECT_CONFIG=true
# No provider configuration or inference is used by these build-time checks.
RUN test "$(opencode --version)" = "1.4.3" && opencode run --help

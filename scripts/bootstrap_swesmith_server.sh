#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bootstrap_swesmith_server.sh PROJECT_ROOT RUNTIME_ROOT" >&2
  exit 2
fi

PROJECT_ROOT="$(realpath "$1")"
RUNTIME_ROOT="$2"
REVISION="9b74ac08118a85c39c356802f7961893af73e07f"

if [[ "$OSTYPE" != linux* ]] || grep -qi microsoft /proc/version; then
  echo "formal bootstrap requires a native Linux server, not Windows or WSL" >&2
  exit 1
fi
if [[ "$RUNTIME_ROOT" != /* ]] || [[ "$RUNTIME_ROOT" == "/" ]]; then
  echo "RUNTIME_ROOT must be a scoped absolute Linux path" >&2
  exit 1
fi

RUNTIME_ROOT="$(realpath -m "$RUNTIME_ROOT")"
case "$RUNTIME_ROOT" in
  /|/home|/root|/usr|/var)
    echo "refusing broad RUNTIME_ROOT: $RUNTIME_ROOT" >&2
    exit 1
    ;;
esac

DEPENDENCIES_ROOT="$RUNTIME_ROOT/dependencies"
REPOSITORY="$DEPENDENCIES_ROOT/SWE-smith-$REVISION"
VENV="$RUNTIME_ROOT/venvs/capo-v2"

# This script never removes or moves an existing path. Unexpected content stops it.
mkdir -p "$DEPENDENCIES_ROOT" "$RUNTIME_ROOT/venvs" "$RUNTIME_ROOT/cache" "$RUNTIME_ROOT/artifacts"
if [[ -e "$REPOSITORY" && ! -d "$REPOSITORY/.git" ]]; then
  echo "refusing non-git path: $REPOSITORY" >&2
  exit 1
fi
if [[ ! -d "$REPOSITORY/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/SWE-bench/SWE-smith.git "$REPOSITORY"
  git -C "$REPOSITORY" checkout --detach "$REVISION"
fi

ACTUAL_REVISION="$(git -C "$REPOSITORY" rev-parse HEAD)"
if [[ "$ACTUAL_REVISION" != "$REVISION" ]]; then
  echo "refusing unexpected SWE-smith revision: $ACTUAL_REVISION" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --copies "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install --editable "$REPOSITORY"
"$VENV/bin/python" -m pip install --editable "$PROJECT_ROOT[data,inference,train]"

"$VENV/bin/python" - <<'PY'
import docker
import swesmith
import coding_agent_policy_opt

print("imports=ok")
print(f"swesmith={swesmith.__file__}")
print(f"docker_sdk={docker.__version__}")
print(f"project={coding_agent_policy_opt.__file__}")
PY

cat <<EOF
bootstrap=complete
venv=$VENV
runtime_root=$RUNTIME_ROOT
export HF_HOME=$RUNTIME_ROOT/cache/huggingface
export HF_DATASETS_CACHE=$RUNTIME_ROOT/cache/huggingface/datasets
export CAPO_ARTIFACTS_ROOT=$RUNTIME_ROOT/artifacts
EOF

#!/bin/zsh
set -euo pipefail

# Get the absolute path of the repo root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH_DEFAULT="$REPO_DIR/AgentOccam/configs/pre-execution_o3_1.yml"

cd "$REPO_DIR"

export OPENAI_API_KEY= ### Enter your OpenAI API key here ###
export GEMINI_API_KEY= ### Enter your Gemini API key here ###

# Optional config path argument: absolute or relative to repo
EXTRA_ARGS=()
if [ "${1-}" != "" ]; then
  if [[ "$1" = /* ]]; then
    CONFIG_PATH="$1"
  else
    CONFIG_PATH="$REPO_DIR/$1"
  fi
  shift || true
  if [ "$#" -gt 0 ]; then
    EXTRA_ARGS=($@)
  fi
else
  CONFIG_PATH="$CONFIG_PATH_DEFAULT"
fi

# Pre-checks
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi
if [ ! -f "agent_runner.py" ]; then
  echo "Missing agent_runner.py in $REPO_DIR"
  exit 1
fi

# Use the environment's Python directly to avoid conda activator issues
# Resolve Python interpreter (prefer $PY_BIN, then active env, then PATH)
if [ "${PY_BIN-}" != "" ] && [ -x "$PY_BIN" ]; then
  :
elif [ "${VIRTUAL_ENV-}" != "" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
  PY_BIN="$VIRTUAL_ENV/bin/python3"
elif [ "${CONDA_PREFIX-}" != "" ] && [ -x "$CONDA_PREFIX/bin/python3" ]; then
  PY_BIN="$CONDA_PREFIX/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="$(command -v python)"
else
  PY_BIN=""
fi

if [ "$PY_BIN" = "" ] || [ ! -x "$PY_BIN" ]; then
  echo "Python interpreter not found. Set PY_BIN to your env's python."
  exit 1
fi

echo "Using python: $PY_BIN"
"$PY_BIN" --version || true
"$PY_BIN" agent_runner.py --config "$CONFIG_PATH" ${EXTRA_ARGS:+${EXTRA_ARGS[@]}}



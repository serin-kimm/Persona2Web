#!/bin/zsh
set -euo pipefail

# Get repo path relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH_DEFAULT="$REPO_DIR/config/on-demand_o3_1.yml"

cd "$REPO_DIR"

# Load environment variables
if [ -f ".env" ]; then
  source .env 2>/dev/null || true
fi

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
    EXTRA_ARGS=("$@")
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

echo "Using config: $CONFIG_PATH"
uv run agent_runner.py --config "$CONFIG_PATH" ${EXTRA_ARGS:+${EXTRA_ARGS[@]}}

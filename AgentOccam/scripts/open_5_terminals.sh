#!/bin/zsh
set -euo pipefail

# Get the absolute path of the launcher script (in the same directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCHER="$SCRIPT_DIR/launch_eval_shard.sh"

# Detect a usable python interpreter to pass through (optional)
if [ "${PY_BIN-}" != "" ] && [ -x "$PY_BIN" ]; then
  PASS_PY_BIN="$PY_BIN"
elif [ "${VIRTUAL_ENV-}" != "" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
  PASS_PY_BIN="$VIRTUAL_ENV/bin/python3"
elif [ "${CONDA_PREFIX-}" != "" ] && [ -x "$CONDA_PREFIX/bin/python3" ]; then
  PASS_PY_BIN="$CONDA_PREFIX/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PASS_PY_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PASS_PY_BIN="$(command -v python)"
else
  PASS_PY_BIN=""
fi

if [ ! -x "$LAUNCHER" ]; then
  echo "Launcher not executable; attempting to set +x"
  chmod +x "$LAUNCHER" || true
fi

# Use AppleScript to open 5 Terminal windows and run the launcher
osascript <<APPLESCRIPT
tell application "Terminal"
  activate
  do script "zsh -lc 'export PY_BIN=\"${PASS_PY_BIN}\"; \"${LAUNCHER}\" --num-shards 5 --shard-index 0'"
  do script "zsh -lc 'export PY_BIN=\"${PASS_PY_BIN}\"; \"${LAUNCHER}\" --num-shards 5 --shard-index 1'"
  do script "zsh -lc 'export PY_BIN=\"${PASS_PY_BIN}\"; \"${LAUNCHER}\" --num-shards 5 --shard-index 2'"
  do script "zsh -lc 'export PY_BIN=\"${PASS_PY_BIN}\"; \"${LAUNCHER}\" --num-shards 5 --shard-index 3'"
  do script "zsh -lc 'export PY_BIN=\"${PASS_PY_BIN}\"; \"${LAUNCHER}\" --num-shards 5 --shard-index 4'"
end tell
APPLESCRIPT

echo "Launched 5 Terminal windows running shards via: $LAUNCHER"



#!/bin/zsh
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$SCRIPT_DIR/launch_eval_shard.sh"

if [ ! -x "$LAUNCHER" ]; then
  echo "Launcher not executable; attempting to set +x"
  chmod +x "$LAUNCHER" || true
fi

# Use AppleScript to open 5 Terminal windows and run the launcher
# Config path is determined by launch_eval_shard.sh's DEFAULT_CONFIG
osascript <<APPLESCRIPT
tell application "Terminal"
  activate
  do script "cd '$REPO_DIR' && zsh -lc '\"$LAUNCHER\" --num-shards 5 --shard-index 0'"
  do script "cd '$REPO_DIR' && zsh -lc '\"$LAUNCHER\" --num-shards 5 --shard-index 1'"
  do script "cd '$REPO_DIR' && zsh -lc '\"$LAUNCHER\" --num-shards 5 --shard-index 2'"
  do script "cd '$REPO_DIR' && zsh -lc '\"$LAUNCHER\" --num-shards 5 --shard-index 3'"
  do script "cd '$REPO_DIR' && zsh -lc '\"$LAUNCHER\" --num-shards 5 --shard-index 4'"
end tell
APPLESCRIPT

echo "Launched 5 Terminal windows running shards via: $LAUNCHER"

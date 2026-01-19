#!/bin/zsh
set -euo pipefail

# Get the absolute path of the repo root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_CONFIG="$REPO_DIR/AgentOccam/configs/pre-execution_o3_1.yml"

cd "$REPO_DIR"

# Args:
#   [CONFIG_PATH] --num-shards N --shard-index I [-- async/concurrency extra args passed to evaluator]
CONFIG_PATH=""
NUM_SHARDS=${NUM_SHARDS-}
SHARD_INDEX=${SHARD_INDEX-}
EXTRA_ARGS=()

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --num-shards)
        NUM_SHARDS="$2"; shift 2 ;;
      --shard-index)
        SHARD_INDEX="$2"; shift 2 ;;
      --)
        shift; while [ $# -gt 0 ]; do EXTRA_ARGS+="$1"; shift; done ;;
      --*)
        # pass-through flags (e.g., --async-run, --concurrency)
        EXTRA_ARGS+="$1"; shift ;;
      *)
        if [ -z "$CONFIG_PATH" ]; then
          CONFIG_PATH="$1"; shift
        else
          EXTRA_ARGS+="$1"; shift
        fi ;;
    esac
  done
}

parse_args "$@"

if [ -z "${CONFIG_PATH}" ]; then
  CONFIG_PATH="$DEFAULT_CONFIG"
fi

# Validate config path
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

# Defaults for shards if not provided
if [ -z "${NUM_SHARDS-}" ]; then NUM_SHARDS=1; fi
if [ -z "${SHARD_INDEX-}" ]; then SHARD_INDEX=0; fi

if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]]; then
  echo "--num-shards and --shard-index must be integers"
  exit 1
fi
if [ "$NUM_SHARDS" -lt 1 ]; then
  echo "--num-shards must be >= 1"; exit 1
fi
if [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -ge "$NUM_SHARDS" ]; then
  echo "--shard-index must be in [0, $((NUM_SHARDS-1))]"; exit 1
fi

# Read relative_task_dir from YAML (simple grep/sed; assumes single-line key). Use POSIX classes for macOS.
RELATIVE_TASK_DIR=$(grep -E '^[[:space:]]*relative_task_dir:' "$CONFIG_PATH" | head -n1 | sed -E 's/.*relative_task_dir:[[:space:]]*"?([^"#]+)"?.*/\1/' | tr -d ' ')
if [ -z "$RELATIVE_TASK_DIR" ]; then
  RELATIVE_TASK_DIR="tasks"
fi

TASK_DIR="$REPO_DIR/config_files/$RELATIVE_TASK_DIR"
if [ ! -d "$TASK_DIR" ]; then
  echo "Task directory not found: $TASK_DIR"
  exit 1
fi

# Prefer task_ids from YAML env.task_ids if provided (single-line list assumed). Fallback to scanning directory.
TASK_IDS_LINE=$(grep -E '^[[:space:]]*task_ids:' "$CONFIG_PATH" | head -n1 || true)
USE_ALL=1
ALL_IDS=()
if [ -n "$TASK_IDS_LINE" ]; then
  TASK_IDS_CSV=$(echo "$TASK_IDS_LINE" | sed -E 's/.*task_ids:[[:space:]]*\[(.*)\].*/\1/' | tr -d ' ' | tr -d '"')
  if [ -n "$TASK_IDS_CSV" ] && [ "$TASK_IDS_CSV" != "all" ] && [ "$TASK_IDS_CSV" != "ALL" ]; then
    # split by comma into array
    ALL_IDS=($(echo "$TASK_IDS_CSV" | tr ',' '\n' | sed '/^$/d'))
    USE_ALL=0
  fi
fi

if [ "$USE_ALL" -eq 1 ]; then
  # Build full task id list (basename without .json), sorted for deterministic sharding
  for f in "$TASK_DIR"/*.json; do
    [ -e "$f" ] || { echo "No tasks found in $TASK_DIR"; exit 1; }
    break
  done
  ALL_IDS=($(ls "$TASK_DIR"/*.json | xargs -n1 basename | sed 's/\.json$//' | sort))
fi

# Optionally filter out already-completed tasks based on existing logs
# Read logging config to locate destination directory
LOGGING_VAL=$(grep -E '^[[:space:]]*logging:' "$CONFIG_PATH" | head -n1 | sed -E 's/.*logging:[[:space:]]*([^ #]+).*/\1/' | tr -d ' ')
LOGDIR_VAL=$(grep -E '^[[:space:]]*logdir:' "$CONFIG_PATH" | head -n1 | sed -E 's/.*logdir:[[:space:]]*\"?([^\"#]+)\"?.*/\1/' | tr -d ' ')
LOGNAME_VAL=$(grep -E '^[[:space:]]*logname:' "$CONFIG_PATH" | head -n1 | sed -E 's/.*logname:[[:space:]]*\"?([^\"#]+)\"?.*/\1/' | tr -d ' ')

REMAINING_IDS=()
if [ -n "$LOGDIR_VAL" ] && [ -n "$LOGNAME_VAL" ] && { [ "$LOGGING_VAL" = "True" ] || [ "$LOGGING_VAL" = "true" ]; }; then
  DSTDIR="$REPO_DIR/$LOGDIR_VAL/$LOGNAME_VAL"
  # helper to check if a task is already logged
  is_task_done() {
    setopt localoptions noglob
    local _id="$1"
    if [ -d "$DSTDIR" ]; then
      # match any pattern ending with _{id}.json or legacy pre_gpt41_{id}.json
      if command find "$DSTDIR" -maxdepth 1 -type f \( -name "*_${_id}.json" -o -name "pre_gpt41_${_id}.json" \) 2>/dev/null | grep -q .; then
        return 0
      fi
    fi
    if [ -d "$DSTDIR/captcha_stopped" ]; then
      if command find "$DSTDIR/captcha_stopped" -maxdepth 1 -type f \( -name "*_${_id}.json" -o -name "pre_gpt41_${_id}.json" \) 2>/dev/null | grep -q .; then
        return 0
      fi
    fi
    return 1
  }
  for tid in "${ALL_IDS[@]}"; do
    if ! is_task_done "$tid"; then
      REMAINING_IDS+=("$tid")
    fi
  done
  if [ ${#REMAINING_IDS[@]} -gt 0 ]; then
    echo "Filtering completed tasks using logs at: $DSTDIR"
    echo "Remaining: ${#REMAINING_IDS[@]} / ${#ALL_IDS[@]}"
    ALL_IDS=("${REMAINING_IDS[@]}")
  else
    echo "No remaining tasks detected; all ${#ALL_IDS[@]} tasks appear completed."
  fi
fi

TOTAL=${#ALL_IDS[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "No task ids to run (empty list)."; exit 1
fi

# Compute shard slice (contiguous chunks, ceiling division)
PER_SHARD=$(( (TOTAL + NUM_SHARDS - 1) / NUM_SHARDS ))
START=$(( SHARD_INDEX * PER_SHARD ))
COUNT=$PER_SHARD
if [ $(( START + COUNT )) -gt "$TOTAL" ]; then
  COUNT=$(( TOTAL - START ))
fi
if [ "$COUNT" -le 0 ]; then
  echo "Shard $SHARD_INDEX/$NUM_SHARDS has no tasks (TOTAL=$TOTAL). Exiting."
  exit 0
fi

SELECTED_IDS=(${ALL_IDS[@]:$START:$COUNT})
echo "Shard selection: total=$TOTAL, num_shards=$NUM_SHARDS, shard_index=$SHARD_INDEX, per_shard=$PER_SHARD, start=$START, count=$COUNT"

# Create temp config overriding env.task_ids with the selected list
TMP_DIR="$REPO_DIR/tmp"
mkdir -p "$TMP_DIR"
BASENAME=$(basename "$CONFIG_PATH")
TMP_CONFIG="$TMP_DIR/shard_${SHARD_INDEX}_of_${NUM_SHARDS}__$BASENAME"
cp "$CONFIG_PATH" "$TMP_CONFIG"

# Construct YAML inline list: ["id1", "id2", ...]
JOINED=$(printf '"%s", ' ${SELECTED_IDS[@]})
JOINED=${JOINED%%, }

# Replace task_ids line (assumes single-line form in original config). Use POSIX classes for macOS.
sed -E -i '' "s/^[[:space:]]*task_ids:.*$/  task_ids: [${JOINED}]/" "$TMP_CONFIG"

echo "Using temp config: $TMP_CONFIG"

# Reuse launch_eval.sh for interpreter detection and API key export
LAUNCHER="$REPO_DIR/scripts/launch_eval.sh"
if [ ! -x "$LAUNCHER" ]; then chmod +x "$LAUNCHER" || true; fi

"$LAUNCHER" "$TMP_CONFIG" ${EXTRA_ARGS:+${EXTRA_ARGS[@]}}



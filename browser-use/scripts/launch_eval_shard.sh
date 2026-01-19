#!/bin/zsh
set -euo pipefail

# Get repo path relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_CONFIG="$REPO_DIR/config/on-demand_o3_1.yml"

cd "$REPO_DIR"

# Args:
#   [CONFIG_PATH] --num-shards N --shard-index I [-- extra args passed to agent_runner]
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
        shift; while [ $# -gt 0 ]; do EXTRA_ARGS+=("$1"); shift; done ;;
      --*)
        # pass-through flags
        EXTRA_ARGS+=("$1"); shift ;;
      *)
        if [ -z "$CONFIG_PATH" ]; then
          CONFIG_PATH="$1"; shift
        else
          EXTRA_ARGS+=("$1"); shift
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

# Read experiment_id from YAML config
EXPERIMENT_ID=$(grep -E '^[[:space:]]*experiment_id:' "$CONFIG_PATH" | head -n1 | sed -E 's/.*experiment_id:[[:space:]]*"?([^"#]+)"?.*/\1/' | tr -d ' ')
if [ -z "$EXPERIMENT_ID" ]; then
  EXPERIMENT_ID="1"
fi

# Task directory: task/personalization_{experiment_id}
TASK_DIR="$REPO_DIR/task/personalization_$EXPERIMENT_ID"
if [ ! -d "$TASK_DIR" ]; then
  echo "Task directory not found: $TASK_DIR"
  exit 1
fi

# Build full task id list (basename without .json), sorted for deterministic sharding
ALL_IDS=()
for f in "$TASK_DIR"/*.json; do
  [ -e "$f" ] || { echo "No tasks found in $TASK_DIR"; exit 1; }
  break
done
ALL_IDS=($(ls "$TASK_DIR"/*.json | xargs -n1 basename | sed 's/\.json$//' | sort))

# Read scheme and model from config for trajectory filename pattern
SCHEME=$(grep -E '^[[:space:]]*scheme:' "$CONFIG_PATH" | head -n1 | sed -E 's/.*scheme:[[:space:]]*"?([^"#]+)"?.*/\1/' | tr -d ' ')
MODEL=$(grep -E '^[[:space:]]*model:' "$CONFIG_PATH" | head -n1 | sed -E 's/.*model:[[:space:]]*"?([^"#]+)"?.*/\1/' | tr -d ' ')

# Scheme short name: on-demand -> "on", pre-execution -> "pre", no-history -> "no"
if [ "$SCHEME" = "on-demand" ]; then
  SCHEME_SHORT="on"
elif [ "$SCHEME" = "pre-execution" ]; then
  SCHEME_SHORT="pre"
elif [ "$SCHEME" = "no-history" ]; then
  SCHEME_SHORT="no"
else
  SCHEME_SHORT="pre"  # fallback
fi

# Simplify model name (remove slashes, dots, dashes)
MODEL_NAME=$(echo "$MODEL" | tr -d '/-.')

# Output directory uses config filename (e.g., "pre-execution_gpt41_1" from "pre-execution_gpt41_1.yml")
CONFIG_NAME=$(basename "$CONFIG_PATH" | sed 's/\.[^.]*$//')
OUT_DIR="$REPO_DIR/result/$CONFIG_NAME"

# Filter out already-completed tasks
REMAINING_IDS=()
if [ -d "$OUT_DIR" ]; then
  for tid in "${ALL_IDS[@]}"; do
    TRAJ_FILE="$OUT_DIR/${SCHEME_SHORT}_${MODEL_NAME}_${tid}.json"
    if [ ! -f "$TRAJ_FILE" ]; then
      REMAINING_IDS+=("$tid")
    fi
  done
  if [ ${#REMAINING_IDS[@]} -gt 0 ]; then
    echo "Filtering completed tasks using logs at: $OUT_DIR"
    echo "Remaining: ${#REMAINING_IDS[@]} / ${#ALL_IDS[@]}"
    ALL_IDS=("${REMAINING_IDS[@]}")
  else
    echo "All ${#ALL_IDS[@]} tasks appear completed."
    exit 0
  fi
fi

TOTAL=${#ALL_IDS[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "No task ids to run (empty list)."; exit 0
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
echo "Selected tasks: ${SELECTED_IDS[*]}"

# Create a temp tasks directory with only the selected task files
TMP_DIR="$REPO_DIR/tmp"
mkdir -p "$TMP_DIR"
SHARD_TASKS_DIR="$TMP_DIR/shard_${SHARD_INDEX}_of_${NUM_SHARDS}_tasks"
rm -rf "$SHARD_TASKS_DIR"
mkdir -p "$SHARD_TASKS_DIR"

# Copy selected task files to shard directory
for tid in "${SELECTED_IDS[@]}"; do
  cp "$TASK_DIR/${tid}.json" "$SHARD_TASKS_DIR/"
done

echo "Created shard tasks directory: $SHARD_TASKS_DIR"
echo "Running ${#SELECTED_IDS[@]} tasks..."

# Run agent_runner with the shard's tasks directory
uv run agent_runner.py --config "$CONFIG_PATH" --tasks-dir "$SHARD_TASKS_DIR" ${EXTRA_ARGS:+${EXTRA_ARGS[@]}}

# Personalization Experiment Runner

Scripts for running Browser-Use agent personalization experiments in parallel.

## Overview

This experiment environment evaluates personalized web browsing tasks using user memory banks. By sharding tasks across 5 terminal windows for parallel execution, experiment time is significantly reduced.

## Directory Structure

```
browser-use/
├── scripts/
│   ├── open_5_terminals.sh     # Launch 5 terminal windows (macOS)
│   ├── launch_eval_shard.sh    # Run tasks for a specific shard
│   └── launch_eval.sh          # Single process execution
├── config/                      # Experiment configuration files
│   ├── on-demand_o3_1.yml      # On-demand scheme (default)
│   ├── pre-execution_o3_1.yml  # Pre-execution scheme
│   └── no-history_o3_1.yml     # No-history scheme
├── task/
│   └── personalization_{id}/   # Task files (per experiment_id)
│       ├── Alex_Garcia_A1.json
│       ├── Alex_Garcia_B1.json
│       └── ...
├── browser_use/
│   └── memory_bank/            # User memory banks
│       ├── Alex_Garcia.json
│       ├── Alex_Lee.json
│       └── ...
├── cache/                       # FAISS index cache
├── result/                      # Experiment results
│   └── {config_name}/
│       ├── on_o3_Alex_Garcia_A1.json
│       └── ...
└── agent_runner.py             # Main execution script
```

## Prerequisites

### Environment Setup

1. **Install Python dependencies**
   ```bash
   uv sync
   ```

2. **Configure environment variables** (`.env` file)
   ```bash
   OPENAI_API_KEY=your-api-key
   # Or other LLM provider keys
   ```

3. **macOS Terminal permissions**
   - System Preferences > Privacy & Security > Automation
   - Grant access permission to Terminal.app

## Experiment Configuration

### Configuration File Structure (`config/*.yml`)

```yaml
experiment_id: "1"                    # Matches task/personalization_{id}
backbone:
  provider: "openai"                  # openai, google, anthropic, browser-use
  model: "o3"                         # Model to use
scheme: "on-demand"                   # on-demand, pre-execution, no-history
memory_bank_path: "browser_use/memory_bank"
use_vision: false
max_steps: 25
judge:
  enabled: false
initial_actions:                      # Initial browser actions
  - navigate:
      url: "https://www.bing.com"
system_instruction: |                 # System instruction for agent
  First, retrieve user's preferred website from their memory bank...
```

### Personalization Schemes

| Scheme | Description | Memory Access |
|--------|-------------|---------------|
| `on-demand` | Query memory during execution as needed | Real-time |
| `pre-execution` | Analyze memory and augment task before execution | Pre-processed |
| `no-history` | Pure web navigation without memory | Disabled |

## Usage

### Parallel Execution (5 Terminals)

```bash
./scripts/open_5_terminals.sh
```

This script:
1. Opens 5 macOS Terminal windows
2. Assigns 1/5 of total tasks to each terminal
3. Default config: `config/on-demand_o3_1.yml`

### Single Process Execution

```bash
# Run with default config
./scripts/launch_eval.sh

# Specify config file
./scripts/launch_eval.sh config/pre-execution_o3_1.yml
```

### Manual Shard Execution

```bash
# First shard of 5 (index 0)
./scripts/launch_eval_shard.sh --num-shards 5 --shard-index 0

# Second of 3 shards with specific config
./scripts/launch_eval_shard.sh config/no-history_o3_1.yml --num-shards 3 --shard-index 1
```

### Direct Python Execution

```bash
uv run agent_runner.py --config config/on-demand_o3_1.yml

# Specify tasks directory
uv run agent_runner.py --config config/on-demand_o3_1.yml --tasks-dir task/personalization_1

# Specify output directory
uv run agent_runner.py --config config/on-demand_o3_1.yml --out-dir result/my_experiment
```

## Task File Format

```json
{
  "sites": ["bing"],
  "task_id": "Alex_Garcia_A1",
  "require_login": false,
  "start_url": "https://www.bing.com/",
  "intent": "Query: At kohls.com, find men's slim-fit chinos in 32x32..."
}
```

## Output

### Trajectory File Naming Convention

```
{scheme}_{model}_{task_id}.json
```

Examples:
- `on_o3_Alex_Garcia_A1.json` (on-demand scheme)
- `pre_o3_Alex_Garcia_A1.json` (pre-execution scheme)
- `no_o3_Alex_Garcia_A1.json` (no-history scheme)

### Resume Behavior

- Already completed tasks (trajectory file exists) are automatically skipped
- When resuming after interruption, only incomplete tasks are executed

## Troubleshooting

### Terminal windows not opening

```bash
# Grant execute permission to scripts
chmod +x scripts/*.sh
```

### Task directory not found

Ensure `experiment_id` in config file matches the `task/personalization_{id}` directory name.

### Memory bank file not found

Verify that a memory bank file exists matching the user name extracted from the task filename (e.g., `Alex_Garcia_A1` -> `Alex_Garcia`).

## License

This project is part of Browser-Use.

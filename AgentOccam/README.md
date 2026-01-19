# AgentOccam Async Server

A personalized web navigation agent system with memory-enhanced capabilities for automated web browsing and task execution.

## Overview

AgentOccam is an intelligent web agent that can perform personalized web navigation tasks using memory banks. The system supports multiple LLM providers (OpenAI, Google Gemini, Qwen, Meta Llama) and offers two memory access strategies:

- **Pre-execution mode**: Retrieves relevant memories before task execution to refine the objective
- **On-demand mode**: Accesses memory dynamically during task execution when needed

## Key Features

- **Personalized Memory Banks**: 50+ pre-configured user personas with preferences stored in JSON format
- **Multi-LLM Support**: Compatible with GPT-4, O3, Gemini, Qwen, and Llama models
- **Parallel Execution**: Run multiple tasks concurrently across 5 terminal shards
- **Browser Automation**: Built on Playwright with stealth mode for realistic web interaction
- **Flexible Memory Strategies**: Choose between pre-execution and on-demand memory access
- **Task Sharding**: Distribute workload across multiple processes for faster execution

## Repository Structure

```
AgentOccam_async_server/
├── AgentOccam/                    # Core agent implementation
│   ├── AgentOccam_personalized.py # Main agent with memory integration
│   ├── access_scheme.py           # Memory access strategy wrapper
│   ├── retriever.py               # Memory bank retrieval system
│   ├── env.py                     # Web environment wrapper
│   ├── llms/                      # LLM provider implementations
│   ├── prompts/                   # Agent prompts and templates
│   ├── configs/                   # Agent configuration files
│   └── memory_bank/               # User persona memory banks (50+ personas)
├── browser_env/                   # Browser automation components
├── config_files/                  # Task configuration directories
│   ├── personalization_1/         # Task set 1 (150 tasks)
│   ├── personalization_2/         # Task set 2 (150 tasks)
│   └── personalization_3/         # Task set 3 (150 tasks)
├── scripts/                       # Execution scripts
│   ├── open_5_terminals.sh        # Launch 5 parallel shards (macOS)
│   ├── launch_eval_shard.sh       # Single shard launcher
│   └── launch_eval.sh             # Base evaluation launcher
├── agent_runner.py                # Main task execution orchestrator
├── setup_embedding_env.py         # Environment setup for embeddings
└── requirements.txt               # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- macOS (for parallel execution with `open_5_terminals.sh`)
- API keys for your chosen LLM provider

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AgentOccam_async_server
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   playwright install
   ```

3. **Setup embedding environment**
   ```bash
   python setup_embedding_env.py
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

5. **Configure API keys**

   Edit `scripts/launch_eval.sh` and add your API keys:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export GEMINI_API_KEY="your_gemini_api_key_here"
   ```

### Running AgentOccam

#### Parallel Execution (Recommended)

Launch 5 terminal shards to process tasks in parallel:

```bash
./scripts/open_5_terminals.sh
```

This will:
- Open 5 Terminal windows
- Distribute tasks evenly across all shards
- Run tasks concurrently for maximum throughput
- Skip already-completed tasks automatically

#### Single Process Execution

Run a single configuration:

```bash
python agent_runner.py --config AgentOccam/configs/pre-execution_o3_1.yml
```

Or use the launcher script:

```bash
./scripts/launch_eval.sh [config_path]
```

#### Custom Shard Execution

Run a specific shard manually:

```bash
./scripts/launch_eval_shard.sh \
  AgentOccam/configs/pre-execution_o3_1.yml \
  --num-shards 5 \
  --shard-index 0
```

## Configuration

### Agent Configuration

Configuration files are located in `AgentOccam/configs/`. The main configuration includes:

```yaml
# Example: pre-execution_o3_1.yml
logging: True
logdir: "AgentOccam-Trajectories"
logname: "pre-execution_o3_1"
max_steps: 20

memory_bank:
  model_name: "dunzhang/stella_en_1.5B_v5"
  memory_bank_path: "AgentOccam/memory_bank/Alex_Garcia.json"
  llm_model: "o3"

agent:
  type: "AgentOccam"
  memory_mode: pre-execution  # or "on-demand"
  actor:
    model: "o3"  # or "gpt-4-turbo", "gemini-2.0-flash-exp", etc.
    max_steps: 20

env:
  fullpage: true
  headless: False
  relative_task_dir: "personalization_1"
  task_ids: ["all"]  # or specific task IDs
```

### Memory Modes

**Pre-execution Mode**:
- Retrieves relevant memories before task execution
- Enhances the task objective with personalized context
- Best for tasks requiring upfront personalization

**On-demand Mode**:
- Accesses memory dynamically during execution
- Uses `memory_access` command when needed
- Best for tasks where memory needs emerge during execution

### Task Configuration

Tasks are stored in `config_files/` with three sets:
- `personalization_1/`: 150 personalized tasks
- `personalization_2/`: 150 personalized tasks
- `personalization_3/`: 150 personalized tasks

Each task file (e.g., `Alex_Garcia_A1.json`) contains:
```json
{
  "task_id": "Alex_Garcia_A1",
  "intent": "Book a hotel in San Francisco",
  "start_url": "https://www.booking.com",
  "persona": "Alex_Garcia"
}
```

### Memory Banks

50+ user personas are stored in `AgentOccam/memory_bank/` with preferences like:
- Favorite websites and brands
- Dietary restrictions and preferences
- Shopping habits and sizes
- Travel preferences
- Entertainment choices

Example personas: `Alex_Garcia.json`, `Jordan_Smith.json`, `Taylor_Williams.json`, etc.

## Supported LLM Models

### OpenAI
- `gpt-4-turbo`
- `gpt-4o`
- `o3-mini`
- `o3`

### Google Gemini
- `gemini-2.0-flash-exp`
- `gemini-1.5-pro`

### Qwen (via DashScope)
- `qwen3-next-80b-a3b-thinking`
- `qwen-max`

### Meta Llama (via OpenRouter)
- `llama-3.1-405b`
- `llama-3.1-70b`

## Output

Results are saved to `AgentOccam-Trajectories/{logname}/`:

```
AgentOccam-Trajectories/
└── pre-execution_o3_1/
    ├── pre-execution_o3_1_Alex_Garcia_A1.json
    ├── pre-execution_o3_1_Alex_Garcia_B1.json
    └── ...
```

Each trajectory file contains:
- Task objective and configuration
- Step-by-step actions and observations
- Memory access events
- Success/failure status
- Execution metadata

## Advanced Usage

### Custom Configuration

Create a new config file in `AgentOccam/configs/`:

```yaml
# my_custom_config.yml
logging: True
logdir: "my_results"
logname: "custom_run"

memory_bank:
  model_name: "dunzhang/stella_en_1.5B_v5"
  memory_bank_path: "AgentOccam/memory_bank/Custom_Persona.json"
  llm_model: "gpt-4-turbo"

agent:
  type: "AgentOccam"
  memory_mode: on-demand
  actor:
    model: "gpt-4-turbo"

env:
  relative_task_dir: "personalization_1"
  task_ids: ["Alex_Garcia_A1", "Jordan_Smith_B2"]
```

Run with:
```bash
./scripts/launch_eval.sh AgentOccam/configs/my_custom_config.yml
```

### Adding Custom Personas

1. Create a new memory bank file:
   ```bash
   cp AgentOccam/memory_bank/Alex_Garcia.json AgentOccam/memory_bank/My_Persona.json
   ```

2. Edit the JSON with your preferences

3. Update your config to use the new persona:
   ```yaml
   memory_bank:
     memory_bank_path: "AgentOccam/memory_bank/My_Persona.json"
   ```

### Filtering Completed Tasks

The shard launcher automatically skips completed tasks by checking:
- `{logdir}/{logname}/*.json`
- `{logdir}/{logname}/captcha_stopped/*.json`

To force re-running completed tasks:
- Delete the output directory
- Or modify the task IDs in your config

## Troubleshooting

### Common Issues

**Playwright browser not found**:
```bash
playwright install chromium
```

**FAISS installation issues**:
```bash
# CPU version
pip install faiss-cpu

# GPU version (if CUDA available)
pip install faiss-gpu
```

**API Key errors**:
- Verify environment variables are set in `scripts/launch_eval.sh`
- Check API key validity and quota

**Memory retriever not available**:
```bash
python setup_embedding_env.py
export OPENAI_API_KEY="your_key"
```

**Parallel execution not working**:
- Ensure you're on macOS (AppleScript required)
- Make scripts executable: `chmod +x scripts/*.sh`

## Performance Tips

1. **Use parallel execution** with `open_5_terminals.sh` for maximum throughput
2. **Set `headless: True`** in config for faster execution without GUI
3. **Use lighter models** (e.g., `gpt-4o-mini`, `gemini-1.5-flash`) for faster responses
4. **Enable caching** - completed tasks are automatically skipped
5. **Adjust `max_steps`** to limit execution time per task

## Architecture

```
┌─────────────────────────────────────────┐
│         agent_runner.py                 │
│  (Task orchestration & management)      │
└─────────────────┬───────────────────────┘
                  │
                  ├──> Task 1 ──> MemoryEnhancedAgent (access_scheme.py)
                  │                    │
                  ├──> Task 2          ├──> Pre-execution: Retrieve → Enhance → Execute
                  │                    │
                  ├──> Task 3          └──> On-demand: Execute with memory_access
                  │                              │
                  ├──> Task 4                    ▼
                  │                    ┌──────────────────┐
                  └──> Task 5          │ AgentOccam       │
                                       │ (personalized)   │
                                       └────────┬─────────┘
                                                │
                        ┌───────────────────────┼───────────────────────┐
                        ▼                       ▼                       ▼
                  ┌──────────┐          ┌──────────┐           ┌──────────┐
                  │ Retriever│          │ Browser  │           │   LLM    │
                  │ (FAISS)  │          │ (Playwright)        │ Provider │
                  └──────────┘          └──────────┘           └──────────┘
```

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{agentoccam_async,
  title={AgentOccam Async Server: Personalized Web Navigation Agent},
  author={[Your Name/Organization]},
  year={2025},
  url={https://github.com/[your-repo]}
}
```

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check `INSTALL.md` for detailed installation instructions
- Review configuration examples in `AgentOccam/configs/`

---

**Note**: This system is designed for research and development purposes. Ensure compliance with website terms of service when using automated browsing.

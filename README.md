# Persona2Web

A comprehensive framework for personalized web navigation agents. This repository contains two distinct memory access schemes for automated web browsing with personalization module.

## Overview

Persona2Web provides web agents that perform personalized web navigation tasks using user memory banks. The framework supports multiple LLM providers and offers different memory access strategies and ambiguity levels for task execution.

## Projects

### 1. AgentOccam

AgentOccam is a web agent system built on Playwright for realistic web interaction.

#### Personalization Schemes

| Scheme | Description | Memory Access |
|--------|-------------|---------------|
| `on-demand` | Query memory during execution as needed | Real-time |
| `pre-execution` | Analyze memory and augment task before execution | Pre-processed |
| `no-history` | Pure web navigation without memory | Disabled |

#### Architecture

```
agent_runner.py (Task orchestration)
       │
       ├──> MemoryEnhancedAgent (access_scheme.py)
       │           │
       │           ├──> Pre-execution: Retrieve → Enhance → Execute
       │           │
       │           └──> On-demand: Execute with memory_access
       │
       ├──> Retriever (FAISS)
       ├──> Browser (Playwright)
       └──> LLM Provider
```

#### Quick Start

```bash
cd AgentOccam

# Install dependencies
pip install -r requirements.txt
playwright install

# Setup embedding environment
python setup_embedding_env.py

# Configure API keys in scripts/launch_eval.sh
export OPENAI_API_KEY="your_key"

# Run parallel execution (5 terminals)
./scripts/open_5_terminals.sh

# Or single process execution
python agent_runner.py --config AgentOccam/configs/pre-execution_o3_1.yml
```

---

### 2. Browser-Use

Browser-Use provides an experiment runner for personalized web browsing tasks.

#### Personalization Schemes

| Scheme | Description | Memory Access |
|--------|-------------|---------------|
| `on-demand` | Query memory during execution as needed | Real-time |
| `pre-execution` | Analyze memory and augment task before execution | Pre-processed |
| `no-history` | Pure web navigation without memory | Disabled |

#### Configuration Example

```yaml
experiment_id: "1"
backbone:
  provider: "openai"
  model: "o3"
scheme: "on-demand"
memory_bank_path: "browser_use/memory_bank"
use_vision: false
max_steps: 25
initial_actions:
  - navigate:
      url: "https://www.bing.com"
```

#### Quick Start

```bash
cd browser-use

# Install dependencies
uv sync

# Configure environment variables in .env
OPENAI_API_KEY=your-api-key

# Run parallel execution (5 terminals)
./scripts/open_5_terminals.sh

# Or single process execution
./scripts/launch_eval.sh config/on-demand_o3_1.yml

# Direct Python execution
uv run agent_runner.py --config config/on-demand_o3_1.yml
```

---

## Task File Format

Both systems use the same task file formats:

```json
{
  "task_id": "Alex_Garcia_A1",
  "intent": "Book a hotel in San Francisco",
  "start_url": "https://www.bing.com"
}
```

## Supported LLM Models

### OpenAI
- gpt-4.1, o3

### Google Gemini
- gemini-2.5-flash, gemini-2.5-pro

### Qwen (via DashScope)
- qwen3-next-80b-a3b-instruct, qwen3-next-80b-a3b-thinking

### Meta Llama (via OpenRouter)
- llama-3.3-70b

## Output

Both systems save trajectories containing:
- Task objective and configuration
- Step-by-step actions and observations
- Memory access events
- Execution metadata

## License

See individual project directories for license information.

## Note

These systems are designed for research and development purposes. Ensure compliance with website terms of service when using automated browsing.

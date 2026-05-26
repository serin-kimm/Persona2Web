# Persona2Web

**Persona2Web** is a benchmark for evaluating personalized web agents on the real open web.

Persona2Web provides web agents that perform personalized web navigation tasks using user memory banks. The framework supports multiple LLM providers and offers different memory access strategies and ambiguity levels for task execution.

Persona2Web consists of three core components:

1. **User History**  
   Browsing-style user histories that reveal preferences implicitly through behavioral patterns over long time spans.

2. **Ambiguous Queries**  
   Query sets that intentionally conceal specific details, such as preferred websites or preference constraints, requiring agents to clarify the missing information from user history.

3. **Reasoning-aware Evaluation**  
   An evaluation framework that examines full trajectories and reasoning traces to distinguish personalization failures from navigation failures.

---

## Repository Overview

This repository provides evaluation runners for Persona2Web using two web agent architectures:

| Runner | Description |
|--------|-------------|
| `AgentOccam` | Persona2Web runner based on AgentOccam, extended with user history retrieval and personalization support. |
| `browser-use` | Persona2Web runner based on Browser-Use, extended with the same history access schemes and task format. |

Both runners execute Persona2Web tasks on the real open web and save full trajectories for later evaluation.

---

## History Access Schemes

Persona2Web supports three history access schemes that differ in when and how the agent accesses user history.

| Scheme | Description |
|--------|-------------|
| `no-history` | The agent performs the task without access to user history. This setting tests whether ambiguous queries can be resolved without personalization. |
| `on-demand` | The agent accesses user history dynamically during task execution, only when the planner determines it is needed. |
| `pre-execution` | The agent retrieves relevant user history before execution begins and uses it to resolve query ambiguity in advance. |

These schemes are used to evaluate different personalization capabilities, including real-time situational awareness and long-horizon planning.

---

## Repository Structure

```text
Persona2Web/
├── AgentOccam/
│   ├── AgentOccam_personalized.py
│   ├── access_scheme.py
│   ├── retriever.py
│   ├── env.py
│   ├── configs/
│   ├── prompts/
│   └── memory_bank/
│
├── browser-use/
│   ├── agent_runner.py
│   ├── config/
│   ├── task/
│   ├── browser_use/
│   │   └── memory_bank/
│   ├── result/
│   └── scripts/
│
├── config_files/
│   ├── personalization_1/
│   ├── personalization_2/
│   └── personalization_3/
│
└── README.md
```

The two runners share the same benchmark goal and history access schemes, but use different base agent architectures and execution pipelines.

---

## Persona2Web Task Format

Each task contains a web navigation query associated with a user persona.

```json
{
  "task_id": "Alex_Garcia_A1",
  "intent": "Search for doctors in my usual area that match my preferred provider gender at my preferred website.",
  "start_url": "https://www.bing.com",
  "persona": "Alex_Garcia"
}
```

For ambiguous queries, the agent must infer missing details from the corresponding user history.

---

## User History Format

User histories are stored as memory banks. Each history entry is structured with the following fields:

| Field | Description |
|-------|-------------|
| `timestamp` | When the user behavior occurred. |
| `type` | The action type, such as web search, web visit, purchase, booking, or review and rating. |
| `object` | Detailed information about the target entity of the action. |
| `website` | The website associated with the behavior. |

User preferences are not given as explicit statements. Instead, they are embedded implicitly through repeated behavioral patterns.

---

## Running Persona2Web with AgentOccam

### Setup

```bash
cd AgentOccam

pip install -r requirements.txt
playwright install

python setup_embedding_env.py
```

Set the required API keys:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
```

### Run a Single Configuration

```bash
python agent_runner.py --config AgentOccam/configs/pre-execution_o3_1.yml
```

or:

```bash
./scripts/launch_eval.sh AgentOccam/configs/pre-execution_o3_1.yml
```

### Run Parallel Shards

```bash
./scripts/open_5_terminals.sh
```

### Example Configuration

```yaml
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
  memory_mode: pre-execution
  actor:
    model: "o3"
    max_steps: 20

env:
  fullpage: true
  headless: false
  relative_task_dir: "personalization_1"
  task_ids: ["all"]
```

### Output

AgentOccam trajectories are saved under:

```text
AgentOccam-Trajectories/{logname}/
```

Each trajectory contains the task objective, observations, actions, history access events, reasoning traces, and execution metadata.

---

## Running Persona2Web with Browser-Use

### Setup

```bash
cd browser-use

uv sync
```

Configure environment variables in `.env` or export them manually:

```bash
OPENAI_API_KEY=your_openai_api_key
```

Add other provider keys as needed.

### Run a Single Configuration

```bash
./scripts/launch_eval.sh config/on-demand_o3_1.yml
```

or:

```bash
uv run agent_runner.py --config config/on-demand_o3_1.yml
```

### Run Parallel Shards

```bash
./scripts/open_5_terminals.sh
```

### Run a Manual Shard

```bash
./scripts/launch_eval_shard.sh \
  config/on-demand_o3_1.yml \
  --num-shards 5 \
  --shard-index 0
```

### Example Configuration

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

The `experiment_id` should match the corresponding task directory, such as:

```text
task/personalization_1
```

### Output

Browser-Use trajectories are saved under:

```text
result/{config_name}/
```

Trajectory files follow the scheme/model/task naming convention, such as:

```text
on_o3_Alex_Garcia_A1.json
pre_o3_Alex_Garcia_A1.json
no_o3_Alex_Garcia_A1.json
```

Completed tasks are automatically skipped when the corresponding trajectory file already exists.

---

## Evaluation

Persona2Web evaluates personalized web agents using four metrics:

| Metric | Description |
|--------|-------------|
| `Pweb` | Measures whether the agent recognizes websites that align with recurring usage patterns in user history. |
| `Ppref` | Measures whether the agent identifies items or attributes that best reflect user preferences. |
| `Intent` | Measures task accuracy apart from personalization. |
| `Success Rate` | Counts cases where the agent both personalizes and executes the task accurately. |

The evaluation is reasoning-aware: it examines the full trajectory, including retrieved histories, reasoning traces, and actions, rather than relying only on final task completion.

---

## Notes

Persona2Web is designed for research on personalized web agents in real open-web environments. The benchmark uses synthetically generated user profiles and browsing histories; no real user data is collected or used.

When running open-web experiments, ensure compliance with website terms of service and avoid excessive automated requests.

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import re

from dotenv import load_dotenv

from browser_use import Agent, ChatBrowserUse

# Optional providers (lazy imports inside factory)
# from browser_use.llm.openai.chat import ChatOpenAI
# from browser_use.llm.google.chat import ChatGoogle
# from browser_use.llm.anthropic.chat import ChatAnthropic


# -------------------------------
# Config models (minimal, typed)
# -------------------------------
BackboneProvider = Literal["browser-use", "openai", "google", "anthropic"]
Scheme = Literal["on-demand", "pre-execution", "no-history"]


@dataclass
class BackboneConfig:
	provider: BackboneProvider
	model: str
	base_url: str | None = None


@dataclass
class JudgeConfig:
	enabled: bool = False


@dataclass
class ExperimentConfig:
	experiment_id: str
	backbone: BackboneConfig
	scheme: Scheme = "on-demand"
	memory_bank_path: str | None = None
	use_vision: bool | None = None
	max_steps: int = 100
	judge: JudgeConfig | None = None
	initial_actions: list[dict[str, dict[str, Any]]] | None = None
	system_instruction: str | None = None

	@staticmethod
	def from_json(data: dict[str, Any]) -> "ExperimentConfig":
		backbone_raw = data.get("backbone") or {}
		backbone = BackboneConfig(
			provider=backbone_raw.get("provider", "browser-use"),
			model=backbone_raw.get("model", ""),
			base_url=backbone_raw.get("base_url"),
		)
		judge_raw = data.get("judge") or {}
		judge = JudgeConfig(enabled=bool(judge_raw.get("enabled", False))) if judge_raw is not None else None
		return ExperimentConfig(
			experiment_id=str(data.get("experiment_id", "default")),
			backbone=backbone,
			scheme=data.get("scheme", "on-demand"),
			memory_bank_path=data.get("memory_bank_path"),
			use_vision=data.get("use_vision"),
			max_steps=int(data.get("max_steps", 100)),
			judge=judge,
			initial_actions=data.get("initial_actions"),
			system_instruction=data.get("system_instruction"),
		)


# -------------------------------
# Utilities
# -------------------------------
def read_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def read_yaml(path: Path) -> dict[str, Any]:
	try:
		import yaml  # type: ignore
	except Exception as e:
		raise RuntimeError(
			"YAML config requested but 'pyyaml' is not installed. Install with: uv add pyyaml"
		) from e
	with path.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	if not isinstance(data, dict):
		raise ValueError(f"YAML file must contain a mapping at root: {path}")
	return data


def read_config_any(path: Path) -> dict[str, Any]:
	ext = path.suffix.lower()
	if ext in (".yml", ".yaml"):
		return read_yaml(path)
	if ext == ".json":
		return read_json(path)
	# Fallback by trying JSON first, then YAML
	try:
		return read_json(path)
	except Exception:
		return read_yaml(path)


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def discover_tasks(tasks_dir: Path) -> list[Path]:
	# Consider .json, .yaml, .yml files as tasks
	candidates: list[Path] = []
	for pattern in ("*.json", "*.yaml", "*.yml"):
		candidates.extend([p for p in tasks_dir.glob(pattern) if p.is_file()])
	return sorted(candidates)


def read_task_any(path: Path) -> dict[str, Any]:
	ext = path.suffix.lower()
	if ext in (".yml", ".yaml"):
		try:
			return read_yaml(path)
		except Exception as e:
			raise RuntimeError(f"Failed to read YAML task file {path}: {e}") from e
	if ext == ".json":
		return read_json(path)
	# fallback
	return read_json(path)


def extract_task_text(task_json: Any) -> str:
	# Accept dict or other types; normalize to text
	if isinstance(task_json, dict):
		# Heuristic: prefer common keys
		for key in ("intent",):
			if key in task_json and isinstance(task_json[key], str) and task_json[key].strip():
				return task_json[key].strip()
		# Fallback: stringify entire dict
		return json.dumps(task_json, ensure_ascii=False, indent=2)
	if isinstance(task_json, str):
		return task_json.strip()
	if isinstance(task_json, list):
		try:
			return "\n".join([str(x) for x in task_json])
		except Exception:
			return json.dumps(task_json, ensure_ascii=False)
	# Heuristic: prefer common keys
	return str(task_json)


def llm_factory(provider: BackboneProvider, model: str, base_url: str | None = None):
	if provider == "browser-use":
		# Best default per project rules
		return ChatBrowserUse() if not model else ChatBrowserUse(model=model)
	if provider == "openai":
		from browser_use.llm.openai.chat import ChatOpenAI

		if base_url:
			return ChatOpenAI(model=model or "gpt-4.1", base_url=base_url)
		return ChatOpenAI(model=model or "gpt-4.1")
	if provider == "google":
		from browser_use.llm.google.chat import ChatGoogle

		return ChatGoogle(model=model or "gemini-flash-latest")
	if provider == "anthropic":
		from browser_use.llm.anthropic.chat import ChatAnthropic

		return ChatAnthropic(model=model or "claude-3-5-sonnet-latest")
	raise ValueError(f"Unsupported provider: {provider}")


def build_agent(task: str, cfg: ExperimentConfig, memory_bank_file: str | None = None, skip_system_instruction: bool = False) -> Agent:
	llm = llm_factory(cfg.backbone.provider, cfg.backbone.model, cfg.backbone.base_url)

	# Prepend system instruction if provided (skip for pre-execution scheme)
	if cfg.system_instruction and not skip_system_instruction:
		task = f"{cfg.system_instruction} {task}"

	agent_kwargs: dict[str, Any] = {
		"task": task,
		"llm": llm,
		"initial_actions": cfg.initial_actions or [],
	}
	print(f"[debug] initial_actions passed to Agent: {cfg.initial_actions}")

	# Personalization scheme
	if cfg.scheme == "on-demand":
		# Enable only if memory bank is provided
		if memory_bank_file:
			agent_kwargs["memory_bank_path"] = memory_bank_file
		elif cfg.memory_bank_path:
			mem_path = Path(cfg.memory_bank_path).expanduser().resolve()
			if mem_path.is_dir():
				# Directory provided but no per-task file resolved by caller
				# Proceed without memory (warn), since per-task mapping is preferred
				print(f"[warn] No per-task memory file resolved for directory: {mem_path}")
			else:
				agent_kwargs["memory_bank_path"] = str(mem_path)
	elif cfg.scheme == "pre-execution":
		# Pre-execution: task already enhanced, no memory_bank_path needed
		# Memory access is disabled during execution
		pass
	elif cfg.scheme == "no-history":
		# No-history: memory completely disabled, plain web navigation
		# No memory_bank_path, no memory checks, no caching
		pass
	else:
		raise ValueError(f"Unknown scheme: {cfg.scheme}")

	# Optional flags
	if cfg.use_vision is not None:
		agent_kwargs["use_vision"] = cfg.use_vision

	agent = Agent(**agent_kwargs)

	# Optional settings
	if cfg.max_steps:
		agent.settings.max_actions_per_step = min(agent.settings.max_actions_per_step, 3)  # keep default cap sane

	# Judge - explicitly set based on config (default is True in AgentSettings)
	if cfg.judge is not None:
		agent.settings.use_judge = cfg.judge.enabled

	return agent


def _derive_user_name_from_task(task_path: Path) -> str:
	stem = task_path.stem
	# Prefer pattern: <user>_(A|B|C)\d+
	m = re.match(r'^(?P<user>.+)_(?:A|B|C)\d+$', stem)
	if m:
		return m.group('user')
	# Fallback: first token before underscore
	return stem.split('_')[0] if '_' in stem else stem


async def run_task_file(task_path: Path, cfg: ExperimentConfig, out_dir: Path) -> dict[str, Any]:
	task_obj = read_task_any(task_path)
	task_text = extract_task_text(task_obj)

	# Resolve per-task memory bank file by user name derived from task filename
	memory_bank_file: str | None = None
	pre_execution_info: dict[str, Any] | None = None

	# No-history scheme: skip all memory processing
	if cfg.scheme == "no-history":
		pass  # No memory_bank_file, no pre-execution, just plain navigation
	elif cfg.memory_bank_path:
		mem_base = Path(cfg.memory_bank_path).expanduser().resolve()
		if mem_base.is_dir():
			user_name = _derive_user_name_from_task(task_path)
			candidate = mem_base / f"{user_name}.json"
			if candidate.exists():
				memory_bank_file = str(candidate)
			else:
				print(f"[warn] Memory file not found for user '{user_name}' in {mem_base}")
		elif mem_base.is_file():
			# Back-compat: direct file path provided
			memory_bank_file = str(mem_base)

	# Handle pre-execution scheme: process memory BEFORE agent execution
	if cfg.scheme == "pre-execution" and memory_bank_file:
		from browser_use.personalization.pre_execution import PreExecutionService

		llm = llm_factory(cfg.backbone.provider, cfg.backbone.model, cfg.backbone.base_url)
		pre_exec_service = PreExecutionService(
			memory_bank_path=memory_bank_file,
			llm=llm,
		)

		# For memory query generation, use system_instruction + task (full context)
		# But enhanced_task returned will be WITHOUT system_instruction for Agent
		full_task_for_analysis = task_text
		if cfg.system_instruction:
			full_task_for_analysis = f"{cfg.system_instruction} {task_text}"

		print(f"[pre-execution] Analyzing task and retrieving memory...")
		original_task = task_text
		# Pass full_task for analysis, original task_text for rewriting (no system_instruction in output)
		task_text, memory_events = await pre_exec_service.preprocess_task(full_task_for_analysis, original_task=task_text)

		if memory_events:
			print(f"[pre-execution] Retrieved {len(memory_events)} memory queries")
			pre_execution_info = {
				"original_task": original_task,
				"enhanced_task": task_text,
				"memory_events": memory_events,
			}
		else:
			print(f"[pre-execution] No memory needed for this task")

		# Free embedding model memory before starting browser
		pre_exec_service.cleanup()
		print(f"[pre-execution] Cleanup complete, memory freed")

	# For pre-execution, don't pass memory_bank_file to agent (memory access disabled during execution)
	# Also skip system_instruction for pre-execution since enhanced_task already contains the context
	agent_memory_bank = memory_bank_file if cfg.scheme == "on-demand" else None
	skip_instruction = cfg.scheme == "pre-execution"
	agent = build_agent(task_text, cfg, memory_bank_file=agent_memory_bank, skip_system_instruction=skip_instruction)

	# For pre-execution: max 20 navigation steps (memory queries already consumed up to 5 steps)
	# For on-demand: use config max_steps as-is
	if cfg.scheme == "pre-execution":
		memory_step_count = len(pre_execution_info.get("memory_events", [])) if pre_execution_info else 0
		nav_max_steps = cfg.max_steps - memory_step_count  # e.g., 25 - 5 = 20
	else:
		nav_max_steps = cfg.max_steps

	history = await agent.run(max_steps=nav_max_steps)

	# Save trajectory log with format: scheme_model_taskfilename.json
	# scheme: on-demand -> "on", pre-execution -> "pre", no-history -> "no"
	scheme_short = {"on-demand": "on", "pre-execution": "pre", "no-history": "no"}[cfg.scheme]
	task_name = task_path.stem  # e.g., "Alex_Garcia_A1" from task/personalization_1/Alex_Garcia_A1.json
	# Simplify model name: remove slashes, dots, dashes
	model_name = cfg.backbone.model.replace("/", "").replace("-", "").replace(".", "")
	trajectory_filename = f"{scheme_short}_{model_name}_{task_name}.json"
	trajectory_file = out_dir / trajectory_filename
	agent.save_trajectory(str(trajectory_file))

	# For pre-execution scheme, prepend memory steps to trajectory
	# Each memory query counts as one step (max 5 memory steps)
	# Navigation steps follow after memory steps
	if pre_execution_info and trajectory_file.exists():
		with trajectory_file.open("r", encoding="utf-8") as f:
			navigation_steps = json.load(f)

		memory_events = pre_execution_info.get("memory_events", [])
		memory_steps = []

		# Convert each memory event to a step entry (step 1, 2, 3, ...)
		for i, event in enumerate(memory_events):
			memory_step = {
				"step": i + 1,
				"url": None,  # No URL for pre-execution memory steps
				"step_type": "memory",
				"memory_query": event.get("query", ""),
				"retrieved_memory_indices": event.get("retrieved_indices", []),
				"memory_answer": event.get("answer", ""),
			}
			memory_steps.append(memory_step)

		# Adjust navigation step numbers to continue after memory steps
		memory_step_count = len(memory_steps)
		for nav_step in navigation_steps:
			if "step" in nav_step:
				nav_step["step"] += memory_step_count

		# Combine: memory steps first, then navigation steps
		combined_trajectory = memory_steps + navigation_steps

		with trajectory_file.open("w", encoding="utf-8") as f:
			json.dump(combined_trajectory, f, ensure_ascii=False, indent=2)

	# Collect summary
	result_text = history.final_result()
	success = history.is_successful()
	num_steps = history.number_of_steps()
	usage_obj = getattr(history, "usage", None)
	# Pydantic v2 objects expose model_dump()
	if hasattr(usage_obj, "model_dump"):
		try:
			usage = usage_obj.model_dump()  # type: ignore[attr-defined]
		except Exception:
			usage = str(usage_obj)
	else:
		usage = usage_obj

	record = {
		"task_file": str(task_path),
		"experiment_id": cfg.experiment_id,
		"provider": cfg.backbone.provider,
		"model": cfg.backbone.model,
		"scheme": cfg.scheme,
		"memory_bank_path": cfg.memory_bank_path,
		"success": success,
		"num_steps": num_steps,
		"final_result": result_text,
		"usage": usage,
	}

	# Persist result.json is disabled - trajectory file is sufficient
	# out_file = out_dir / f"{task_path.stem}.result.json"
	# with out_file.open("w", encoding="utf-8") as f:
	# 	json.dump(record, f, ensure_ascii=False, indent=2)

	return record


async def main() -> None:
	load_dotenv()

	parser = argparse.ArgumentParser(description="Run Browser-Use experiments over a folder of task files.")
	parser.add_argument("--config", required=True, help="Path to experiment config (json|yaml).")
	parser.add_argument(
		"--tasks-dir",
		required=False,
		help="Path to a folder containing task files (json|yml|yaml). If omitted, derived as task/personalization_<experiment_id>.",
	)
	parser.add_argument(
		"--out-dir",
		default=None,
		help="Directory to write results. Defaults to result/<config_name>/",
	)
	args = parser.parse_args()

	cfg_path = Path(args.config).expanduser().resolve()
	tasks_dir: Path | None = Path(args.tasks_dir).expanduser().resolve() if args.tasks_dir else None
	if not cfg_path.exists():
		raise FileNotFoundError(f"Config file not found: {cfg_path}")

	cfg_data = read_config_any(cfg_path)
	cfg = ExperimentConfig.from_json(cfg_data)

	# Derive tasks_dir from experiment_id when not provided
	if tasks_dir is None:
		repo_root = Path(__file__).resolve().parent
		default_tasks_dir = repo_root / "task" / f"personalization_{cfg.experiment_id}"
		tasks_dir = default_tasks_dir
	if not tasks_dir.exists():
		raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

	# Default out_dir uses config filename (e.g., "pre-execution_gpt41_1" from "pre-execution_gpt41_1.yml")
	config_name = cfg_path.stem
	out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path("result") / config_name
	ensure_dir(out_dir)

	task_files = discover_tasks(tasks_dir)
	if not task_files:
		raise RuntimeError(f"No task JSON files found in: {tasks_dir}")

	# Check for already completed tasks (skip if trajectory file exists)
	scheme_short = {"on-demand": "on", "pre-execution": "pre", "no-history": "no"}[cfg.scheme]
	model_name = cfg.backbone.model.replace("/", "").replace("-", "").replace(".", "")
	pending_tasks: list[Path] = []
	skipped_count = 0
	for task_file in task_files:
		trajectory_filename = f"{scheme_short}_{model_name}_{task_file.stem}.json"
		if (out_dir / trajectory_filename).exists():
			skipped_count += 1
		else:
			pending_tasks.append(task_file)

	print(f"Running {len(pending_tasks)} tasks (skipping {skipped_count} completed) | experiment={cfg.experiment_id} provider={cfg.backbone.provider} model={cfg.backbone.model} scheme={cfg.scheme}")

	results: list[dict[str, Any]] = []
	for task_file in pending_tasks:
		try:
			record = await run_task_file(task_file, cfg, out_dir)
			results.append(record)
			status = "✅" if record["success"] else "⚠️"
			print(f"{status} {task_file.name} -> steps={record['num_steps']} success={record['success']}")
		except Exception as e:
			err_rec = {
				"task_file": str(task_file),
				"experiment_id": cfg.experiment_id,
				"error": f"{type(e).__name__}: {e}",
			}
			results.append(err_rec)
			err_path = out_dir / f"{task_file.stem}.error.json"
			with err_path.open("w", encoding="utf-8") as f:
				json.dump(err_rec, f, ensure_ascii=False, indent=2)
			print(f"❌ {task_file.name} -> {e}")

	# Write summary
	summary_path = out_dir / "_summary.json"
	with summary_path.open("w", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=2)

	print(f"Done. Results at: {out_dir}")


def preload_embedding_model() -> None:
	"""Pre-load embedding model before asyncio loop to avoid conflicts."""
	import os
	os.environ['TOKENIZERS_PARALLELISM'] = 'false'

	print("Pre-loading embedding model...")
	from browser_use.personalization.retriever import MemoryRetriever
	_preload_retriever = MemoryRetriever.__new__(MemoryRetriever)
	_preload_retriever.model = None
	_preload_retriever.embedding_model_name = "dunzhang/stella_en_1.5B_v5"
	_preload_retriever.embedding_dim = None
	_preload_retriever.logger = __import__('logging').getLogger('preload')
	_preload_retriever._load_embedding_model()
	print("Embedding model pre-loaded successfully")


if __name__ == "__main__":
	# Check if we need to preload (read config first)
	import sys
	for i, arg in enumerate(sys.argv):
		if arg == "--config" and i + 1 < len(sys.argv):
			cfg_path = Path(sys.argv[i + 1]).expanduser().resolve()
			if cfg_path.exists():
				cfg_data = read_config_any(cfg_path)
				cfg = ExperimentConfig.from_json(cfg_data)
				# Preload embedding model for both on-demand and pre-execution schemes
				if cfg.memory_bank_path and cfg.scheme in ("on-demand", "pre-execution"):
					preload_embedding_model()
			break

	asyncio.run(main())



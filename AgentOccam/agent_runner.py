import os
import time
import re
import argparse
import os
import shutil
import asyncio
import threading
import random

from AgentOccam.env import WebArenaEnvironmentWrapper

from AgentOccam.AgentOccam_personalized import AgentOccam
from AgentOccam.access_scheme import MemoryEnhancedAgent

from webagents_step.utils.data_prep import *

from AgentOccam.prompts import AgentOccam_prompt

from AgentOccam.utils import EVALUATOR_DIR

def run():
    parser = argparse.ArgumentParser(
        description="Only the config file argument should be passed"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml config file location"
    )
    parser.add_argument(
        "--async-run", action="store_true", help="Run multiple tasks concurrently (async wrapper)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Max concurrent tasks when using --async-run"
    )
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = DotDict(yaml.safe_load(file))
    overall_start_ts = time.time()

    # Handle memory bank configuration
    memory_bank_config = None
    if hasattr(config, 'memory_bank'):
        # Check if memory_bank_path is False (disable memory functionality)
        memory_bank_path = getattr(config.memory_bank, 'memory_bank_path', None)
        if memory_bank_path is False or str(memory_bank_path).lower() == 'false':
            print("🚫 Memory functionality disabled (memory_bank_path: False)")
            memory_bank_config = None
        else:
            memory_bank_config = config.memory_bank.to_dict() if hasattr(config.memory_bank, 'to_dict') else dict(config.memory_bank)

    if config.logging:
        if config.logname:
            dstdir = f"{config.logdir}/{config.logname}"
        else:
            dstdir = f"{config.logdir}/{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(dstdir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(dstdir, args.config.split("/")[-1]))
    # random.seed(42)  # removed to enable true random ordering of tasks
    
    config_file_list = []
    
    task_ids = config.env.task_ids
    if hasattr(config.env, "relative_task_dir"):
        relative_task_dir = config.env.relative_task_dir
    else:
        relative_task_dir = "tasks"
    # Resolve config directory relative to this script's directory to avoid CWD issues
    project_root = os.path.dirname(__file__)
    config_dir = os.path.join(project_root, "config_files", relative_task_dir)
    if task_ids == "all" or task_ids == ["all"]:
        task_ids = [filename[:-len(".json")] for filename in os.listdir(config_dir) if filename.endswith(".json")]
        random.shuffle(task_ids)
    for task_id in task_ids:
        config_file_list.append(os.path.join(config_dir, f"{task_id}.json"))

    fullpage = config.env.fullpage if hasattr(config.env, "fullpage") else True
    current_viewport_only = not fullpage

    if config.agent.type == "AgentOccam":
        memory_mode = getattr(config.agent, 'memory_mode', None)
        # If memory is disabled (memory_bank_config is None), use original AgentOccam regardless of memory_mode
        if memory_bank_config is None:
            print("🎯 Using original AgentOccam (no memory functionality)")
            agent_init = lambda: AgentOccam(
                prompt_dict = {k: v for k, v in AgentOccam_prompt.__dict__.items() if isinstance(v, dict)},
                config = config,
                memory_bank_config = None
            )
        elif memory_mode in ['on-demand', 'pre-execution']:
            agent_init = lambda: MemoryEnhancedAgent(
                config=config,
                prompt_dict={k: v for k, v in AgentOccam_prompt.__dict__.items() if isinstance(v, dict)},
                memory_bank_config=memory_bank_config
            )
        else:
            agent_init = lambda: AgentOccam(
                prompt_dict = {k: v for k, v in AgentOccam_prompt.__dict__.items() if isinstance(v, dict)},
                config = config,
                memory_bank_config = memory_bank_config
            )
    else:
        raise NotImplementedError(f"{config.agent.type} not implemented")

    def run_single_task(config, dstdir, config_file, current_viewport_only, agent_init, summary_lock):
        with open(config_file, "r") as f:
            task_config = json.load(f)
            print(f"Task {task_config['task_id']}.")
        
        # Check if trajectory file already exists
        # New naming: <mode_prefix>_<model_tag_norm>_<task_id>.json
        # Legacy naming: pre_gpt41_<task_id>.json
        task_id = task_config['task_id']
        mmode = getattr(config.agent, 'memory_mode', None)
        mode_prefix = 'pre' if mmode == 'pre-execution' else ('on' if mmode == 'on-demand' else 'base')

        possible_filenames = set()
        # Legacy fallback
        possible_filenames.add(f"pre_gpt41_{task_id}.json")
        # Actor model-based name
        model_tag = getattr(getattr(config.agent, 'actor', object()), 'model', None)
        if model_tag:
            model_tag_norm = re.sub(r'[^A-Za-z0-9]+', '', str(model_tag))
            possible_filenames.add(f"{mode_prefix}_{model_tag_norm}_{task_id}.json")
        # Memory bank model fallback
        try:
            mb_llm = None
            if isinstance(memory_bank_config, dict):
                mb_llm = memory_bank_config.get('llm_model')
        except Exception:
            mb_llm = None
        if mb_llm:
            mb_tag_norm = re.sub(r'[^A-Za-z0-9]+', '', str(mb_llm))
            possible_filenames.add(f"{mode_prefix}_{mb_tag_norm}_{task_id}.json")

        # Direct path checks
        for fname in possible_filenames:
            traj_path = os.path.join(dstdir, fname)
            if os.path.exists(traj_path):
                print(f"Skip {task_id} - trajectory already exists at {traj_path}")
                return
            captcha_path = os.path.join(dstdir, 'captcha_stopped', fname)
            if os.path.exists(captcha_path):
                print(f"Skip {task_id} - trajectory already exists in captcha_stopped folder")
                return

        # Suffix-based fallback (any model)
        suffix = f"_{task_id}.json"
        try:
            if any(name.endswith(suffix) for name in os.listdir(dstdir) if name.endswith('.json')):
                print(f"Skip {task_id} - trajectory already exists (suffix match) in {dstdir}")
                return
        except Exception:
            pass
        captcha_dir = os.path.join(dstdir, 'captcha_stopped')
        if os.path.isdir(captcha_dir):
            try:
                if any(name.endswith(suffix) for name in os.listdir(captcha_dir) if name.endswith('.json')):
                    print(f"Skip {task_id} - trajectory already exists (suffix match) in captcha_stopped folder")
                    return
            except Exception:
                pass
        # if task_config['task_id'] in list(range(600, 650))+list(range(681, 689)):
        #     print("Reddit post task. Sleep 30 mins.")
        #     time.sleep(1800)
        # Mark task start time (exclude any pre-run sleep)
        start_ts = time.time()
        print(f"[RUN] Start {task_config['task_id']} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_ts))}")
        env = WebArenaEnvironmentWrapper(config_file=config_file, 
                                        max_browser_rows=config.env.max_browser_rows, 
                                        max_steps=config.max_steps, 
                                        slow_mo=1, 
                                        observation_type="accessibility_tree", 
                                        current_viewport_only=current_viewport_only, 
                                        viewport_size={"width": 1920, "height": 1080}, 
                                        headless=config.env.headless,
                                        global_config=config)

        # Derive per-task persona and memory bank path
        persona_name = str(task_config.get('task_id', '')).rsplit('_', 1)[0]
        # Build path relative to project root
        project_root = os.path.dirname(__file__)
        candidate_memory_path = os.path.join(
            project_root,
            'AgentOccam',
            'memory_bank',
            f'{persona_name}.json',
        )

        # Prepare per-task memory bank config
        memory_bank_config_task = None
        if memory_bank_config:
            memory_bank_config_task = dict(memory_bank_config)
            if os.path.exists(candidate_memory_path):
                memory_bank_config_task['memory_bank_path'] = candidate_memory_path
        
        # Instantiate agent per task with per-persona memory bank
        memory_mode = getattr(config.agent, 'memory_mode', None)
        # If memory is disabled (memory_bank_config_task is None), use original AgentOccam regardless of memory_mode
        if memory_bank_config_task is None:
            print(f"🎯 Task {task_config['task_id']}: Using original AgentOccam (no memory functionality)")
            from AgentOccam.prompts import AgentOccam_prompt as _Prompts  # local import to avoid circulars
            agent = AgentOccam(
                prompt_dict={k: v for k, v in _Prompts.__dict__.items() if isinstance(v, dict)},
                config=config,
                memory_bank_config=None
            )
        elif memory_mode in ['on-demand', 'pre-execution']:
            from AgentOccam.prompts import AgentOccam_prompt as _Prompts  # local import to avoid circulars
            agent = MemoryEnhancedAgent(
                config=config,
                prompt_dict={k: v for k, v in _Prompts.__dict__.items() if isinstance(v, dict)},
                memory_bank_config=memory_bank_config_task
            )
        else:
            from AgentOccam.prompts import AgentOccam_prompt as _Prompts  # local import to avoid circulars
            agent = AgentOccam(
                prompt_dict={k: v for k, v in _Prompts.__dict__.items() if isinstance(v, dict)},
                config=config,
                memory_bank_config=memory_bank_config_task
            )
        objective = env.get_objective()
        status = agent.act(objective=objective, env=env)
        memory_details_local = agent.get_memory_details() if hasattr(agent, 'get_memory_details') else []
        memory_retrieved_answers = []
        if memory_details_local:
            for event in memory_details_local:
                if event.get('answer'):
                    memory_retrieved_answers.append(event.get('answer', ''))
        trajectory = agent.get_trajectory()
        original_answer = ""
        if trajectory:
            last_step = trajectory[-1]
            if 'action' in last_step and last_step['action'].startswith('stop ['):
                match = re.search(r'stop \[(.*)\]', last_step['action'])
                if match:
                    original_answer = match.group(1)
        combined_answer = original_answer
        if memory_retrieved_answers:
            memory_context = "\n\nMemory Retrieved:\n" + "\n".join(memory_retrieved_answers)
            combined_answer = original_answer + memory_context
        if combined_answer:
            print(f"Using combined answer for evaluation: {combined_answer[:200]}...")
        if memory_retrieved_answers:
            env.memory_data = memory_retrieved_answers
        env.close()
        if config.logging:
            with open(config_file, "r") as f:
                task_config = json.load(f)
            # Build descriptive log file name: <mode>_<model>_<task_id>.json
            # mode: 'pre' for pre-execution, 'on' for on-demand, else 'base'
            mmode = getattr(config.agent, 'memory_mode', None)
            mode_prefix = 'pre' if mmode == 'pre-execution' else ('on' if mmode == 'on-demand' else 'base')
            # model tag: prefer actor model if available; else memory bank llm_model; fallback 'unknown'
            model_tag = getattr(getattr(config.agent, 'actor', object()), 'model', None)
            if not model_tag:
                try:
                    if memory_bank_config_task and memory_bank_config_task.get('llm_model'):
                        model_tag = memory_bank_config_task.get('llm_model')
                except Exception:
                    pass
            if not model_tag:
                model_tag = getattr(config.agent, 'model_name', 'unknown')
            # normalize model tag for filename: remove non-alphanumerics (e.g., 'gpt-4.1' -> 'gpt41')
            model_tag_norm = re.sub(r'[^A-Za-z0-9]+', '', str(model_tag))
            
            # Check if task was stopped due to CAPTCHA/error detection
            trajectory_temp = agent.get_trajectory()
            is_captcha_stopped = False
            if trajectory_temp:
                last_action = trajectory_temp[-1]
                if isinstance(last_action, dict) and 'action' in last_action:
                    action_text = str(last_action['action']).lower()
                    if ('stop' in action_text and 
                        ('captcha' in action_text or 'automatically stopped' in action_text or 
                         'error condition detected' in action_text)):
                        is_captcha_stopped = True
                        print(f"🚨 CAPTCHA/Error auto-stop detected for task {task_config['task_id']}")
            
            # Choose destination directory based on stop reason
            if is_captcha_stopped:
                captcha_dir = os.path.join(dstdir, "captcha_stopped")
                os.makedirs(captcha_dir, exist_ok=True)
                log_file = os.path.join(captcha_dir, f"{mode_prefix}_{model_tag_norm}_{task_config['task_id']}.json")
                print(f"📁 Saving to captcha_stopped folder: {log_file}")
            else:
                log_file = os.path.join(dstdir, f"{mode_prefix}_{model_tag_norm}_{task_config['task_id']}.json")
            structured_memory_data = []
            # NOTE (preserved original):
            # The original behavior stored the full answer text as-is:
            # try:
            #     if memory_details_local:
            #         for event in memory_details_local:
            #             structured_memory_data.append({
            #                 "Memory Query": event.get('query', ''),
            #                 "Retrieved memory index": event.get('retrieved_indices', []),
            #                 "Memory Retrieved": event.get('answer', '')
            #             })
            # except Exception as e:
            #     print(f"Failed to structure memory details: {e}")
            #     structured_memory_data = []
            
            # helper: keep only last sentence of an answer
            def _extract_last_sentence(text):
                try:
                    normalized = re.sub(r'\s+', ' ', str(text or '')).strip()
                    if not normalized:
                        return ""
                    parts = re.split(r'(?<=[.!?。！？])\s+', normalized)
                    parts = [p.strip() for p in parts if p and len(p.strip()) > 0]
                    return parts[-1] if parts else normalized
                except Exception:
                    return str(text or '').strip()
            try:
                # Branch save format by model: only save last sentence for qwen3-next-80b-a3b-thinking
                try:
                    actor_model = config.agent.actor.model
                except Exception:
                    actor_model = ''
                is_qwen_thinking = str(actor_model).strip().lower() == 'qwen3-next-80b-a3b-thinking'
                if memory_details_local:
                    for event in memory_details_local:
                        if is_qwen_thinking:
                            answer_val = _extract_last_sentence(event.get('answer', ''))
                        else:
                            answer_val = event.get('answer', '')
                        structured_memory_data.append({
                            "Memory Query": event.get('query', ''),
                            "Retrieved memory index": event.get('retrieved_indices', []),
                            "Memory Retrieved": answer_val
                        })
            except Exception as e:
                print(f"Failed to structure memory details: {e}")
                structured_memory_data = []
            trajectory = agent.get_trajectory()
            log_data = {
                "task": config_file,
                "id": task_config['task_id'],
                "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                "type": config.agent.type,
                "trajectory": trajectory,
                "memory": structured_memory_data
            }
            summary_file = os.path.join(dstdir, "summary.csv")
            # Calculate execution time before logging
            end_ts = time.time()
            elapsed = end_ts - start_ts
            
            summary_data = {
                "task": config_file,
                "task_id": task_config['task_id'],
                "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                "type": config.agent.type,
                "logfile": re.search(r"/([^/]+/[^/]+\.json)$", log_file).group(1),
                "execution_time": round(elapsed, 2),  # Execution time in seconds (rounded to 2 decimals)
            }
            if status:
                summary_data.update(status)
            with summary_lock:
                log_run(
                    log_file=log_file,
                    log_data=log_data,
                    summary_file=summary_file,
                    summary_data=summary_data,
                )

        # Print end time and duration to console  
        print(f"[RUN] End   {task_config['task_id']} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_ts))} (elapsed {elapsed:.1f}s)")

    if args.async_run and len(config_file_list) > 1:
        print(f"Running {len(config_file_list)} tasks asynchronously with concurrency={args.concurrency}")
        summary_lock = threading.Lock()
        sem = asyncio.Semaphore(args.concurrency)

        async def run_one(cf):
            async with sem:
                await asyncio.to_thread(
                    run_single_task,
                    config,
                    dstdir,
                    cf,
                    current_viewport_only,
                    agent_init,
                    summary_lock,
                )

        async def _main():
            await asyncio.gather(*(run_one(cf) for cf in config_file_list))

        asyncio.run(_main())
    else:
        for config_file in config_file_list:
            run_single_task(config, dstdir, config_file, current_viewport_only, agent_init, threading.Lock())
    overall_end_ts = time.time()
    overall_elapsed = overall_end_ts - overall_start_ts
    print(f"[RUN] Total elapsed: {overall_elapsed:.1f}s (start {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_ts))} -> end {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end_ts))})")
    
if __name__ == "__main__":
    run()

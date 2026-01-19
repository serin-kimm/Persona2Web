from AgentOccam.obs_opt import parse_node_descendants, parse_node_ancestors, parse_node_siblings, action_set_invisible, action_set_visible, action_set_visible_if_with_name, translate_node_to_str, construct_new_DOM_with_visible_nodes
from AgentOccam.llms.claude import call_claude, call_claude_with_messages, arrange_message_for_claude
from AgentOccam.llms.mistral import call_mistral, call_mistral_with_messages, arrange_message_for_mistral
from AgentOccam.llms.cohere import call_cohere, call_cohere_with_messages, arrange_message_for_cohere
from AgentOccam.llms.llama import call_llama, call_llama_with_messages, arrange_message_for_llama
from AgentOccam.llms.titan import call_titan, call_titan_with_messages, arrange_message_for_titan
from AgentOccam.llms.gpt import call_gpt, call_gpt_with_messages, arrange_message_for_gpt
from AgentOccam.llms.gemini import call_gemini, call_gemini_with_messages, arrange_message_for_gemini
from AgentOccam.llms.qwen import (
    call_gpt as call_qwen,
    call_gpt_with_messages as call_qwen_with_messages,
    arrange_message_for_gpt as arrange_message_for_qwen,
)
from AgentOccam.utils import CURRENT_DIR, HOMEPAGE_URL
from AgentOccam.retriever import MemoryBankRetriever
from AgentOccam.obs_opt import prune_tree, translate_node_to_str


from typing import Dict, Any 
import re
import copy
import os
from functools import partial
import random
import json
import time

import warnings
warnings.filterwarnings("ignore")


DEFAULT_DOCUMENTED_INTERACTION_ELEMENTS = ["observation", "action"]
DEFAULT_ONLINE_INTERACTION_ELEMENTS = ["url", "observation"]
MODEL_FAMILIES = ["claude", "mistral", "cohere", "llama", "titan", "gpt", "gemini", "qwen"]
CALL_MODEL_MAP = {
    "claude": call_claude,
    "mistral": call_mistral,
    "cohere": call_cohere,
    "llama": call_llama,
    "titan": call_titan,
    "gpt": call_gpt,
    "gemini": call_gemini,
    "qwen": call_qwen,
}
CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP = {
    "claude": call_claude_with_messages,
    "mistral": call_mistral_with_messages,
    "cohere": call_cohere_with_messages,
    "llama": call_llama_with_messages,
    "titan": call_titan_with_messages,
    "gpt": call_gpt_with_messages,
    "gemini": call_gemini_with_messages,
    "qwen": call_qwen_with_messages,
}
ARRANGE_MESSAGE_FOR_MODEL_MAP = {
    "claude": arrange_message_for_claude,
    "mistral": arrange_message_for_mistral,
    "cohere": arrange_message_for_cohere,
    "llama": arrange_message_for_llama,
    "titan": arrange_message_for_titan,
    "gpt": arrange_message_for_gpt,
    "gemini": arrange_message_for_gemini,
    "qwen": arrange_message_for_qwen,
}

class Agent:
    def __init__(self, config, objective, prompt_template):
        self.config = config
        self.objective = objective
        self.prompt_template = prompt_template

        if hasattr(self.config, "documented_interaction_elements"):
            self.previous_interactions = {k: [] for k in set(DEFAULT_DOCUMENTED_INTERACTION_ELEMENTS+self.config.documented_interaction_elements)}
        else:
            self.previous_interactions = {k: [] for k in DEFAULT_DOCUMENTED_INTERACTION_ELEMENTS}
        if hasattr(self.config, "online_interaction_elements"):
            self.online_interaction = {k: None for k in set(DEFAULT_ONLINE_INTERACTION_ELEMENTS+self.config.online_interaction_elements)}
        else:
            self.online_interaction = {k: None for k in DEFAULT_ONLINE_INTERACTION_ELEMENTS}

        self.model_family = [model_family for model_family in MODEL_FAMILIES if model_family in self.config.model][0]
        self.call_model = partial(CALL_MODEL_MAP[self.model_family], model_id=self.config.model)
        self.call_model_with_message = partial(CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP[self.model_family], model_id=self.config.model)
        self.arrange_message_for_model = ARRANGE_MESSAGE_FOR_MODEL_MAP[self.model_family]

    def shift_model(self, model_id):
        self.model_family = [model_family for model_family in MODEL_FAMILIES if model_family in model_id][0]
        self.call_model = partial(CALL_MODEL_MAP[self.model_family], model_id=model_id)
        self.call_model_with_message = partial(CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP[self.model_family], model_id=model_id)
        self.arrange_message_for_model = ARRANGE_MESSAGE_FOR_MODEL_MAP[self.model_family]

    def prune_message_list(self, message_list):
        return self.merge_adjacent_text([m for m in message_list if not (m[0]=="text" and len(m[1])==0)])
    
    def merge_adjacent_text(self, message_list):
        merged_list = []
        current_tuple = None
        
        for tup in message_list:
            if tup[0] == "text":
                if current_tuple:
                    current_tuple = (current_tuple[0], current_tuple[1] + tup[1])
                else:
                    current_tuple = tup
            else:
                if current_tuple:
                    merged_list.append(current_tuple)
                    current_tuple = None
                merged_list.append(tup)
        
        if current_tuple:
            merged_list.append(current_tuple)
        
        return merged_list

    def _clip_text_by_chars(self, text: str, max_chars: int) -> str:
        """Truncate given text by character count while preserving both ends."""
        try:
            if not isinstance(text, str) or max_chars <= 0:
                return ""
            if len(text) <= max_chars:
                return text
            head = max_chars * 7 // 10
            tail = max_chars - head
            return f"{text[:head]}\n...[TRUNCATED]...\n{text[-tail:]}"
        except Exception:
            # Safe truncation without crash even for abnormal data
            try:
                s = str(text)
                return s[:max_chars]
            except Exception:
                return ""

    def _clip_message_list_text(self, message_list, per_text_max_chars: int):
        """Truncate content of ('text', content) message items individually."""
        try:
            clipped = []
            for item in (message_list or []):
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "text":
                    clipped.append(("text", self._clip_text_by_chars(str(item[1]), per_text_max_chars)))
                else:
                    clipped.append(item)
            return clipped
        except Exception:
            return message_list

    def _is_context_exceeded_error(self, err: Exception) -> bool:
        """Detect context exceeded error messages."""
        s = (str(err) or "").lower()
        keys = [
            "maximum context length",
            "maximum context",
            "reduce the length of the input messages",
            "input tokens",
            "context length is",  # general pattern
        ]
        return any(k in s for k in keys)

    def _invoke_with_context_fallback(self, system_prompt, online_input):
        """
        When message length exceeded error occurs in 'this call',
        truncate online_input and retry 1~2 times.
        """
        # First attempt
        try:
            return self.call_model_with_message(
                system_prompt=system_prompt,
                messages=self.arrange_message_for_model(online_input)
            )
        except Exception as e:
            if not self._is_context_exceeded_error(e):
                raise
            # Second attempt: conservative truncation
            try_input = self._clip_message_list_text(online_input, per_text_max_chars=25000)
            try:
                return self.call_model_with_message(
                    system_prompt=system_prompt,
                    messages=self.arrange_message_for_model(try_input)
                )
            except Exception as e2:
                if not self._is_context_exceeded_error(e2):
                    raise
                # Third attempt: more aggressive truncation
                try_input2 = self._clip_message_list_text(online_input, per_text_max_chars=15000)
                return self.call_model_with_message(
                    system_prompt=system_prompt,
                    messages=self.arrange_message_for_model(try_input2)
                )

    def _invoke_simple_with_context_fallback(self, prompt: str):
        """
        When context exceeded error occurs in single string prompt call,
        truncate the prompt for this call and retry.
        """
        try:
            return self.call_model(prompt=prompt)
        except Exception as e:
            if not self._is_context_exceeded_error(e):
                raise
            # Second attempt
            clipped = self._clip_text_by_chars(prompt, 100000)
            try:
                return self.call_model(prompt=clipped)
            except Exception as e2:
                if not self._is_context_exceeded_error(e2):
                    raise
                # Third attempt
                clipped2 = self._clip_text_by_chars(prompt, 50000)
                return self.call_model(prompt=clipped2)

    
    def get_step(self):
        return len(self.previous_interactions["action"])

    def update_objective(self, objective):
        self.objective = objective

    def update_online_state(self, **online_states):
        for k in online_states.keys():
            if k in self.online_interaction.keys():
                self.online_interaction[k] = online_states[k]

    def update_history(self, **interaction_dict):
        for k in interaction_dict.keys():
            if k in self.previous_interactions.keys():
                self.previous_interactions[k].append(interaction_dict[k])

    def equal_history_length(self):
        lengths = [len(self.previous_interactions[k]) for k in self.previous_interactions.keys()]
        return (len(set(lengths)) == 1)

    def parse_elements(self, text, key_list):
        element_dict = {}
        for k in key_list:
            # _match = re.search(rf'{k.upper()}:\s*(.*?)\s*(?=\n[A-Z\d\s\W]*: *\n|$)', text, re.DOTALL)
            _match = re.search(rf'{k.upper()}:\s*(.*?)\s*(?=\n[A-Z\s]*:|$)', text, re.DOTALL)
            element_dict[k] = _match.group(1).strip() if _match else ""
        return element_dict

    def get_output_specifications(self):
        output_specifications = "\n".join([f"{o.upper()}:\n" + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "output_specifications", "{}.txt".format(o.replace(" ", "_"))), "r").readlines()) for o in self.config.output])
        return output_specifications

    def parse_stipulated_action_list(self, text: str, action: str, actions: list) -> str:
        pattern = rf'({re.escape(action)}\s*(.*?))(?=\n(?:{"|".join(map(re.escape, actions))})|$)'
        return [match[0].strip() for match in re.findall(pattern, text, re.DOTALL)]

    def parse_str_to_action_list(self, text:str, actions: list):
        remain_text = copy.deepcopy(text)
        action_list = []
        while remain_text:
            find_action = False
            for action in actions:
                if remain_text.startswith(action):
                    match = re.search(rf'({re.escape(action)}\s*(.*?))(?=\n(?:{"|".join(map(re.escape, actions))})|$)', remain_text, re.DOTALL)
                    action_list.append(match[0])
                    remain_text = remain_text[len(match[0]):].strip()
                    find_action = True
            if not find_action:
                break
        return action_list
    
    def get_observation_text(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx:
                return self.previous_interactions["observation"][idx]["text"]
            return self.online_interaction["observation"]["text"]
        elif isinstance(self.online_interaction["observation"], str):
            if idx:
                return self.previous_interactions["observation"][idx]
            return self.online_interaction["observation"]
        
    def get_observation_image(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx:
                return self.previous_interactions["observation"][idx]["image"]
            return self.online_interaction["observation"]["image"]
        elif isinstance(self.online_interaction["observation"], str):
            return None
        
    def get_observation_node(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx != None:
                return self.previous_interactions["observation"][idx]["node"]
            return self.online_interaction["observation"]["node"]
        elif isinstance(self.online_interaction["observation"], str):
            return None
        
    def get_observation_node_str(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx != None:
                return self.previous_interactions["observation"][idx]["node_str"]
            return translate_node_to_str(self.online_interaction["observation"]["node"], mode="name_only")
        elif isinstance(self.online_interaction["observation"], str):
            return None
        
    def del_observation_node(self):
        if isinstance(self.online_interaction["observation"], str):
            return
        if isinstance(self.online_interaction["observation"], dict):
            for idx in range(len(self.previous_interactions["observation"])):
                if "node" in self.previous_interactions["observation"][idx].keys() and self.previous_interactions["observation"][idx]["node"]:
                    node_str = translate_node_to_str(self.previous_interactions["observation"][idx]["node"], mode="name_only")
                    self.previous_interactions["observation"][idx]["node_str"] = node_str
                    self.previous_interactions["observation"][idx]["node"].delete_tree()
                    self.previous_interactions["observation"][idx]["node"] = None

class PlanTreeNode:
    def __init__(self, id, type, text, level, url, step):
        self.visible = True
        self.id = id
        self.type = type
        self.text = text
        self.level = level
        self.url = url
        self.step = step
        self.children = []
        self.parent = None
        self.note = []
        self.hint = []
        self.resume_reason = []
        self.steps_taken = []

    def reset(self):
        self.visible = True
        self.note = []
        self.hint = []
        self.steps_taken = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def search_node_by_id(self, target_id):
        if self.visible and self.id == target_id:
            return self
        for child in self.children:
            result = child.search_node_by_id(target_id)
            if result:
                return result
        return None
    
    def traverse(self, action=None, tree_buffer=[]):
        res_action = action(self)
        if res_action:
            if isinstance(res_action, list):
                tree_buffer.extend(res_action)
            else:
                tree_buffer.append(res_action)
        for child in self.children:
            child.traverse(action, tree_buffer=tree_buffer)

class QAActor(Agent):
    def __init__(self, config, objective, prompt_template):
        super().__init__(config, objective, prompt_template)
    def get_instruction(self):
        return self.prompt_template["instruction_template"]
    def get_online_input(self):
        return [("text", self.prompt_template["input_template"].replace("{current_observation}", self.get_observation_text()).replace("{objective}", self.objective))]
    def get_action(self, instruction, online_input):
        model_response = self._invoke_with_context_fallback(instruction, online_input)
        action_elements = self.parse_elements(text=model_response, key_list=self.config.output)
        action = action_elements["response"]
        action_elements["action"] = f"note [{action}]"
        action_elements["instruction"] = instruction
        action_elements["input"] = online_input
        return model_response, action_elements
    
class PlanningActor(Agent):
    def __init__(self, config, objective, prompt_template):
        super().__init__(config, objective, prompt_template)
        self.instruction = None

    def get_planning_specifications(self):
        return "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "planning_specifications", f"{p}.txt"), "r").readlines()) for p in self.config.planning_command])
    
    def get_instruction(self):
        if self.instruction:
            return self.instruction
        output_specifications = self.get_output_specifications()
        self.instruction = self.prompt_template["instruction_template"].replace("{output_specifications}", output_specifications).replace("{planning_specifications}", self.get_planning_specifications())
        return self.instruction
    
    def get_online_input(self):
        return None
    
    def get_action(self, instruction, online_input):
        model_response = self._invoke_with_context_fallback(instruction, online_input)
        action_elements = self.parse_elements(text=model_response, key_list=self.config.output)
        action_elements["action"] = copy.deepcopy(action_elements["plan"])
        del action_elements["plan"]
        action_elements["reason"] = "N/A"
        action_elements["instruction"] = instruction
        action_elements["input"] = online_input
        return model_response, action_elements

class ReflectionActor(Agent):
    def __init__(self, config, objective, prompt_template):
        super().__init__(config, objective, prompt_template)
        self.instruction = None

    def get_planning_specifications(self):
        return "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "planning_specifications", f"{p}.txt"), "r").readlines()) for p in self.config.planning_command])
    
    def get_navigation_specifications(self):
        return "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "navigation_specifications", f"{n}.txt"), "r").readlines()) for n in self.config.navigation_command])
    
    def get_instruction(self):
        if self.instruction:
            return self.instruction
        output_specifications = self.get_output_specifications()
        planning_specifications = self.get_planning_specifications()
        navigation_specifications = self.get_navigation_specifications()
        instruction = self.prompt_template["instruction_template"]
        instruction = instruction.replace("{output_specifications}", output_specifications)
        instruction = instruction.replace("{planning_specifications}", planning_specifications)
        instruction = instruction.replace("{navigation_specifications}", navigation_specifications)
        self.instruction = instruction
        return self.instruction
    
    def get_online_input(self):
        return None
    
    def get_action(self, instruction, online_input):
        model_response = self._invoke_with_context_fallback(instruction, online_input)
        action_elements = self.parse_elements(text=model_response, key_list=self.config.output)
        action_elements["instruction"] = instruction
        action_elements["input"] = online_input
        return model_response, action_elements

IDENTITY_CLASS_MAP = {
    "QA": QAActor,
    "planning": PlanningActor,
    "reflection": ReflectionActor,
}

class Actor(Agent):
    def __init__(self, config, objective, prompt_template, plan_tree_node):
        super().__init__(config, objective, prompt_template)
        self.plan_tree_root = plan_tree_node
        self.active_node = plan_tree_node
        self.output_specifications = None
        self.planning_specifications = None
        self.navigation_specifications = None
        self.criticism_element_list = None

        self.output_play_path = os.path.join(CURRENT_DIR, f"play-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"play.txt")
        self.output_trash_path = os.path.join(CURRENT_DIR, f"trash-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"trash.txt")

        self.identities = []
        if hasattr(self.config, "identities"):
            i = 0
            while hasattr(self.config.identities, f"identity_{i}"):
                identity_config = getattr(self.config.identities, f"identity_{i}")
                self.identities.append(IDENTITY_CLASS_MAP[identity_config.name](identity_config, objective=objective, prompt_template=prompt_template[identity_config.name]))
                i += 1

        self.disable_memory_check = getattr(config, 'disable_memory_check', False)

        # Set memory prompt paths
        self.memory_prompt_dir = os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "memory")

        # Load memory prompt templates
        self.memory_instruction_template = self.load_prompt_template("memory_instruction.txt")
        self.memory_input_template = self.load_prompt_template("memory_input_context.txt")

        # Memory context cache (accumulated)
        self.memory_context_cache = {}
        self.accessed_memory_indices = []

    def load_prompt_template(self, filename):
        """Load prompt template file"""
        filepath = os.path.join(self.memory_prompt_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Warning: Prompt template {filename} not found")
            return ""
    
    def get_actor_instruction_with_memory(self, memory_context):
        """Generate instruction including memory context"""
        base_instruction = self.get_actor_instruction()

        # Fill in template values
        memory_instruction = self.memory_instruction_template.format(
            memory_query=memory_context.get('query', ''),
            memory_answer=memory_context.get('answer', ''),
            memories_count=memory_context.get('memories_count', 0)
        )
        
        # Combine base instruction with memory instruction
        return f"{base_instruction}\n\n{memory_instruction}"
    
    def get_online_input_with_memory(self, criticism_elements, memory_context):
        return self.get_online_input(criticism_elements)

    def _combine_memory_context(self, memory_cache_dict):
        """Combine accumulated memory cache into single context"""
        if not memory_cache_dict:
            return {'query': '', 'answer': '', 'memories_count': 0}
        
        # Combine all queries and answers
        combined_queries = []
        combined_answers = []
        total_memories = 0
        
        for query, context in memory_cache_dict.items():
            combined_queries.append(query)
            combined_answers.append(context['answer'])
            total_memories += context.get('memories_count', 0)
        
        return {
            'query': '; '.join(combined_queries),
            'answer': '\n\n'.join(combined_answers),
            'memories_count': total_memories
        }

    def update_online_state(self, **online_states):
        super().update_online_state(**online_states)
        for identity in self.identities:
            identity.update_online_state(**online_states)

    def is_planning(self, action):
        for c in self.config.planning_command:
            if action.startswith(c):
                return c
        return False

# Memory bank access request
    def is_memoryaccess(self, action):
        for c in self.config.memoryaccess_command:
            if action.startswith(c):
                return c
        return False
    
    def is_navigation(self, action):
        action_without_note = re.sub(rf'(note\s*(.*?))(?=\n(?:{"|".join(map(re.escape, self.config.navigation_command))})|$)', "", action).strip()
        for c in self.config.navigation_command:
            if action_without_note.startswith(c):
                return c
        return False
    
    def is_valid_action(self, action_str):
        action = (
            action_str.split("[")[0].strip()
            if "[" in action_str
            else action_str.split()[0].strip()
        )
        match action:
            case "click":
                match = re.search(r"click ?\[(\d+)\]", action_str)
                if not match:
                    return False
                element_id = match.group(1)
                if element_id in self.get_observation_text():
                    return True
                return False
            case "type":
                if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                    action_str += " [1]"

                match = re.search(
                    r"type ?\[(\d+)\] ?\[(.*)\] ?\[(\d+)\]", action_str, re.DOTALL
                )
                if not match:
                    return False
                element_id, text, enter_flag = (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                )
                enter_flag = True if enter_flag == "1" else False
                if enter_flag:
                    text += "\n"
                if element_id in self.get_observation_text():
                    return True
            case "go_back":
                return True
            case "go_home":
                return True
            case "note":
                return True
            case "stop":
                return True
            case "branch":
                return True
            case "prune":
                return True
            case "goto":
                return True
            case "scroll":
                return True
            case "memory_access":
                return True

    def are_valid_actions(self, actions):
        action_list = self.parse_str_to_action_list(actions, self.config.planning_command+self.config.navigation_command+self.config.memoryaccess_command+["goto"])
        if not action_list:
            return False
        for action in action_list:
            if not self.is_valid_action(action):
                return False
        return True

    def get_previous_plans(self, verbose=False):
        def action_return_visible_node(node, verbose=False):
            if node.id == self.active_node.id:
                basic = "\t" * node.level + f"[{node.id}] (Active Plan) {node.text}" if node.visible else None
            else:
                basic = "\t" * node.level + f"[{node.id}] {node.text}" if node.visible else None
            if basic and len(node.resume_reason) > 0:
                basic += f" # Was resumed to this step {len(node.resume_reason)} time(s) for:"
                for i, reason in enumerate(node.resume_reason):
                    basic += f" {i}. {reason}"
            if verbose and basic and len(node.note) > 0:
                for i, note in enumerate(node.note):
                    basic += "\n" + "\t" * node.level + f"Note {i}. {note}"
            return basic
        plan_tree_buffer = []
        parse_node_descendants(self.plan_tree_root, partial(action_return_visible_node, verbose=verbose), tree_buffer=plan_tree_buffer)
        return "\n".join(plan_tree_buffer)
    
    def get_active_plan(self):
        return f"[{self.active_node.id}] {self.active_node.text}"
    
    def get_interaction_history(self, interaction_history_config=False, mode="highlight"):
        interaction_history_config = interaction_history_config if interaction_history_config else self.config.interaction_history

        previous_observation = []
        for i in self.active_node.steps_taken:
            if self.get_observation_node_str() and self.get_observation_node_str(i) and not self.get_observation_node_str() == self.get_observation_node_str(i):
                if self.previous_interactions["observation highlight"][i] and mode == "highlight" and len(translate_node_to_str(self.previous_interactions["observation highlight"][i], mode="name_only", retained_ids=self.previous_interactions["retained element ids"][i]).split()) < 200:
                    try:
                        previous_observation.append({"text": translate_node_to_str(self.previous_interactions["observation highlight"][i], mode="name_only", retained_ids=self.previous_interactions["retained element ids"][i]), "image": self.get_observation_image(i)})
                    except:
                        print(i, self.previous_interactions["observation"][i]["text"])
                        raise ValueError("Cannot translate highlight node to text.")
                else:
                    previous_observation.append({"text": self.previous_interactions["observation summary"][i], "image": self.get_observation_image(i)})
            elif not self.get_observation_node() or mode == "full":
                if len(self.get_observation_text(i).split()) < 200:
                    previous_observation.append({"text": self.get_observation_text(i), "image": self.get_observation_image(i)})
                else:
                    previous_observation.append({"text": self.previous_interactions["observation summary"][i], "image": self.get_observation_image(i)})
            else:
                previous_observation.append({"text": "The same as the CURRENT OBSERVATION (see below CURRENT OBSERVATION section).", "image": self.get_observation_image(i)})

        previous_observation_summary = [self.previous_interactions["observation summary"][i] for i in self.active_node.steps_taken]

        def get_text(obs):
            if isinstance(obs, dict):
                return obs["text"]
            elif isinstance(obs, str):
                return obs

        def get_image(obs):
            if isinstance(obs, dict):
                return obs.get("image")
            elif isinstance(obs, str):
                return obs

        if interaction_history_config.step_num == "all":
            textual_observations = [get_text(obs) for obs in previous_observation] if interaction_history_config.verbose else previous_observation_summary
            visual_observations = [get_image(obs) for obs in previous_observation]
        else:
            textual_observations = previous_observation_summary[:-interaction_history_config.step_num]
            visual_observations = [None] * len(previous_observation_summary[:-interaction_history_config.step_num])
            textual_observations += [get_text(obs) for obs in previous_observation][-interaction_history_config.step_num:] if interaction_history_config.verbose else previous_observation_summary[-interaction_history_config.step_num:]
            visual_observations += [get_image(obs) for obs in previous_observation][-interaction_history_config.step_num:]

        plans = [self.previous_interactions["plan"][i] for i in self.active_node.steps_taken]
        reasons = [self.previous_interactions["reason"][i] for i in self.active_node.steps_taken]
        actions = [self.previous_interactions["action"][i] for i in self.active_node.steps_taken]
            
        if "image" in interaction_history_config.type:
            message_list = []
            for step, (obs, vi_obs, plan, reason, action) in enumerate(zip(textual_observations, visual_observations, plans, reasons, actions)):
                message_list.append(("text", f"<step_{step}_interaction>\n"))
                if vi_obs:
                    message_list.append(("text", "VISUAL OBSERVATION:\n"))
                    message_list.append(("image", vi_obs))
                if self.active_node.id != 0:
                    message_list.append(("text", f"TEXTUAL OBSERVATION:\n{obs}\nACTIVE PLAN:\n{plan}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"))
                else:
                    message_list.append(("text", f"TEXTUAL OBSERVATION:\n{obs}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"))
            return self.prune_message_list(message_list=message_list)
        else:
            message = ""
            for step, (obs, plan, reason, action) in enumerate(zip(textual_observations, plans, reasons, actions)):
                if self.active_node.id != 0:
                    message += f"<step_{step}_interaction>\nOBSERVATION:\n{obs}\nACTIVE PLAN:\n{plan}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n" # f"<step_{step}_interaction>\nOBSERVATION:\n{obs}\nACTIVE PLAN:\n{plan}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"
                else:
                    message += f"<step_{step}_interaction>\nOBSERVATION:\n{obs}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n" # f"<step_{step}_interaction>\nOBSERVATION:\n{obs}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"
            return self.prune_message_list(message_list=[("text", message)])
        
    def pre_process_atomic_actions(self, atomic_action_list=["combobox"]):
        if self.get_observation_node() and "combobox" in atomic_action_list:
            self.online_interaction["observation"]["text"] = translate_node_to_str(self.get_observation_node(), mode="concise", hidden_roles=["menu", "combobox", "listbox"])

    def get_online_input(self, criticism_elements):
        input_template = self.prompt_template["input_template"]
        input_prefix, input_suffix = input_template.split("{input}")
        INPUT_TYPE_TO_CONTENT_MAP = {
            "step": self.get_step(),
            "objective": self.objective,
            "previous plans": self.get_previous_plans(verbose=True),
            "interaction history": self.get_interaction_history(),
            "current observation": self.get_observation_text(),
            "current visual observation": self.get_observation_image()
        }
        input_list = []
        for input_type in self.config.input:
            input_content = None
            if input_type == "current visual observation":
                continue
            elif input_type in INPUT_TYPE_TO_CONTENT_MAP.keys():
                input_content = INPUT_TYPE_TO_CONTENT_MAP[input_type]
            elif input_type.startswith("critic: ") and criticism_elements and input_type[len("critic: "):] in criticism_elements.keys() and criticism_elements[input_type[len("critic: "):]]:
                input_type = input_type[len("critic: "):]
                input_content = criticism_elements[input_type]
                input_type = "FROM USER: " + input_type
            if input_content and isinstance(input_content, str):
                input_list.append(("text", f"{input_type.upper()}:\n{input_content}\n"))
            elif input_content and isinstance(input_content, list):
                input_list.append(("text", f"{input_type.upper()}:\n"))
                input_list += input_content if len(input_content) > 0 else ["N/A"]

        if "image" in self.config.current_observation.type:
            input_type = "current visual observation"
            input_list.append(("text", f"{input_type.upper()}:\n"))
            input_list.append(("image", INPUT_TYPE_TO_CONTENT_MAP["current visual observation"]))

        return self.prune_message_list(message_list=[("text", input_prefix)] + input_list + [("text", input_suffix)])
    
    def get_planning_specifications(self):
        if self.planning_specifications:
            return self.planning_specifications
        self.planning_specifications = "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "planning_specifications", f"{p}.txt"), "r").readlines()) for p in self.config.planning_command])
        return self.planning_specifications
    
    def get_navigation_specifications(self):
        if self.navigation_specifications:
            return self.navigation_specifications
        self.navigation_specifications = "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "navigation_specifications", f"{n}.txt"), "r").readlines()) for n in self.config.navigation_command])
        return self.navigation_specifications
    
    def get_actor_instruction(self, examples=None):
        if self.config.planning_command:
            instruction = self.prompt_template["instruction_template"]["with_planning"]
        else:
            instruction = self.prompt_template["instruction_template"]["without_planning"]
        output_specifications = self.get_output_specifications()
        planning_specifications = self.get_planning_specifications()
        navigation_specifications = self.get_navigation_specifications()
        instruction = instruction.replace("{output_specifications}", output_specifications)
        instruction = instruction.replace("{planning_specifications}", planning_specifications)
        instruction = instruction.replace("{navigation_specifications}", navigation_specifications)

        example_source = examples if examples is not None else self.prompt_template.get("examples", [])
        if len(example_source) > 0:
            instruction += f"\n\n## Here are a few examples:"
            for i, example in enumerate(example_source):
                example_input = example["input"]
                example_output = example["output"]
                if "example_template" in self.prompt_template.keys():
                    instruction += "\n\n"
                    instruction += self.prompt_template.get("example_template", "| Example {i}\n### Input:\n{example_input}\n### Response: Let's think step by step.\n{example_response}").replace("{i}", i).replace("{example_input}", example_input).replace("{example_output}", example_output)
                else:
                    instruction += f"\n\n| Example {i}\n\n### Input:\n{example_input}\n\n### Response: Let's think step by step.\n{example_output}"
        
        if self.get_step() == self.config.others.max_steps - 1:
            instruction += f"\n\nWARNING: You have a {self.config.others.max_steps}-step budget, and this would be your FINAL STEP. Wrap up your observations and return your answer with `stop [answer]` to maximize the reward."
        # else:
        #     instruction += f"\n\nWARNING: You have a {self.config.others.max_steps}-step budget, and there are {self.config.others.max_steps-self.get_step()} remaining attempts."

        return instruction
    
    def verbose(self, instruction, online_input, model_response_list, action_element_list):
        action_element_keys = [k for k in self.config.play if k in action_element_list[0].keys()]
        other_play_keys = [k for k in self.config.play if k not in action_element_list[0].keys()]

        VERBOSE_TO_CONTENT_MAP = {
            "step": self.get_step(),
            "objective": self.objective,
            "previous plans": self.get_previous_plans(verbose=True),
            "url": self.online_interaction["url"],
            "observation": self.get_observation_text(),
            "response": "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n".join([f"|\tAgent {i}:\n{model_response}" for i, model_response in enumerate(model_response_list[:self.config.number])]) if self.config.number > 1 else model_response_list[0],
            "instruction": instruction,
            "online input": "\n".join([i[1] for i in online_input if i[0]=="text"]),
            "alter ego response": "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n".join(["|\tAgent {}:\n{}".format(identity.config.name, response) for identity, response in zip(self.identities, model_response_list[self.config.number:])])
        }

        if self.config.others.verbose > 0 and self.config.verbose > 0:
            with open(self.output_trash_path, "a") as af:
                af.write("-"*32+"ACTOR"+"-"*32+"\n")
            for t in self.config.trash:
                content = VERBOSE_TO_CONTENT_MAP.get(t, "")
                with open(self.output_trash_path, "a") as af:
                    af.write(f"{t.upper()}:\n{content}\n\n")
            with open(self.output_play_path, "w") as _:
                pass
            for p in other_play_keys:
                content = VERBOSE_TO_CONTENT_MAP.get(p, "")
                with open(self.output_play_path, "a") as af:
                    af.write(f"{p.upper()}:\n{content}\n\n")
            for i, action_elements in enumerate(action_element_list):
                if len(action_element_list) > 1:
                    with open(self.output_play_path, "a") as af:
                        af.write("-"*32+f"AGENT {i}"+"-"*32+"\n")
                for action_element_key in action_element_keys:
                    content = action_elements.get(action_element_key, "N/A")
                    with open(self.output_play_path, "a") as af:
                        af.write(f"{action_element_key.upper()}:\n{content}\n\n")
    
    def parse_plan(self, planning):
        planning_type = self.is_planning(action=planning)
        match = re.search(
            rf"{planning_type} ?\[(\d+)\] ?\[(.+)\]", planning, re.DOTALL
        )
        if not match:
            raise ValueError("Invalid planning command.")
        node_id, planning_content = (
            int(match.group(1)),
            match.group(2)
        )
        return planning_type, node_id, planning_content
    
    def prune_planning(self, node:PlanTreeNode, planning_content):
        def set_invisible(node:PlanTreeNode):
            node.visible = False
        def return_steps_taken(node:PlanTreeNode):
            return [node.step] + node.steps_taken
        after_node = False
        if node.id > 0:
            for child in node.parent.children:
                if not after_node and child != node:
                    continue
                elif child == node:
                    after_node = True
                    continue
                child.visible = False
        node.traverse(set_invisible)
        node.reset()
        steps_taken = []
        node.traverse(action=return_steps_taken, tree_buffer=steps_taken)
        node.steps_taken = sorted(list(set(steps_taken)), reverse=False)
        node.resume_reason.append(planning_content)
        navigation = f"goto [{node.url}] [1]"
        self.active_node = node
        return navigation
    
    def branch_planning(self, node, planning_content):
        new_node = PlanTreeNode(id=self.active_node.id+1, type=type, text=planning_content, level=node.level+1, url=self.online_interaction["url"], step=self.get_step())
        self.active_node = new_node
        node.add_child(new_node)
    
    def planning(self, action):
        if action and self.is_planning(action):
            try:
                planning_type, node_id, planning_content = self.parse_plan(planning=action)
                node = self.plan_tree_root.search_node_by_id(node_id)
                if not node:
                    raise ValueError(f"Invalid node id {node_id}: {action}.")
                if planning_type == "prune":
                    navigation_action = self.prune_planning(node=node, planning_content=planning_content)
                    return navigation_action
                elif planning_type == "branch":
                    self.branch_planning(node=node, planning_content=planning_content)
                else:
                    raise ValueError(f"Invalid planning operation {planning_type}: {action}.")
            except Exception as e:
                print("Invalid plan node:", str(e))
                flaw_node = self.active_node
                flaw_node.note.append(f"You previously generate plan \"{action}\", which has INVALID syntax. User planning command like `branch [parent_plan_id] [new_subplan_intent]` or `prune [resume_plan_id] [reason]`.")
        else:
            self.active_node.steps_taken.append(self.get_step())
        return None
    
    def go_home(self, action):
        if "go_home" in action:
            return f"goto [{HOMEPAGE_URL}] [1]"
        return None
    
    def parse_action(self, action_str):
        try:
            DOM_root_node = self.get_observation_node()
            action_str = action_str.strip()
            action = (
                action_str.split("[")[0].strip()
                if "[" in action_str
                else action_str.split()[0].strip()
            )
            match action:
                case "click":
                    match = re.search(r"click ?\[(\d+)\]", action_str)
                    if not match:
                        raise ValueError(f"Invalid click action {action_str}")
                    element_id = match.group(1)
                    node = DOM_root_node.search_node_by_id(element_id)
                    return f"click [{element_id}] ({node.role} {node.name})"
                case "hover":
                    match = re.search(r"hover ?\[(\d+)\]", action_str)
                    if not match:
                        raise ValueError(f"Invalid hover action {action_str}")
                    element_id = match.group(1)
                    node = DOM_root_node.search_node_by_id(element_id)
                    return f"hover [{element_id}] ({node.role} {node.name})"
                case "type":
                    if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                        action_str += " [1]"

                    match = re.search(
                        r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str
                    )
                    if not match:
                        raise ValueError(f"Invalid type action {action_str}")
                    element_id, text, enter_flag = (
                        match.group(1),
                        match.group(2),
                        match.group(3),
                    )
                    enter_flag = True if enter_flag == "1" else False
                    if enter_flag:
                        text += "\n"
                    node = DOM_root_node.search_node_by_id(element_id)
                    return action + f" ({node.name})"
                case "scroll":
                    return action_str
                case "goto":
                    return action
                case "new_tab":
                    return action
                case "go_back":
                    return action
                case "go_forward":
                    return action
                case "stop":
                    return action

            return False
        except:
            return False
    
    def parse_actions_to_element_ids(self, actions):
        action_str_list = []
        for a in self.config.navigation_command:
            action_str_list += self.parse_stipulated_action_list(text=actions, action=a, actions=self.config.planning_command+self.config.navigation_command+["goto"])
        retained_element_ids = []
        for action_str in action_str_list:
            try:
                action_str = action_str.strip()
                action = (
                    action_str.split("[")[0].strip()
                    if "[" in action_str
                    else action_str.split()[0].strip()
                )
                match action:
                    case "click":
                        match = re.search(r"click ?\[(\d+)\]", action_str)
                        if not match:
                            raise ValueError(f"Invalid click action {action_str}")
                        element_id = match.group(1)
                        element_id = int(element_id)
                        retained_element_ids.append(element_id)
                    case "hover":
                        match = re.search(r"hover ?\[(\d+)\]", action_str)
                        if not match:
                            raise ValueError(f"Invalid hover action {action_str}")
                        element_id = match.group(1)
                        element_id = int(element_id)
                        retained_element_ids.append(element_id)
                    case "type":
                        if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                            action_str += " [1]"

                        match = re.search(
                            r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str
                        )
                        if not match:
                            raise ValueError(f"Invalid type action {action_str}")
                        element_id, text, enter_flag = (
                            match.group(1),
                            match.group(2),
                            match.group(3),
                        )
                        element_id = int(element_id)
                        retained_element_ids.append(element_id)
                    case "scroll":
                        pass
                    case "goto":
                        pass
                    case "new_tab":
                        pass
                    case "go_back":
                        pass
                    case "go_forward":
                        pass
                    case "stop":
                        pass
                    case "note":
                        pass

                return retained_element_ids
            except:
                continue

        return retained_element_ids
    
    def take_note(self, action, note_as_action=True):
        if action and "note [" in action:
            none_note_action_list = []
            action_list = self.parse_str_to_action_list(action, actions=self.config.planning_command+self.config.navigation_command+["goto"])
            for a in action_list:
                if "note [" in a:
                    note = re.search(r"note ?\[?(.+)", a, re.DOTALL).group(1)
                    if note.endswith("]"):
                        note = note[:-1]
                    self.active_node.note.append(f"STEP {self.get_step()}: {note}")
                    self.note_buffer = note
                else:
                    none_note_action_list.append(a)
            if note_as_action:
                return action
            return "\n".join(none_note_action_list)
        # action_note = self.parse_action(action)
        # if action_note:
        #     self.active_node.note.append(f"STEP {self.get_step()} ACTION: {action_note}")
        return action
        
    def get_observation_highlight(self, action_elements:dict):
        action_elements["observation highlight idxs"] = copy.deepcopy(action_elements.get("observation highlight", ""))
        DOM_root_node = self.get_observation_node()
        if not DOM_root_node:
            action_elements["observation highlight"] = None
            return
        observation_highlight_idxs = [int(idx.strip()) for idx in action_elements.get("observation highlight", "").split(",") if idx.strip().isdigit()]
        if observation_highlight_idxs:
            parse_node_descendants(node=DOM_root_node, action=action_set_invisible)
            for idx in observation_highlight_idxs:
                try:
                    node = DOM_root_node.search_node_by_id(idx)
                    parse_node_descendants(node=node, action=action_set_visible)
                    parse_node_ancestors(node=node, action=action_set_visible)
                    parse_node_siblings(node=node, action=action_set_visible_if_with_name)
                except:
                    pass
        try: 
            assert DOM_root_node.get_visible_node_number() < 30 and construct_new_DOM_with_visible_nodes(DOM_root=DOM_root_node)
            action_elements["observation highlight"] = construct_new_DOM_with_visible_nodes(DOM_root=DOM_root_node)
            parse_node_descendants(node=DOM_root_node, action=action_set_visible)
        except:
            parse_node_descendants(node=DOM_root_node, action=action_set_visible)
            action_elements["observation highlight"] = None

        action_elements["retained element ids"] = self.parse_actions_to_element_ids(action_elements["action"])

    def parse_action_from_action_candidates(self, action_elements):
        if "action" in action_elements.keys():
            return action_elements
        assert any("action candidates" in k for k in action_elements.keys())
        action_candidates_key = [k for k in action_elements.keys() if "action candidates" in k][0]
        def parse_reasons_and_actions(input_string):
            pattern = r'- reason: \[(.*?)\]\s*(?:- action: \[(.*?)\])?\s*(?:\n|\Z)'

            matches = re.findall(pattern, input_string, re.DOTALL)

            parsed_data = []
            for match in matches:
                reason = match[0].strip()
                action = match[1].strip()
                if reason and action:
                    parsed_data.append({'reason': reason, 'action': action})

            return parsed_data
        action_elements[action_candidates_key] = parse_reasons_and_actions(action_elements[action_candidates_key])
        return action_elements


    def check_captcha_or_error(self):
        """Detect CAPTCHA or error situations in current observation"""
        observation_text = self.get_observation_text()
        if not observation_text:
            return False
            
        # CAPTCHA and error related keywords (case insensitive)
        error_keywords = [
            'captcha', 'recaptcha', 'hcaptcha',
            'verify you are human', 'human verification', 'prove you are not a robot',
            'security check', 'verify your identity',
            'access denied', 'access blocked', 'blocked',
            'rate limit', 'rate limited', 'too many requests',
            'forbidden', '403 forbidden', '401 unauthorized', '429 too many requests',
            'cloudflare', 'please wait while we verify', 'checking your browser',
            'ddos protection', 'bot protection', 'anti-bot',
            'unusual traffic', 'suspicious activity',
            'solve this puzzle', 'complete the captcha',
            'click to verify', 'press and hold', 'press & hold',
            'service unavailable', '503 service unavailable',
            'temporary block', 'temporarily blocked'
        ]
        
        observation_lower = observation_text.lower()
        
        for keyword in error_keywords:
            if keyword in observation_lower:
                print(f"🚨 CAPTCHA/Error detected: '{keyword}' found in observation")
                print(f"Observation excerpt: ...{observation_text[:300]}...")
                return True
                
        return False

    def check_memory_needed_before_action(self):
        if self.get_observation_node():
            # Clean up complex UI like Combobox first
            original_text = self.online_interaction["observation"]["text"]
            self.pre_process_atomic_actions(["combobox", "select", "listbox"])
            
            selection_objective = f"{self.objective} choose select option"
            
            pruned_node = prune_tree(
                objective=selection_objective,
                root_node=self.get_observation_node(),
                mode="node"
            )
            
            filtered_obs = translate_node_to_str(pruned_node, mode="concise")

            filtered_obs = '\n'.join([
                line for line in filtered_obs.split('\n')
                if not line.strip().startswith('image [')
            ])
            filtered_obs = re.sub(r'\s*\[url:.*?\]', '', filtered_obs)
        
            self.online_interaction["observation"]["text"] = original_text

        interaction_history = self.get_interaction_history()

        # history_text = ""
        # if isinstance(interaction_history, list):
        #     for item in interaction_history:
        #         if item[0] == "text":
        #             history_text += item[1]


        # Create context including existing cache information
        cached_info = ""
        if self.memory_context_cache:
            cached_items = []
            # Model branching: apply reasoning-specific summary only for qwen3-next-80b-a3b-thinking
            try:
                actor_model = self.config.agent.actor.model
            except Exception:
                actor_model = ''
            is_qwen_thinking = str(actor_model).strip().lower() == 'qwen3-next-80b-a3b-thinking'
            for query, context in self.memory_context_cache.items():
                if is_qwen_thinking:
                    answer_text = str(context.get('answer', '')).strip()
                    if answer_text:
                        normalized = re.sub(r'\s+', ' ', answer_text).strip()
                        sentences = re.split(r'(?<=[.!?。！？])\s+', normalized)
                        sentences = [s.strip() for s in sentences if s and len(s.strip()) > 0]
                        last_sentence = sentences[-1] if len(sentences) > 0 else normalized
                    else:
                        last_sentence = ""
                    cached_items.append(f"- Query: {query}\n  Answer: {last_sentence}")
                else:
                    ans = str(context.get('answer', '') or '')
                    cached_items.append(f"- Query: {query}\n  Answer: {ans[:200]}...")

            cached_info = f"\nPreviously retrieved memory information:\n" + "\n".join(cached_items)
        print("===LOGGING===")
        print("Cached info:", cached_info)
        # print("Interaction history:", interaction_history)
        print("***obs***:", filtered_obs[:300])
        prompt = f"""Task: {self.objective}
    Current page: {filtered_obs}
    Step: {self.get_step()}
    Interaction history: {interaction_history}{cached_info}

    Think step by step:
    1. What page am I currently on?
    2. What is the IMMEDIATE next action I need to take?
    3. Does THAT SPECIFIC ACTION require much more specified instruction/preference info/history?
    4. Could this action be ENHANCED or MADE MORE SPECIFIC with user's personal information/preferences?
    5. Does the [note] in 'Interaction history' require specified instruction/preference info/history?
    6. IMPORTANT: Is the information I need already available in the 'Previously retrieved memory information' section?

    Only answer YES to 'NEED_MEMORY' if:
    - You MUST choose one RIGHT NOW
    - The task has personal references that affect THIS choice
    - You need to filter based on user preferences/history
    - The information is NOT already available in previously retrieved memory

    Answer NO if:
    - The action is purely navigational (clicking a menu, going to homepage)
    - There are NO options to choose from yet
    - The action cannot be customized with user information
    - The required information is ALREADY available in previously retrieved memory

    CRITICAL CONSTRAINTS FOR ON-DEMAND MODE:
    - You must complete this task within 25 limited steps. Note that if you attempt too many memory accesses, you won't be able to perform sufficient actions.
    - Allocate memory accesses efficiently.
    - Generate ONLY ONE specific memory query at a time
    - Focus on the MOST IMPORTANT information needed for the immediate next action
    - Do NOT generate multiple queries or lists of information
    - Keep the query focused and actionable

    Respond in EXACTLY this format:
    CURRENT_OBSERVATION: Describe information in the CURRENT OBSERVATION section. Emphasize elements and features that are relevant or potentially helpful for fulfilling the objective in detail.
    NEED_MEMORY: [YES or NO]
    QUERY_DETAIL: [ONE specific information needed (detailed domain context, if applicable) - if NO, write N/A]
    REASON: [reason why cannot proceed without user's additional info, or why cached info is sufficient]
    
    Example:
    CURRENT_OBSERVATION: The page shows a list of shirts from various brands including Nike, Adidas, and Puma. There are filters for size, color, and price range. The user has not specified any preferences yet.
    NEED_MEMORY: YES
    QUERY_DETAIL: user's favorite purchased shirts brand
    REASON: Multiple shirts options available but objective mentions 'my favorite brand'"""
        
        response = self._invoke_simple_with_context_fallback(prompt=prompt).strip()
        print("Memory check response:", response)
        lines = response.split('\n')
        parsed = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                parsed[key.strip()] = value.strip()
        
        if parsed.get('NEED_MEMORY') == 'YES':
            query_detail = parsed.get('QUERY_DETAIL')
            reason = parsed.get('REASON')
            
            # On-demand mode: Ensure only ONE focused query
            if query_detail:
                # Remove any list indicators or multiple queries
                query_detail = query_detail.strip()
                # Take only the first line if multi-line
                query_detail = query_detail.split('\n')[0].strip()
                # Remove common list prefixes (bullet points, numbers, dashes)
                query_detail = re.sub(r'^[•\-\*]\s*', '', query_detail)  # Remove bullets/dashes
                query_detail = re.sub(r'^\d+\.\s*', '', query_detail)    # Remove numbered lists
                # Take only the first sentence if multiple sentences exist
                if '.' in query_detail:
                    sentences = [s.strip() for s in query_detail.split('.') if s.strip()]
                    if sentences:
                        query_detail = sentences[0]
                
                query_detail = query_detail.strip()
                print(f"🎯 On-demand mode: Single query extracted - '{query_detail}'")
            
            return {
                "action": f"memory_access [{query_detail}]",
                "reason": f"{reason}",
                "needs_memory": True
            }
        
        return None

    def predict_action(self, criticism_elements):        
        if self.config.debug > 1:
            action_elements = {k: "" for k in self.config.output}
            human_input = input("ACTION: ")
            action_elements["action"] = human_input
            return [action_elements]
        
        # ===== CAPTCHA/error detection and auto-stop =====
        # if self.check_captcha_or_error():
        #     print("🛑 CAPTCHA or error condition detected. Automatically stopping task.")
        #     action_elements = {k: "" for k in self.config.output}
        #     action_elements["action"] = "stop [Task automatically stopped due to CAPTCHA or error condition detected in current observation. Manual intervention required.]"
        #     action_elements["reason"] = "CAPTCHA, security check, or access error detected in the current page observation"
        #     for key in self.config.output:
        #         if key not in action_elements:
        #             action_elements[key] = ""
        #     return [action_elements]

        # ===== Guide user to manually resolve CAPTCHA/error (browser operation) =====
        # if self.check_captcha_or_error():
        #     print("🧩 CAPTCHA/access error detected: Please resolve directly in Chromium window.")
        #     # Short wait (slight tempo adjustment for spam prevention)
        #     try:
        #         time.sleep(1.0)
        #     except Exception:
        #         pass
        #     # Return 'note' action to recheck with new observation in next step (environment performs step and loop continues)
        #     action_elements = {k: "" for k in self.config.output}
        #     action_elements["action"] = f"note [Manual CAPTCHA/access-error resolution in progress. Please solve in Chromium and continue. step={self.get_step()}]"
        #     action_elements["reason"] = "Waiting for user to resolve CAPTCHA/security check in the browser"
        #     for key in self.config.output:
        #         if key not in action_elements:
        #             action_elements[key] = ""
        #     return [action_elements]
        
        if hasattr(self, 'disable_memory_check') and self.disable_memory_check:
            self.pre_process_atomic_actions()
            instruction = self.get_actor_instruction()
            online_input = self.get_online_input(criticism_elements=criticism_elements)
            # print("***obs***:", self.get_observation_text()[:1000])
        else:
            # ===== 1. Check if memory is needed (consider accumulated cache) =====
            print(f"Checking if memory is needed for step {self.get_step()}...")
            memory_check = self.check_memory_needed_before_action()
            
            if memory_check:
                print(f"Memory needed: {memory_check.get('needs_memory')}")
                print(f"Query: {memory_check.get('action')}")
            else:
                print("Memory not needed, proceeding normally")
            print("="*50)

            if memory_check and memory_check.get("needs_memory"):
                # Generate action_elements
                action_elements = {}
                for key in self.config.output:
                    if key == "action":
                        action_elements[key] = memory_check["action"]
                    elif key == "reason":
                        action_elements[key] = memory_check.get("reason", "Need user preferences before proceeding")
                    elif key == "observation description":
                        action_elements[key] = "Accessing memory for user preferences"
                    elif key == "interaction history summary":
                        action_elements[key] = f"Step {self.get_step()}: Requesting memory access"
                    elif key == "observation highlight":
                        action_elements[key] = ""
                    else:
                        action_elements[key] = ""
                
                # Call finalize_action
                action_elements = self.finalize_action(action_elements)
                action_elements["is_memory_action"] = True
                
                # Return as list (memory access ends here)
                return [action_elements]
            else:
                # ===== 2. General processing (apply cache if available) =====
                self.pre_process_atomic_actions()
                
                # Reflect accumulated memory context in instruction
                if self.memory_context_cache:
                    print(f"Using accumulated memory context for planning ({len(self.memory_context_cache)} cached items)...")
                    combined_memory_context = self._combine_memory_context(self.memory_context_cache)
                    instruction = self.get_actor_instruction_with_memory(combined_memory_context)
                    online_input = self.get_online_input_with_memory(
                        criticism_elements=criticism_elements,
                        memory_context=combined_memory_context
                    )
                else:
                    instruction = self.get_actor_instruction()
                    online_input = self.get_online_input(criticism_elements=criticism_elements)
            
        model_response_list = []
        action_element_list = []
        for _ in range(self.config.number):
            get_valid_actions = False
            repetitive_note = False
            invalid_actions = False
            invalid_action_count = 0
            max_invalid_attempts = 3
            while not get_valid_actions:
                if repetitive_note:
                    adjusted_instr = instruction+"\nGenerating the command `note [{}]` will be severely punished! Don't generate repetitive notes!".format(getattr(self, "note_buffer", ""))
                    model_response = self._invoke_with_context_fallback(adjusted_instr, online_input)
                elif invalid_actions:
                    adjusted_instr = instruction+"\nGenerating the command `{}` will be severely punished! Don't generate invalid actions! We don't have that element id in the current observation!".format(invalid_action_str)
                    model_response = self._invoke_with_context_fallback(adjusted_instr, online_input)
                else:
                    model_response = self._invoke_with_context_fallback(instruction, online_input)
                action_elements = self.parse_elements(text=model_response, key_list=self.config.output)
                action_elements = self.parse_action_from_action_candidates(action_elements=action_elements)
                assert not ("action" in action_elements.keys() and any("action candidates" in k for k in action_elements.keys()))
                if "action" in action_elements.keys():
                    if self.are_valid_actions(action_elements["action"]):
                        note_buffer = getattr(self, "note_buffer", "")
                        if note_buffer and f"note [{note_buffer}" in action_elements["action"]:
                            print(f"Repetitive note: {note_buffer}")
                            repetitive_note = True
                            continue
                        get_valid_actions = True
                        action_elements["input"] = online_input
                        model_response_list.append(model_response)
                        action_element_list.append(action_elements)
                    else:
                        invalid_action_str = action_elements["action"]
                        invalid_action_count += 1
                        print(f"Invalid actions: {invalid_action_str} (attempt {invalid_action_count}/{max_invalid_attempts})")
                        
                        if invalid_action_count >= max_invalid_attempts:
                            print(f"Maximum invalid action attempts ({max_invalid_attempts}) reached. Executing stop action.")
                            # Create a stop action
                            action_elements = {
                                "action": "stop [Too many invalid actions attempted]",
                                "reason": f"Stopped due to {max_invalid_attempts} consecutive invalid actions",
                                "input": online_input
                            }
                            model_response_list.append(f"SYSTEM: Stopped due to repeated invalid actions: {invalid_action_str}")
                            action_element_list.append(action_elements)
                            get_valid_actions = True
                            break
                        
                        invalid_actions = True
                elif any("action candidates" in k for k in action_elements.keys()):
                    action_candidates_key = [k for k in action_elements.keys() if "action candidates" in k][0]
                    if isinstance(action_elements[action_candidates_key], str):
                        continue
                    filtered_action_candidates = []
                    note_buffer = getattr(self, "note_buffer", "")
                    for action_reason_pair in action_elements[action_candidates_key]:
                        action = action_reason_pair["action"]
                        reason = action_reason_pair["reason"]
                        if self.are_valid_actions(action):
                            if note_buffer and f"note [{note_buffer}" in action:
                                print(f"Repetitive note: {note_buffer}")
                                repetitive_note = True
                                continue
                            filtered_action_candidates.append({'reason': reason, 'action': action})
                        else:
                            invalid_action_str = action
                            invalid_action_count += 1
                            print(f"Invalid actions: {invalid_action_str} (attempt {invalid_action_count}/{max_invalid_attempts})")
                            invalid_actions = True
                    if filtered_action_candidates:
                        action_elements[action_candidates_key] = filtered_action_candidates
                        get_valid_actions = True
                        action_elements["input"] = online_input
                        model_response_list.append(model_response)
                        action_element_list.append(action_elements)
                    elif invalid_action_count >= max_invalid_attempts:
                        print(f"Maximum invalid action attempts ({max_invalid_attempts}) reached in action candidates. Executing stop action.")
                        # Create a stop action
                        action_elements = {
                            "action": "stop [Too many invalid action candidates attempted]",
                            "reason": f"Stopped due to {max_invalid_attempts} consecutive invalid action candidates",
                            "input": online_input
                        }
                        model_response_list.append(f"SYSTEM: Stopped due to repeated invalid action candidates: {invalid_action_str}")
                        action_element_list.append(action_elements)
                        get_valid_actions = True
                else:
                    raise NotImplementedError("You have to generate either action or action candidates.")
        # if self.config.number != 1:
        if True:
            for identity in self.identities:
                identity_instruction = identity.get_instruction() if identity.get_instruction() else instruction
                identity_online_input = identity.get_online_input() if identity.get_online_input() else online_input
                get_valid_actions = False
                invalid_actions = False
                invalid_action_count = 0
                max_invalid_attempts = 3
                
                while not get_valid_actions:
                    if invalid_actions:
                        model_response, action_elements = identity.get_action(identity_instruction+"\nGenerating the command `{}` will be severely punished! Don't generate invalid actions! We don't have that element id in the current observation!".format(invalid_action_str), identity_online_input)
                    else:
                        model_response, action_elements = identity.get_action(identity_instruction, identity_online_input)      
                    
                    if self.are_valid_actions(action_elements["action"]):
                        get_valid_actions = True
                        model_response_list.append(model_response)
                        action_element_list.append(action_elements)
                    else:
                        invalid_action_str = action_elements["action"]
                        invalid_action_count += 1
                        print(f"Invalid actions: {invalid_action_str} (attempt {invalid_action_count}/{max_invalid_attempts})")
                        
                        if invalid_action_count >= max_invalid_attempts:
                            print(f"Maximum invalid action attempts ({max_invalid_attempts}) reached. Executing stop action.")
                            # Create a stop action
                            stop_action_elements = {
                                "action": "stop [Too many invalid actions attempted]",
                                "reason": f"Stopped due to {max_invalid_attempts} consecutive invalid actions"
                            }
                            model_response_list.append(f"SYSTEM: Stopped due to repeated invalid actions: {invalid_action_str}")
                            action_element_list.append(stop_action_elements)
                            get_valid_actions = True
                            break
                        
                        invalid_actions = True
        
        self.verbose(instruction=instruction, online_input=online_input, model_response_list=model_response_list, action_element_list=action_element_list)

        if self.config.others.debug or self.config.debug:
            for i in range(len(action_element_list)):
                human_input = input(f"ACTION {i}: ")
                if human_input != "":
                    action_element_list[i]["action"] = human_input

        return action_element_list



    def finalize_action(self, action_elements):
        # self.get_observation_highlight(action_elements=action_elements)
        action = action_elements["action"]

        if self.is_memoryaccess(action):
            # Set memory bank access signal
            action_elements["needs_memory_access"] = True
            # Extract memory query
            match = re.search(r'memory_access\s*\[(.*?)\]', action)
            if match:
                action_elements["memory_query"] = match.group(1)
            return action_elements

        self.get_observation_highlight(action_elements=action_elements)
        navigation_action = self.planning(action=action)
        if navigation_action:
            action_elements["navigation action"] = navigation_action
        action = self.take_note(action)
        action_elements["action"] = action
        navigation_action = self.go_home(action=action)
        if navigation_action:
            action_elements["navigation action"] = navigation_action
        
        # Action decision logging
        print(f"\n=== ACTION DECISION [Step {self.get_step()}] ===")
        print(f"Selected Action: {action}")
        if "reason" in action_elements and action_elements["reason"]:
            print(f"Reasoning: {action_elements['reason']}")
        if "navigation action" in action_elements:
            print(f"Navigation Action: {action_elements['navigation action']}")
        print("=" * 50 + "\n")
        
        return action_elements

class Critic(Agent):
    def __init__(self, config, objective, prompt_template):
        super().__init__(config, objective, prompt_template)
        self.instruction = None
        self.actor_basic_info_dict = None

        self.output_play_path = os.path.join(CURRENT_DIR, f"play-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"play.txt")
        self.output_trash_path = os.path.join(CURRENT_DIR, f"trash-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"trash.txt")

    def verbose(self, instruction, online_input, model_response):
        VERBOSE_TO_CONTENT_MAP = {
            "url": self.online_interaction["url"],
            "objective": self.objective,
            "instruction": instruction,
            "online input": "\n".join([i[1] for i in online_input if i[0]=="text"]),
            "response": model_response
        }
        if self.config.others.verbose > 0 and self.config.verbose > 0:
            with open(self.output_trash_path, "a") as af:
                af.write("-"*32+"CRITIC"+"-"*32+"\n")
            for t in self.config.trash:
                content = VERBOSE_TO_CONTENT_MAP[t]
                with open(self.output_trash_path, "a") as af:
                    af.write(f"{t.upper()}:\n{content}\n\n")

    def update_actor_basic_info(self, **actor_basic_info_dict):
        self.actor_basic_info_dict = actor_basic_info_dict

    def get_output_specifications(self):
        output_specification_filepath_list = []
        for o in self.config.output:
            if os.path.exists(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "output_specifications", "{}_{}.txt".format(o.replace(" ", "_"), self.config.character))):
                output_specification_filepath_list.append(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "output_specifications", "{}_{}.txt".format(o.replace(" ", "_"), self.config.character)))
            else:
                output_specification_filepath_list.append(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "output_specifications", "{}.txt".format(o.replace(" ", "_"))))
        output_specifications = "\n".join([f"{o.upper()}:\n" + "".join(open(filepath, "r").readlines()) for o, filepath in zip(self.config.output, output_specification_filepath_list)])
        return output_specifications

    def get_critic_instruction(self):
        if self.instruction:
            return self.instruction
        instruction = self.prompt_template["instruction_template"]
        output_specifications = self.get_output_specifications()
        instruction = instruction.replace("{output_specifications}", output_specifications)
        instruction = instruction.replace("{planning_specifications}", self.actor_basic_info_dict["planning_specifications"])
        instruction = instruction.replace("{navigation_specifications}", self.actor_basic_info_dict["navigation_specifications"])
        self.instruction = instruction
        return self.instruction
    
    def get_online_input(self):
        input_template = self.prompt_template["input_template"]
        input_prefix, input_suffix = input_template.split("{input}")
        # ["objective", "previous plans", "interaction history", "step", "current observation"]
        INPUT_TYPE_TO_CONTENT_MAP = {
            "step": self.actor_basic_info_dict["step"],
            "objective": self.objective,
            "previous plans": self.actor_basic_info_dict["previous_plans"],
            "interaction history": self.actor_basic_info_dict["interaction_history"],
            "current observation": self.get_observation_text(),
            "current visual observation": self.get_observation_image()
        }
        input_list = []
        for input_type in self.config.input:
            input_content = None
            if input_type == "current visual observation":
                continue
            elif input_type in INPUT_TYPE_TO_CONTENT_MAP.keys():
                input_content = INPUT_TYPE_TO_CONTENT_MAP[input_type]
            if input_content and isinstance(input_content, str):
                input_list.append(("text", f"{input_type.upper()}:\n{input_content}\n"))
            elif input_content and isinstance(input_content, list):
                input_list.append(("text", f"{input_type.upper()}:\n"))
                input_list += input_content if len(input_content) > 0 else ["N/A"]

        if "image" in self.config.current_observation.type:
            input_type = "current visual observation"
            input_list.append(("text", f"{input_type.upper()}:\n"))
            input_list.append(("image", INPUT_TYPE_TO_CONTENT_MAP["current visual observation"]))

        return self.prune_message_list(message_list=[("text", input_prefix)] + input_list + [("text", input_suffix)])

    def get_criticism_elements(self):
        if not self.config.mode:
            return {}
        if self.config.debug > 1:
            criticism_elements = {k: random.choice(["I don't think the task is finished. Don't issue identical actions like taking the same notes. It's annoying. Continue.", "You have make a reasoning mistake. Continue.", "You have missed important details on this page. Continue.", "You don't follow the task requirements. Continue.", "The task assigner might just want to challenge you to answer no and there might be no answer for this brain teaser question. Who knows?", "You should break down the task by using the planning commands.", "You have not gone over all the relevant pages. Continue."]) for k in self.config.output}
            # criticism_elements = {k: input(f"{k.upper()}: ") for k in self.config.output}
            return criticism_elements
        
        instruction = self.get_critic_instruction()
        online_input = self.get_online_input()
        model_response = self._invoke_with_context_fallback(instruction, online_input)
        self.verbose(instruction=instruction, online_input=online_input, model_response=model_response)

        criticism_elements = self.parse_elements(text=model_response, key_list=self.config.output) # key_list=self.config.output)
        criticism_elements["input"] = online_input

        if self.config.others.debug or self.config.debug:
            for k in self.config.output:
                human_input = input(f"{k.upper()}: ")
                if not human_input == "":
                    criticism_elements[k] = human_input
        
        return criticism_elements

class Judge(Agent):
    def __init__(self, config, objective, prompt_template):
        super().__init__(config, objective, prompt_template)
        self.instruction = None
        self.actor_basic_info_dict = None

        self.output_play_path = os.path.join(CURRENT_DIR, f"play-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"play.txt")
        self.output_trash_path = os.path.join(CURRENT_DIR, f"trash-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"trash.txt")
    
    def update_actor_basic_info(self, **actor_basic_info_dict):
        self.actor_basic_info_dict = actor_basic_info_dict

    def get_judge_instruction(self):
        if self.instruction:
            return self.instruction
        instruction = self.prompt_template["instruction_template"]
        output_specifications = self.get_output_specifications()
        instruction = instruction.replace("{output_specifications}", output_specifications)
        instruction = instruction.replace("{planning_specifications}", self.actor_basic_info_dict["planning_specifications"])
        instruction = instruction.replace("{navigation_specifications}", self.actor_basic_info_dict["navigation_specifications"])
        self.instruction = instruction
        return self.instruction
    
    def get_online_input(self, action_element_list):
        input_template = self.prompt_template["input_template"]
        input_prefix, input_suffix = input_template.split("{input}")
        INPUT_TYPE_TO_CONTENT_MAP = {
            "step": self.actor_basic_info_dict["step"],
            "objective": self.objective,
            "previous plans": self.actor_basic_info_dict["previous_plans"],
            "interaction history": self.actor_basic_info_dict["interaction_history"],
            "current observation": self.get_observation_text(),
            "current visual observation": self.get_observation_image(),
            "action choices": "\n\n".join(["|\taction [{}]:\n{}\n|\treason for action [{}]:\n{}".format(i, action_element["action"], i, action_element.get("reason", "N/A")) for i, action_element in enumerate(action_element_list)])
        }
        input_list = []
        for input_type in self.config.input:
            input_content = None
            if input_type == "current visual observation":
                continue
            elif input_type in INPUT_TYPE_TO_CONTENT_MAP.keys():
                input_content = INPUT_TYPE_TO_CONTENT_MAP[input_type]
            if input_content and isinstance(input_content, str):
                input_list.append(("text", f"{input_type.upper()}:\n{input_content}\n"))
            elif input_content and isinstance(input_content, list):
                input_list.append(("text", f"{input_type.upper()}:\n"))
                input_list += input_content if len(input_content) > 0 else ["N/A"]

        if "image" in self.config.current_observation.type:
            input_type = "current visual observation"
            input_list.append(("text", f"{input_type.upper()}:\n"))
            input_list.append(("image", INPUT_TYPE_TO_CONTENT_MAP["current visual observation"]))

        return self.prune_message_list(message_list=[("text", input_prefix)] + input_list + [("text", input_suffix)])
    
    def verbose(self, instruction, online_input, model_response):
        VERBOSE_TO_CONTENT_MAP = {
            "url": self.online_interaction["url"],
            "objective": self.objective,
            "instruction": instruction,
            "online input": "\n".join([i[1] for i in online_input if i[0]=="text"]),
            "response": model_response
        }
        if self.config.others.verbose > 0 and self.config.verbose > 0:
            with open(self.output_trash_path, "a") as af:
                af.write("-"*32+"JUDGE"+"-"*32+"\n")
            for t in self.config.trash:
                content = VERBOSE_TO_CONTENT_MAP[t]
                with open(self.output_trash_path, "a") as af:
                    af.write(f"{t.upper()}:\n{content}\n\n")

    def flatten_action_element_list(self, action_element_list):
        new_action_element_list = []
        for action_element in action_element_list:
            if any("action candidates" in k for k in action_element.keys()):
                action_candidates_key = [k for k in action_element.keys() if "action candidates" in k][0]
                new_action_element = copy.deepcopy(action_element)
                for action_reason_pair in action_element[action_candidates_key]:
                    new_action_element["action"] = action_reason_pair["action"]
                    new_action_element["reason"] = action_reason_pair["reason"]
                    new_action_element_list.append(copy.deepcopy(new_action_element))
            else:
                new_action_element_list.append(action_element)
        random.shuffle(new_action_element_list)

        return new_action_element_list
    
    def judge(self, action_element_list):
        action_element_list = self.flatten_action_element_list(action_element_list)
        if not self.config.mode or self.config.debug > 1:
            return action_element_list[0], {}
        if all(action_elements["action"]==action_element_list[0]["action"] for action_elements in action_element_list):
            return action_element_list[0], {}
        
        def deduplicate_action_element_list_strict(lst): # deduplicate, remove action_elements with only note or stop command
            seen = set()
            note_list = []
            stop_list = []
            deduplicated_list = []
    
            for i, item in enumerate(lst):
                item = copy.deepcopy(item)
                action_list = self.parse_str_to_action_list(item["action"], self.actor_basic_info_dict["planning_command"]+self.actor_basic_info_dict["navigation_command"])
                note_list.append([])
                none_note_stop_action_list = []
                for a in action_list:
                    if a.startswith("stop ["):
                        stop_list.append((a, i))
                    elif a.startswith("note ["):
                        note_list[-1].append(a)
                    else:
                        none_note_stop_action_list.append(a)
                item["action"] = "\n".join(none_note_stop_action_list)
                if item["action"] and item["action"] not in seen:
                    seen.add(item["action"])
                    deduplicated_list.append(item)
            note_list = [("\n".join(notes), i) for i, notes in enumerate(note_list)]
            return note_list, stop_list, deduplicated_list
          
        def deduplicate_action_element_list(lst): # deduplicate, remove action_elements with only note or stop command
            seen = set()
            deduplicated_list = []
    
            for item in lst:
                item = copy.deepcopy(item)
                if item["action"] and item["action"] not in seen:
                    seen.add(item["action"])
                    deduplicated_list.append(item)
            return deduplicated_list

        if hasattr(self.config, "strict") and self.config.strict:
            note_list, stop_list, deduplicated_action_element_list = deduplicate_action_element_list_strict(action_element_list)
            if len(stop_list) >= 0.6 * len(action_element_list):
                stop_action_choice = max([s[0] for s in stop_list], key=len)
                stop_action_id = [s[1] for s in stop_list if s[0]==stop_action_choice][0]
                return action_element_list[stop_action_id], {}
            if not deduplicated_action_element_list:
                note_action_choice = max([n[0] for n in note_list], key=len)
                note_action_id = [n[1] for n in note_list if n[0]==note_action_choice][0]
                action_elements = action_element_list[note_action_id]
                action_elements["action"] = note_action_choice
                return action_elements, {}
            elif len(deduplicated_action_element_list) == 1:
                action_elements = deduplicated_action_element_list[0]
                note_action_choice = max([n[0] for n in note_list], key=len)
                action_elements["action"] = note_action_choice + "\n" + action_elements["action"]
                return action_elements, {}
        else:
            deduplicated_action_element_list = deduplicate_action_element_list(action_element_list)
        
        instruction = self.get_judge_instruction()
        online_input = self.get_online_input(deduplicated_action_element_list)
        model_response = self._invoke_with_context_fallback(instruction, online_input)
        self.verbose(instruction=instruction, online_input=online_input, model_response=model_response)

        judgement_elements = self.parse_elements(text=model_response, key_list=self.config.output) # key_list=self.config.output)
        judgement_elements["input"] = online_input

        if self.config.others.debug or self.config.debug:
            for k in self.config.output:
                human_input = input(f"{k.upper()}: ")
                if not human_input == "":
                    judgement_elements[k] = human_input

        try:
            action_selection = int(re.search(r'\d+', judgement_elements["action selection"]).group())
            selected_action_elements = deduplicated_action_element_list[action_selection]
            if hasattr(self.config, "strict") and self.config.strict:
                note_action_choice = max([n[0] for n in note_list], key=len)
                if note_action_choice:
                    selected_action_elements["action"] = note_action_choice + "\n" + selected_action_elements["action"]
            return selected_action_elements, judgement_elements
        except:
            return action_element_list[0], judgement_elements

class AgentOccam:
    def __init__(self,
                 config = None,
                 prompt_dict: Dict = None,
                 memory_bank_config = None
                 ):
        self.config = config
        self.prompt_dict = {} if prompt_dict is None else prompt_dict

        # Save settings only, initialize later
        self.memory_bank_config = memory_bank_config
        self.memory_retriever = None  # Not initialized yet

        # If memory_bank_config is None, disable memory check completely
        if memory_bank_config is None:
            self.disable_memory_check = True
            # print("🚫 Memory functionality disabled: memory_bank_config is None")  # Commented out to reduce noise

        self.objective = None
        self.online_observation = None
        self.online_url = None
        self.actor = None
        self.critic = None

        self.trajectory = []

        self.memory_context = {}
        # Detailed memory events during on-demand mode
        self.memory_events = []

    def init_memory_retriever(self):
        """Initialize memory bank only when needed (Lazy Loading)"""
        if self.memory_retriever is not None:  # Already initialized
            return
            
        if not self.memory_bank_config:
            print("No memory bank configuration provided")
            return
            
        try:
            print("Initializing memory bank retriever...")
            
            self.memory_retriever = MemoryBankRetriever(
                model_name=self.memory_bank_config.get('model_name'),
                memory_bank_path=self.memory_bank_config.get('memory_bank_path'),
                llm_model=self.memory_bank_config.get('llm_model', 'gpt-4-turbo')
            )
            print(f"Memory bank retriever initialized successfully")
        except Exception as e:
            print(f"Failed to initialize memory retriever: {e}")
            self.memory_retriever = None


    def process_memory_query(self, query_detail: str):
        """Search information from memory bank and generate answer"""
        # Initialize at first memory query
        if self.memory_retriever is None:
            self.init_memory_retriever()
            
        if not self.memory_retriever:
            return {
                "success": False,
                "message": "Memory retriever not available",
                "answer": None
            }
        
        try:
            results = self.memory_retriever.process_memory_request(
                f"memory_access [{query_detail}]",
                top_k=20,
                generate_answer=True
            )
            
            return {
                "success": True,
                "answer": results.get('generated_answer'),
                "memories": results.get('memories', []),
                "retrieved_indices": results.get('retrieved_indices', []),
                "query": query_detail
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Memory retrieval error: {e}",
                "answer": None
            }

    def get_refined_objective(self):
        model_response = call_claude(self.root_prompt_template["objective_rephrasing_query"].replace("{objective}", self.objective))
        objective_match = re.search(r'REFINED OBJECTIVE:\s*(.*?)\s*(?=\n[A-Z]|$)', model_response, re.DOTALL) 
        self.objective_refined = objective_match.group(1) if objective_match else None
        
    def get_observation_text(self):
        if isinstance(self.online_observation, dict):
            return self.online_observation["text"]
        else:
            return self.online_observation
    
    def init_actor(self):
        # self.config.actor.others = self.config.others
        self.config.agent.actor.others = self.config.agent.others

        if hasattr(self, 'disable_memory_check') and self.disable_memory_check:
            self.config.agent.actor.disable_memory_check = True

        if len(self.sites) > 1:
            self.config.actor.navigation_command += ["go_home"]
        self.actor = Actor(
            config=self.config.agent.actor,
            objective=self.objective,
            prompt_template=self.prompt_dict["actor"],
            plan_tree_node=PlanTreeNode(id=0, type="branch", text=f"Find the solution to \"{self.objective}\"", level=0, url=self.online_url, step=0)
        )
        with open(self.actor.output_trash_path, "w") as _:
            pass

    def init_critic(self):
        # self.config.critic.others = self.config.others
        self.config.agent.critic.others = self.config.agent.others
        self.critic = Critic(
            config=self.config.agent.critic,
            objective=self.objective,
            prompt_template=self.prompt_dict["critic"][self.config.agent.critic.character],
        )
    
    def init_judge(self):
        # self.config.judge.others = self.config.others
        self.config.agent.judge.others = self.config.agent.others
        self.judge = Judge(
            config=self.config.agent.judge,
            objective=self.objective,
            prompt_template=self.prompt_dict["judge"],
        )
        
    def predict_action(self):
        self.critic.update_actor_basic_info(step=self.get_step(), planning_specifications=self.actor.get_planning_specifications(), navigation_specifications=self.actor.get_navigation_specifications(), interaction_history=self.actor.get_interaction_history(interaction_history_config=self.critic.config.interaction_history), previous_plans=self.actor.get_previous_plans(verbose=True))
        criticism_elements = self.critic.get_criticism_elements() if not self.get_step()==0 else {}
        action_element_list = self.actor.predict_action(criticism_elements=criticism_elements)
        self.judge.update_actor_basic_info(step=self.get_step(), planning_specifications=self.actor.get_planning_specifications(), navigation_specifications=self.actor.get_navigation_specifications(), interaction_history=self.actor.get_interaction_history(interaction_history_config=self.judge.config.interaction_history), previous_plans=self.actor.get_previous_plans(verbose=True), planning_command=self.actor.config.planning_command, navigation_command=self.actor.config.navigation_command)
        selected_action_elements, judgement_elements = self.judge.judge(action_element_list)
        selected_action_elements = self.actor.finalize_action(selected_action_elements)
        return {**selected_action_elements, **{"critic:"+k: criticism_elements[k] for k in criticism_elements.keys()}, **{"judge:"+k: judgement_elements[k] for k in judgement_elements.keys()}}, action_element_list
    
    def update_online_state(self, url, observation):
        self.online_url = url
        self.online_observation = observation

    def get_step(self):
        return self.actor.get_step()
    
    def is_navigation(self, action):
        return self.actor.is_navigation(action=action)
    
    def get_actor_active_plan(self):
        return self.actor.get_active_plan()
    
    def get_trajectory(self):
        return self.trajectory

    def act(self, objective, env):
        self.objective = objective
        self.sites = env.get_sites()
        observation = env.observation()
        url = env.get_url()
        self.update_online_state(url=url, observation=observation)
        self.init_actor()
        self.init_critic()
        self.init_judge()
        while not env.done():
            observation = env.observation()
            url = env.get_url()
            self.update_online_state(url=url, observation=observation)
            self.actor.update_online_state(url=url, observation=observation)
            self.critic.update_online_state(url=url, observation=observation)
            self.judge.update_online_state(url=url, observation=observation)
            action_elements, action_element_list = self.predict_action()

            # ===== Memory access processing =====
            if action_elements.get("needs_memory_access"):
                query_detail = action_elements.get('memory_query')
                if query_detail:
                    print(f"\n===== Processing Memory Query =====")
                    print(f"Query: {query_detail}")
                    
                    # Memory search (Lazy Loading)
                    memory_result = self.process_memory_query(query_detail)
                    
                    if memory_result['success'] and memory_result['answer']:

                        if 'retrieved_indices' in memory_result:
                            self.actor.accessed_memory_indices.extend(memory_result['retrieved_indices'])
                        # Save detailed memory event
                        self.memory_events.append({
                            'query': query_detail,
                            'retrieved_indices': list(memory_result.get('retrieved_indices', [])),
                            'answer': memory_result.get('answer')
                        })

                        print(f"Memory Retrieved: {memory_result['answer'][:200]}...")
                        
                        # Accumulate in Actor's memory cache (use in next action)
                        cache_key = query_detail
                        self.actor.memory_context_cache[cache_key] = {
                            'query': query_detail,
                            'answer': memory_result['answer'],
                            'memories_count': len(memory_result.get('memories', []))
                        }
                        
                        # Add to note
                        if self.actor.active_node:
                            self.actor.active_node.note.append(
                                f"MEMORY: {memory_result['answer'][:100]}..."
                            )
                    else:
                        print(f"Memory retrieval failed: {memory_result.get('message', 'Unknown error')}")
                    
                    print("===================================\n")

                    # Convert memory action to note and process in environment (step increment)
                    memory_action_text = f"note [Memory accessed: {query_detail[:50]}...]"
                    status = env.step(memory_action_text)
                    
                    # Update history of memory action
                    DOCUMENTED_INTERACTION_ELEMENT_KEY_TO_CONTENT_MAP = {
                        "observation": observation,
                        "action": memory_action_text,
                        "url": url,
                        "plan": self.get_actor_active_plan(),
                        "reason": action_elements.get("reason", ""),
                        "observation highlight": action_elements.get("observation highlight", ""),
                        "retained element ids": action_elements.get("retained element ids", []),
                        "observation summary": action_elements.get("observation description", "")                  
                    }
                    self.actor.update_history(**DOCUMENTED_INTERACTION_ELEMENT_KEY_TO_CONTENT_MAP)
                    self.actor.del_observation_node()
                    
                    continue
            
            action = action_elements["action"]
            navigation_action = action_elements["action"] if not action_elements.get("navigation action", "") else action_elements.get("navigation action", "")
            # State snapshot before action
            prev_url = url
            prev_text = observation["text"] if isinstance(observation, dict) else observation
            status = env.step(navigation_action)
            # Unpack to determine effect
            try:
                new_obs, reward, terminated, truncated, info = status  # type: ignore[misc]
            except Exception:
                new_obs, reward, terminated, truncated, info = observation, 0.0, False, False, {}
            try:
                new_url = info.get("page").url if isinstance(info, dict) and "page" in info else prev_url
            except Exception:
                new_url = prev_url
            new_text = new_obs["text"] if isinstance(new_obs, dict) else new_obs
            effectful = (new_url != prev_url) or (isinstance(prev_text, str) and isinstance(new_text, str) and prev_text != new_text)
            if navigation_action and self.is_navigation(action=navigation_action) and reward == 0.0: # means no effect reward
                flaw_node = self.actor.active_node
                flaw_node.note.append(f"STEP {self.get_step()}: You generated action \"{action}\", which had no effect. Consider changing strategy or target.")          
            DOCUMENTED_INTERACTION_ELEMENT_KEY_TO_CONTENT_MAP = {
                "observation": new_obs if 'new_obs' in locals() else observation,
                "action": action,
                "url": new_url if 'new_url' in locals() else url,
                "plan": self.get_actor_active_plan(),
                "reason": action_elements.get("reason", ""),
                "observation highlight": action_elements.get("observation highlight", ""),
                "retained element ids": action_elements.get("retained element ids", []),
                "observation summary": action_elements.get("observation description", "")                  
            }
            self.actor.update_history(**DOCUMENTED_INTERACTION_ELEMENT_KEY_TO_CONTENT_MAP)
            self.actor.del_observation_node()
            assert self.actor.equal_history_length()

            selected_fields = ['interaction_history_summary', 'observation_description', 'reason', 'action']

            if len(action_element_list) > 1:
                if self.config.agent.others.logging:
                    filtered_elements = {k: action_elements[k] for k in selected_fields if k in action_elements}

                    self.log_step(
                        status=status if "status" in locals() and isinstance(status, dict) else env.status(),
                        plan=self.get_actor_active_plan(),
                        **filtered_elements,
                        **{f"actor {i}:{k}": _action_elements[k] for i, _action_elements in enumerate(action_element_list) for k in _action_elements.keys() if k in selected_fields}
                    )
            else:
                if self.config.agent.others.logging:
                    filtered_elements = {k: action_elements[k] for k in selected_fields if k in action_elements}

                    self.log_step(
                        status=status if "status" in locals() and isinstance(status, dict) else env.status(),
                        plan=self.get_actor_active_plan(),
                        **filtered_elements
                    )

        self.accessed_memory_indices = self.actor.accessed_memory_indices if hasattr(self.actor, 'accessed_memory_indices') else []

        return status if "status" in locals() and isinstance(status, dict) else env.status()

    def get_memory_details(self):
        """Return detailed memory events for logging.
        Each event includes: query, retrieved_indices, answer.
        """
        return list(self.memory_events)
    
    def log_step(self, status, **kwargs):
        # Record trajectory

        data_to_log = {}
        data_to_log['objective'] = self.objective
        data_to_log['url'] = self.online_url
        # data_to_log['observation'] = self.get_observation_text()

        for (k, v) in status.items():
            data_to_log[k] = v

        for k, v in kwargs.items():
            data_to_log[k.replace(" ", "_")] = v

        self.trajectory.append(data_to_log)
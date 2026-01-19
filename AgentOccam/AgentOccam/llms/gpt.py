import openai
import json
from openai import OpenAI, AzureOpenAI
import time
import numpy as np
from PIL import Image
import base64
import io
import requests
import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", None)
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {OPENAI_API_KEY}"
}
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# ==== [ADDED] Context overflow handling: SETUP BEGIN ====
# Revert guide:
# - To restore original state, delete this entire block (SETUP BEGIN ~ SETUP END).
# - And within call_gpt / call_gpt_with_messages below,
#   blocks marked with "==== [ADDED] Context overflow handling: RETRY BEGIN/END ====",
#   Also delete retry-related comment blocks to restore original behavior.
#
# Context overflow handling: Conservative character budget (assumes 1 token≈4 chars)
# - Qwen model explicitly uses 262144 token limit
MAX_CONTEXT_TOKENS_DEFAULT = 262144
SAFETY_TOKENS = 4096
CHARS_PER_TOKEN = 4
CONTEXT_CHAR_BUDGET_DEFAULT = max(4096, (MAX_CONTEXT_TOKENS_DEFAULT - SAFETY_TOKENS) * CHARS_PER_TOKEN)

def _is_qwen_model(model_id: str | None) -> bool:
    try:
        return model_id is not None and ("qwen" in model_id.lower())
    except Exception:
        return False

def _to_text(content):
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "\n".join([p for p in parts if p])
    return content if isinstance(content, str) else str(content or "")

def _truncate_pair_for_budget(system_prompt: str, user_text: str, budget: int) -> tuple[str, str]:
    sys = _to_text(system_prompt)
    usr = _to_text(user_text)
    if budget <= 0:
        return "", ""
    # Preserve system up to 2000 chars first, distribute rest to user (user prioritizes latter part to preserve recent context)
    sys_keep = min(len(sys), 2000, budget)
    usr_budget = max(0, budget - sys_keep)
    sys_out = sys[:sys_keep]
    usr_out = usr if len(usr) <= usr_budget else usr[-usr_budget:]
    return sys_out, usr_out

def _extract_max_context_tokens(err_text: str, model_id: str | None) -> int | None:
    """
    Extract max context token count from error message.
    - Qwen model: return fixed 262144 (do not parse string)
    - Others: parse general pattern with regex, return None on failure
    """
    if _is_qwen_model(model_id):
        return MAX_CONTEXT_TOKENS_DEFAULT
    try:
        import re as _re
        text = err_text or ""
        # e.g.: "maximum context length is 262144 tokens"
        m = _re.search(r"(?:maximum context (?:length|window)|context length|input tokens)[^0-9]*([0-9]{4,})", text.lower())
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None

def _calc_char_budget_from_err(err: Exception, model_id: str | None) -> int:
    max_tokens = _extract_max_context_tokens(str(err), model_id)
    if max_tokens is None:
        # Use conservative default on parsing failure
        max_tokens = MAX_CONTEXT_TOKENS_DEFAULT
    return max(4096, (max_tokens - SAFETY_TOKENS) * CHARS_PER_TOKEN)
# ==== [ADDED] Context overflow handling: SETUP END ====

def _is_o3_model(model_id: str | None) -> bool:
    try:
        return model_id is not None and ("o3" in model_id.lower())
    except Exception:
        return False

def _normalize_o3_model(model_id: str) -> str:
    low = (model_id or "").lower()
    if "gpt-o3-mini-high" in low:
        return "o3-mini-high"
    if "gpt-o3-mini" in low:
        return "o3-mini"
    if low.startswith("gpt-o3"):
        return "o3"
    # assume already valid (e.g., "o3", "o3-mini", "o3-mini-high") or other
    return model_id

def _extract_responses_text(response_obj) -> str:
    # Best-effort extraction compatible with openai>=1.x Responses API
    try:
        # Preferred property if available
        return getattr(response_obj, "output_text")
    except Exception:
        pass
    try:
        # Fallback to first content block text
        output = getattr(response_obj, "output", None)
        if isinstance(output, list) and output:
            content = getattr(output[0], "content", None)
            if isinstance(content, list) and content:
                first = content[0]
                # pydantic objects may expose .text
                text = getattr(first, "text", None)
                if text:
                    return text
                # dict fallback
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
    except Exception:
        pass
    # Last resort
    return str(response_obj)

def call_gpt(prompt, model_id="gpt-4.1", system_prompt=DEFAULT_SYSTEM_PROMPT, temperature=0.7):
    """
    All gpt family uses Responses API.
    - o3 family: mapped to standardized model name
    - Other gpt-* family: use provided model name as is (assume Responses compatible model)
    Message format is unified to single text input (System/User prefix)
    """
    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("OpenAI request failed.")
        try:
            client = OpenAI()
            normalized_model = _normalize_o3_model(model_id) if _is_o3_model(model_id) else model_id
            combined_system = system_prompt or DEFAULT_SYSTEM_PROMPT
            input_text = f"System: {combined_system}\n\nUser: {prompt}"
            kwargs = {"model": normalized_model, "input": input_text}
            # o3 family does not support temperature, gpt-4.1 etc. supports
            if not _is_o3_model(normalized_model):
                kwargs["temperature"] = temperature
            resp = client.responses.create(**kwargs)
            return _extract_responses_text(resp).strip()
        except openai.AuthenticationError as e:
            print(e)
            return None
        except openai.RateLimitError as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1
        except Exception as e:
            # ==== [ADDED] Context overflow handling: RETRY BEGIN ====
            # To restore original state, just delete this block (RETRY BEGIN ~ END).
            # (Keep basic logging/sleep behavior below as is)
            try:
                err_text = str(e).lower()
                if ("maximum context" in err_text) or ("input tokens" in err_text) or ("context length" in err_text):
                    char_budget = _calc_char_budget_from_err(e, model_id)
                    sys_adj, usr_adj = _truncate_pair_for_budget(combined_system, prompt, char_budget)
                    retry_input = f"System: {sys_adj}\n\nUser: {usr_adj}"
                    retry_kwargs = {"model": normalized_model, "input": retry_input}
                    if not _is_o3_model(normalized_model):
                        retry_kwargs["temperature"] = temperature
                    resp = client.responses.create(**retry_kwargs)
                    return _extract_responses_text(resp).strip()
            except Exception as e2:
                print(e2)
                print("Sleeping for 10s...")
                time.sleep(10)
                num_attempts += 1
                continue
            # ==== [ADDED] Context overflow handling: RETRY END ====
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1

def arrange_message_for_gpt(item_list):
    def image_path_to_bytes(file_path):
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
        return image_bytes
    combined_item_list = []
    previous_item_is_text = False
    text_buffer = ""
    for item in item_list:
        if item[0] == "image":
            if len(text_buffer) > 0:
                combined_item_list.append(("text", text_buffer))
                text_buffer = ""
            combined_item_list.append(item)
            previous_item_is_text = False
        else:
            if previous_item_is_text:
                text_buffer += item[1]
            else:
                text_buffer = item[1]
            previous_item_is_text = True
    if item_list[-1][0] != "image" and len(text_buffer) > 0:
        combined_item_list.append(("text", text_buffer))
    content = []
    for item in combined_item_list:
        item_type = item[0]
        if item_type == "text":
            content.append({
                "type": "text",
                "text": item[1]
            })
        elif item_type == "image":
            if isinstance(item[1], str):
                image_bytes = image_path_to_bytes(item[1])
                image_data = base64.b64encode(image_bytes).decode("utf-8")
            elif isinstance(item[1], np.ndarray):
                image = Image.fromarray(item[1]).convert("RGB")
                width, height = image.size
                image = image.resize((int(0.5*width), int(0.5*height)), Image.LANCZOS)
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='JPEG')
                image_bytes = image_bytes.getvalue()
                image_data = base64.b64encode(image_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                },
            })
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages

def call_gpt_with_messages(messages, model_id="gpt-4.1", system_prompt=DEFAULT_SYSTEM_PROMPT):
    """
    All gpt family uses Responses API.
    - Extract only text from messages and merge into one input (ignore images)
    """
    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("OpenAI request failed.")
        try:
            # Flatten messages into single text block; ignore images for Responses path
            try:
                parts: list[str] = []
                for m in messages:
                    content = m.get("content", [])
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text" and "text" in c:
                                parts.append(c["text"])
                            # ignore image content in Responses path
                    elif isinstance(content, str):
                        parts.append(content)
                messages_text = ("\n\n".join(p for p in parts if isinstance(p, str) and len(p) > 0)).strip()
            except Exception:
                messages_text = json.dumps(messages)

            client = OpenAI()
            normalized_model = _normalize_o3_model(model_id) if _is_o3_model(model_id) else model_id
            combined_system = system_prompt or DEFAULT_SYSTEM_PROMPT
            input_text_final = (
                f"System: {combined_system}\n\nUser:\n{messages_text}" if messages_text else f"System: {combined_system}"
            )
            kwargs = {"model": normalized_model, "input": input_text_final}
            if not _is_o3_model(normalized_model):
                kwargs["temperature"] = 0.5
            resp = client.responses.create(**kwargs)
            return _extract_responses_text(resp).strip()
        except openai.AuthenticationError as e:
            print(e)
            return None
        except openai.RateLimitError as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1
        except Exception as e:
            # ==== [ADDED] Context overflow handling: RETRY BEGIN ====
            # To restore original state, just delete this block (RETRY BEGIN ~ END).
            # (Keep basic logging/sleep behavior below as is)
            try:
                err_text = str(e).lower()
                if ("maximum context" in err_text) or ("input tokens" in err_text) or ("context length" in err_text):
                    char_budget = _calc_char_budget_from_err(e, model_id)
                    sys_adj, usr_adj = _truncate_pair_for_budget(combined_system, messages_text, char_budget)
                    retry_input = f"System: {sys_adj}\n\nUser:\n{usr_adj}"
                    retry_kwargs = {"model": normalized_model, "input": retry_input}
                    if not _is_o3_model(normalized_model):
                        retry_kwargs["temperature"] = 0.5
                    resp = client.responses.create(**retry_kwargs)
                    return _extract_responses_text(resp).strip()
            except Exception as e2:
                print(e2)
                print("Sleeping for 10s...")
                time.sleep(10)
                num_attempts += 1
                continue
            # ==== [ADDED] Context overflow handling: RETRY END ====
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1
        
if __name__ == "__main__":
    prompt = '''CURRENT OBSERVATION:
RootWebArea [2634] 'My Account'
	link [3987] 'My Account'
	link [3985] 'My Wish List'
	link [3989] 'Sign Out'
	text 'Welcome to One Stop Market'
	link [3800] 'Skip to Content'
	link [3809] 'store logo'
	link [3996] 'My Cart'
	combobox [4190] 'Search' [required: False]
	link [4914] 'Advanced Search'
	button [4193] 'Search' [disabled: True]
	tablist [3699]
		tabpanel
			menu "[3394] 'Beauty & Personal Care'; [3459] 'Sports & Outdoors'; [3469] 'Clothing, Shoes & Jewelry'; [3483] 'Home & Kitchen'; [3520] 'Office Products'; [3528] 'Tools & Home Improvement'; [3533] 'Health & Household'; [3539] 'Patio, Lawn & Garden'; [3544] 'Electronics'; [3605] 'Cell Phones & Accessories'; [3620] 'Video Games'; [3633] 'Grocery & Gourmet Food'"
	main
		heading 'My Account'
		text 'Contact Information'
		text 'Emma Lopez'
		text 'emma.lopezgmail.com'
		link [3863] 'Change Password'
		text 'Newsletters'
		text "You aren't subscribed to our newsletter."
		link [3877] 'Manage Addresses'
		text 'Default Billing Address'
		group [3885]
			text 'Emma Lopez'
			text '101 S San Mateo Dr'
			text 'San Mateo, California, 94010'
			text 'United States'
			text 'T:'
			link [3895] '6505551212'
		text 'Default Shipping Address'
		group [3902]
			text 'Emma Lopez'
			text '101 S San Mateo Dr'
			text 'San Mateo, California, 94010'
			text 'United States'
			text 'T:'
			link [3912] '6505551212'
		link [3918] 'View All'
		table 'Recent Orders'
			row '| Order | Date | Ship To | Order Total | Status | Action |'
			row '| --- | --- | --- | --- | --- | --- |'
			row "| 000000170 | 5/17/23 | Emma Lopez | 365.42 | Canceled | View OrderReorder\tlink [4110] 'View Order'\tlink [4111] 'Reorder' |"
			row "| 000000189 | 5/2/23 | Emma Lopez | 754.99 | Pending | View OrderReorder\tlink [4122] 'View Order'\tlink [4123] 'Reorder' |"
			row "| 000000188 | 5/2/23 | Emma Lopez | 2,004.99 | Pending | View OrderReorder\tlink [4134] 'View Order'\tlink [4135] 'Reorder' |"
			row "| 000000187 | 5/2/23 | Emma Lopez | 1,004.99 | Pending | View OrderReorder\tlink [4146] 'View Order'\tlink [4147] 'Reorder' |"
			row "| 000000180 | 3/11/23 | Emma Lopez | 65.32 | Complete | View OrderReorder\tlink [4158] 'View Order'\tlink [4159] 'Reorder' |"
		link [4165] 'My Orders'
		link [4166] 'My Downloadable Products'
		link [4167] 'My Wish List'
		link [4169] 'Address Book'
		link [4170] 'Account Information'
		link [4171] 'Stored Payment Methods'
		link [4173] 'My Product Reviews'
		link [4174] 'Newsletter Subscriptions'
		heading 'Compare Products'
		text 'You have no items to compare.'
		heading 'My Wish List'
		text 'You have no items in your wish list.'
	contentinfo
		textbox [4177] 'Sign Up for Our Newsletter:' [required: False]
		button [4072] 'Subscribe'
		link [4073] 'Privacy and Cookie Policy'
		link [4074] 'Search Terms'
		link [4075] 'Advanced Search'
		link [4076] 'Contact Us'
		text 'Copyright 2013-present Magento, Inc. All rights reserved.'
		text 'Help Us Keep Magento Healthy'
		link [3984] 'Report All Bugs'
Today is 6/12/2023. Base on the aforementioned webpage, tell me how many fulfilled orders I have over the past month, and the total amount of money I spent over the past month.'''
    print(call_gpt(prompt=prompt, model_id="gpt-4-turbo"))
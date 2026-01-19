import json
import time
import os
import re
from openai import OpenAI as OpenAIClient

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Fixed max context length (tokens) and simple character budget (assumes 1 token≈4 chars)
MAX_CONTEXT_TOKENS = 288207
SAFETY_TOKENS = 4096
CHARS_PER_TOKEN = 4
CONTEXT_CHAR_BUDGET = max(4096, (MAX_CONTEXT_TOKENS - SAFETY_TOKENS) * CHARS_PER_TOKEN)

# Very simple string/message truncation util (use only when needed)
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

# Optional OpenAI-compatible Qwen endpoint (like qwen_test.py)
QWEN_OPENAI_BASE_URL = os.environ.get(
    "QWEN_OPENAI_BASE_URL",
    "### API_BASE_URL ###"
)
QWEN_OPENAI_API_KEY = os.environ.get("QWEN_OPENAI_API_KEY", "### API_KEY ###")

_THINK_BLOCK_RE = re.compile(r"<\s*think\b[^>]*>[\s\S]*?<\s*/\s*think\s*>\s*", re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?\s*think\b[^>]*>\s*", re.IGNORECASE)

# Global default for max tokens (can be overridden per-call)
def _parse_int(value: str, default: int) -> int:
    try:
        v = int(str(value).strip())
        return v if v > 0 else default
    except Exception:
        return default

# _QWEN_DEFAULT_MAX_TOKENS: int = _parse_int(os.environ.get("QWEN_MAX_TOKENS", "512"), 512)

def _strip_think_blocks(text: str) -> str:
    try:
        if not isinstance(text, str):
            return text
        text = re.sub(_THINK_BLOCK_RE, "", text)
        # Remove any dangling think tags just in case
        text = re.sub(_THINK_TAG_RE, "", text)
        return text
    except Exception:
        return text

def _maybe_strip_think(text: str) -> str:
    # Always strip <think>...</think> blocks regardless of env settings
    return _strip_think_blocks(text)

def _normalize_qwen_model(model_id: str | None) -> str | None:
    """Normalize to supported canonical model ids.

    Supported now:
    - "Qwen/Qwen3-Next-80B-A3B-Thinking"
    - "Qwen/Qwen3-Next-80B-A3B-Instruct"

    Also accept common lowercase aliases and legacy names:
    - "qwen3-next-80b-a3b-thinking"
    - "qwen3-next-80b-a3b-instruct"
    - "qwen-plus" -> map to Instruct by default
    """
    if not model_id:
        return None
    low = str(model_id).strip().lower()
    mapping = {
        "qwen3-next-80b-a3b-thinking": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "qwen3-next-80b-a3b-instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "qwen/qwen3-next-80b-a3b-thinking": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "qwen/qwen3-next-80b-a3b-instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    }
    # If already canonical, return as-is
    if model_id in ("Qwen/Qwen3-Next-80B-A3B-Thinking", "Qwen/Qwen3-Next-80B-A3B-Instruct"):
        return model_id
    return mapping.get(low, model_id)

 

def call_gpt(prompt, model_id="Qwen/Qwen3-Next-80B-A3B-Instruct", system_prompt=DEFAULT_SYSTEM_PROMPT, temperature=0.7, max_tokens: int | None = None):
    """Call Qwen via OpenAI-compatible endpoint (text-only)."""
    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("Qwen request failed.")
        try:
            # Prefer OpenAI-compatible endpoint if configured (matches qwen_test.py behavior)
            if QWEN_OPENAI_BASE_URL:
                normalized_model = _normalize_qwen_model(model_id) or "Qwen/Qwen3-Next-80B-A3B-Instruct"
                client = OpenAIClient(base_url=QWEN_OPENAI_BASE_URL, api_key=QWEN_OPENAI_API_KEY)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                try:
                    resp = client.chat.completions.create(
                        model=normalized_model,
                        messages=messages,
                        temperature=temperature,
                    )
                except Exception as inner_e:
                    err_text = str(inner_e).lower()
                    if ("maximum context" in err_text) or ("input tokens" in err_text):
                        # For this errored call only, truncate once to fit budget and retry
                        sys_adj, usr_adj = _truncate_pair_for_budget(system_prompt, prompt, CONTEXT_CHAR_BUDGET)
                        messages = [
                            {"role": "system", "content": sys_adj},
                            {"role": "user", "content": usr_adj},
                        ]
                        resp = client.chat.completions.create(
                            model=normalized_model,
                            messages=messages,
                            temperature=temperature,
                        )
                    else:
                        raise
                try:
                    message = resp.choices[0].message
                    content = getattr(message, "content", None)
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, dict):
                                val = part.get("text") or part.get("content") or ""
                                if isinstance(val, str):
                                    parts.append(val)
                            elif isinstance(part, str):
                                parts.append(part)
                        content = "\n".join([p for p in parts if p])
                    elif not isinstance(content, str):
                        content = message.get("content") if isinstance(message, dict) else None
                    if isinstance(content, str) and content.strip():
                        return _maybe_strip_think(content).strip()
                except Exception:
                    pass
                # As a fallback, stringify
                return str(resp)
            # No DashScope fallback in simplified text-only mode
            raise RuntimeError("QWEN_OPENAI_BASE_URL not configured")
        except Exception as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1

def arrange_message_for_gpt(item_list):
    # Text-only arrangement: concatenate all text items, ignore images
    combined_text = ""
    for item_type, value in item_list:
        if item_type == "text" and isinstance(value, str):
            combined_text += value
    return [{"role": "user", "content": combined_text}]

def call_gpt_with_messages(messages, model_id="Qwen/Qwen3-Next-80B-A3B-Instruct", system_prompt=DEFAULT_SYSTEM_PROMPT, max_tokens: int | None = None):
    """Call Qwen via OpenAI-compatible endpoint (text-only messages)."""
    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("Qwen request failed.")
        try:
            normalized_model = _normalize_qwen_model(model_id) or "Qwen/Qwen3-Next-80B-A3B-Instruct"
            client = OpenAIClient(base_url=QWEN_OPENAI_BASE_URL, api_key=QWEN_OPENAI_API_KEY)
            # Ensure system prompt is prepended if not provided
            msgs = messages if (messages and messages[0].get("role") == "system") else ([{"role": "system", "content": system_prompt}] + messages)
            try:
                resp = client.chat.completions.create(
                    model=normalized_model,
                    messages=msgs,
                    temperature=0.5,
                )
            except Exception as inner_e:
                err_text = str(inner_e).lower()
                if ("maximum context" in err_text) or ("input tokens" in err_text):
                    # For this errored call only, keep only system + last user message, truncate once to fit budget
                    sys_content = msgs[0]["content"] if (msgs and msgs[0].get("role") == "system") else system_prompt
                    last_user_content = ""
                    for i in range(len(msgs) - 1, -1, -1):
                        if str(msgs[i].get("role")).lower() == "user":
                            last_user_content = msgs[i].get("content")
                            break
                    sys_adj, usr_adj = _truncate_pair_for_budget(sys_content, last_user_content, CONTEXT_CHAR_BUDGET)
                    msgs = [
                        {"role": "system", "content": sys_adj},
                        {"role": "user", "content": usr_adj},
                    ]
                    resp = client.chat.completions.create(
                        model=normalized_model,
                        messages=msgs,
                        temperature=0.5,
                    )
                else:
                    raise
            try:
                message = resp.choices[0].message
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            val = part.get("text") or part.get("content") or ""
                            if isinstance(val, str):
                                parts.append(val)
                        elif isinstance(part, str):
                            parts.append(part)
                    content = "\n".join([p for p in parts if p])
                elif not isinstance(content, str):
                    content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str) and content.strip():
                    return _maybe_strip_think(content).strip()
            except Exception:
                pass
            return str(resp)
        except Exception as e:
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
    print(call_gpt(prompt=prompt, model_id="qwen-plus"))
from openai import OpenAI
import os
import time
from typing import Tuple

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# vLLM API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "### API_KEY ###")
OPENAI_API_BASE = "### API_BASE_URL ###"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Initialize client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

def _normalize_llama_model_id(model_id: str) -> str:
    """
    Normalize common llama model aliases to the server's canonical id.
    Falls back to DEFAULT_MODEL if input is empty or an unknown short alias like 'llama'.
    """
    if not model_id:
        return DEFAULT_MODEL
    mid = str(model_id).strip()
    lower = mid.lower()
    # Common aliases used in configs
    if lower in {
        "llama",
        "llama-3.3-70b-instruct",
        "llama3.3-70b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
    }:
        return DEFAULT_MODEL
    return mid

def _is_context_exceeded_error(err: Exception) -> bool:
    s = (str(err) or "").lower()
    keys = [
        "maximum context length",
        "maximum context",
        "reduce the length of the input messages",
        "input tokens",
        "context length is",
    ]
    return any(k in s for k in keys)

def _clip_text_by_chars(text: str, max_chars: int) -> str:
    try:
        if not isinstance(text, str) or max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        head = max_chars * 7 // 10
        tail = max_chars - head
        return f"{text[:head]}\n...[TRUNCATED]...\n{text[-tail:]}"
    except Exception:
        try:
            s = str(text)
            return s[:max_chars]
        except Exception:
            return ""

def call_llama(prompt, model_id=DEFAULT_MODEL, system_prompt=DEFAULT_SYSTEM_PROMPT, temperature=0.7):
    """
    Call Llama model with a simple prompt string.
    Converts prompt to chat format internally.
    """
    model_id = _normalize_llama_model_id(model_id)
    original_prompt = prompt
    # 1st: original, 2nd: 100k chars, 3rd: 50k chars gradual truncation
    shrink_budgets = [None, 100000, 50000]
    shrink_idx = 0
    
    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("Llama request failed.")
        try:
            user_content = original_prompt if shrink_budgets[shrink_idx] is None else _clip_text_by_chars(original_prompt, shrink_budgets[shrink_idx])  # type: ignore[index]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            chat_response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=512,
                temperature=temperature
            )
            
            response_text = chat_response.choices[0].message.content
            return response_text.strip() if isinstance(response_text, str) else ""
            
        except Exception as e:
            if _is_context_exceeded_error(e) and shrink_idx + 1 < len(shrink_budgets):
                # On context overflow: truncate only this step input and retry immediately (no wait)
                shrink_idx += 1
                continue
            else:
                print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
                time.sleep(30)
                num_attempts += 1

def arrange_message_for_llama(item_list):
    """
    Flatten to a single text block (images not supported here to keep parity with current pipeline)
    """
    for item in item_list:
        if item[0] == "image":
            raise NotImplementedError()
    prompt = "".join([item[1] for item in item_list])
    return prompt

def _flatten_messages_for_llama(messages):
    """
    Flatten messages to a single text string (same logic as Gemini)
    """
    try:
        text_parts = []
        for m in messages:
            content = m.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "text":
                        text_parts.append(str(c.get("text", "")))
                    elif "image" in str(c.get("type", "")):
                        raise NotImplementedError()
            elif isinstance(content, str):
                text_parts.append(content)
        return "".join(text_parts)
    except Exception:
        # Fallback: if messages is already a string
        if isinstance(messages, str):
            return messages
        raise

def call_llama_with_messages(messages, model_id=DEFAULT_MODEL, system_prompt=DEFAULT_SYSTEM_PROMPT, temperature=0.5):
    """
    Accept either a combined string or a GPT-style messages list
    """
    model_id = _normalize_llama_model_id(model_id)
    if isinstance(messages, str):
        prompt = messages
    else:
        prompt = _flatten_messages_for_llama(messages)
    return call_llama(prompt=prompt, model_id=model_id, system_prompt=system_prompt, temperature=temperature)

if __name__ == "__main__":
    print(call_llama('''CURRENT OBSERVATION:
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
Today is 6/12/2023. Base on the aforementioned webpage, tell me how many fulfilled orders I have over the past month, and the total amount of money I spent over the past month.'''))
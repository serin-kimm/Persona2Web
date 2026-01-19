import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union
import random

import numpy as np
import numpy.typing as npt
from beartype import beartype
from beartype.door import is_bearable
from gymnasium import Env
from gymnasium.spaces import Box, Text
from playwright.sync_api import (
    CDPSession,
    Page,
    Playwright,
    ViewportSize,
    expect,
    sync_playwright,
)

# Optional stealth support
try:
    from playwright_stealth import stealth_sync as _stealth_sync  # type: ignore
except Exception:
    _stealth_sync = None

from .actions import Action, execute_action, get_action_space
from .processors import ObservationHandler, ObservationMetadata
from .utils import (
    AccessibilityTree,
    DetachedPage,
    Observation,
    png_bytes_to_numpy,
)

import base64
from .scripts import *

@dataclass
class PlaywrightScript:
    function: str  # goto, get_by_role
    destination: str  # https://www.google.com/, combobox
    name: str | None = None  # Search, Avatar 2009
    operation: str | None = None  # click, fill, press
    value: str | None = None  # avatar movie, Enter


def parse_action(action: str) -> PlaywrightScript:
    splitted = action.strip().split(" ")
    assert len(splitted) >= 2
    match splitted[:2]:
        case ["goto", url]:
            assert len(splitted) == 2
            return PlaywrightScript("goto", url)
        case ["get_by_role", destination]:
            assert len(splitted) >= 4
            match splitted[2:]:
                case [name, operation]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation
                    )
                case [name, operation, value]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation, value
                    )
                case _:
                    raise ValueError("Invalid action")
        case _:
            raise ValueError(f"Invalid action {action}")


class ScriptBrowserEnv(Env[dict[str, Observation], Action]):
    """
    The goal of this environment is to produce a prototype of a browser environment.
    In the end, we want to support a fully configurable browser environment with wide
    range of action spaces and observation spaces, both structured and unstructured.
    But in this prototype, we just support action space specified by Playwright script,
    and observation space is the html content of the page.
    """

    @beartype
    def __init__(
        self,
        max_page_length: int = 8192,
        headless: bool = True,
        slow_mo: int = 0,
        observation_type: str = "html",
        current_viewport_only: bool = False,
        viewport_size: ViewportSize = {"width": 1280, "height": 720},
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 5.0,
        global_config = None,
    ):
        # TODO: make Space[Action] = ActionSpace
        self.action_space = get_action_space()  # type: ignore[assignment]
        self.headless = headless
        self.slow_mo = slow_mo
        self.current_viewport_only = current_viewport_only
        self.reset_finished = False
        self.viewport_size = viewport_size
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution
        self.global_config = global_config

        match observation_type:
            case "html" | "accessibility_tree":
                self.text_observation_type = observation_type
                self.image_observation_type = ""
                self.main_observation_type = "text"
            case "image":
                self.image_observation_type = observation_type
                self.text_observation_type = ""  # type: ignore[assignment]
                self.main_observation_type = "image"
            case _:
                raise ValueError(
                    f"Unsupported observation type: {observation_type}"
                )

        self.observation_handler = ObservationHandler(
            self.main_observation_type,
            self.text_observation_type,
            self.image_observation_type,
            self.current_viewport_only,
            self.viewport_size,
        )

        self.observation_space = (
            self.observation_handler.get_observation_space()
        )

    @beartype
    def setup(self, config_file: Path | None = None) -> None:
        def handle_dialog(dialog):
            self.page.dialog_message = dialog.message
            dialog.dismiss()
        self.context_manager = sync_playwright()
        self.playwright = self.context_manager.__enter__()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless, slow_mo=self.slow_mo,
            args=[
            '--disable-blink-features=AutomationControlled',
            '--disable-dev-shm-usage',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor',
            '--disable-http2',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
        ]
        )

        if config_file:
            with open(config_file, "r") as f:
                instance_config = json.load(f)
        else:
            instance_config = {}

        storage_state = instance_config.get("storage_state", None)
        start_url = instance_config.get("start_url", None)
        geolocation = instance_config.get("geolocation", None)

        self.context = self.browser.new_context(
            viewport=self.viewport_size,
            storage_state=storage_state,
            geolocation=geolocation,
            device_scale_factor=1,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            locale='en-US',
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',  # Set English priority
            }
        )
        # Ensure stealth and handlers are applied to ANY new page in this context
        def _on_new_page(new_page: Page):
            # Apply stealth as soon as the page is created
            if _stealth_sync is not None:
                try:
                    _stealth_sync(new_page)
                except Exception as e:
                    print(f"[stealth] Failed to apply on context 'page' event: {e}")
            # Attach dialog handler
            new_page.on("dialog", handle_dialog)
            # Prepare CDP client and enable Accessibility if needed
            try:
                client = new_page.context.new_cdp_session(new_page)
                if self.text_observation_type == "accessibility_tree":
                    client.send("Accessibility.enable")
                new_page.client = client  # type: ignore
            except Exception as e:
                print(f"[context] Failed to setup CDP client for new page: {e}")

        try:
            self.context.on("page", _on_new_page)
        except Exception as e:
            print(f"[context] Failed to attach 'page' event handler: {e}")
        if self.save_trace_enabled:
            self.context.tracing.start(screenshots=True, snapshots=True)
        if start_url:
            start_urls = start_url.split(" |AND| ")
            for url in start_urls:
                page = self.context.new_page()
                # Apply stealth on new page if available
                if _stealth_sync is not None:
                    try:
                        _stealth_sync(page)
                    except Exception as e:
                        print(f"[stealth] Failed to apply on initial page: {e}")
                page.on("dialog", handle_dialog)
                client = page.context.new_cdp_session(
                    page
                )  # talk to chrome devtools
                if self.text_observation_type == "accessibility_tree":
                    client.send("Accessibility.enable")
                page.client = client  # type: ignore # TODO[shuyanzh], fix this hackey client
                try:
                    page.goto(url, timeout=30000, wait_until="domcontentloaded")
                except Exception as e:
                    print(f"Page goto failed: {e}, trying with networkidle...")
                    try:
                        page.goto(url, timeout=30000, wait_until="networkidle")
                    except Exception as e2:
                        print(f"Page goto failed again: {e2}, trying with commit...")
                        try:
                            page.goto(url, timeout=30000, wait_until="commit")
                        except Exception as e3:
                            print(f"All page goto attempts failed: {e3}")
                            # Create a simple error page content
                            page.set_content(f"""
                            <html>
                            <body>
                                <h1>Page Load Error</h1>
                                <p>Failed to load: {url}</p>
                                <p>Error: {str(e3)}</p>
                            </body>
                            </html>
                            """)
            # set the first page as the current page
            self.page = self.context.pages[0]
            self.page.bring_to_front()
        else:
            self.page = self.context.new_page()
            # Apply stealth on new page if available
            if _stealth_sync is not None:
                try:
                    _stealth_sync(self.page)
                except Exception as e:
                    print(f"[stealth] Failed to apply on default page: {e}")
            self.page.on("dialog", handle_dialog)
            client = self.page.context.new_cdp_session(self.page)
            if self.text_observation_type == "accessibility_tree":
                client.send("Accessibility.enable")
            self.page.client = client  # type: ignore

    def get_page_client(self, page: Page) -> CDPSession:
        return page.client  # type: ignore
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL for comparison"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except:
            return url.lower()
    
    def _ensure_page_has_client(self, page: Page) -> None:
        """Ensure the page has a CDP client, create one if missing"""
        if not hasattr(page, 'client') or page.client is None:
            print(f"🔧 Setting up CDP client for page: {page.url}")
            client = page.context.new_cdp_session(page)
            if self.text_observation_type == "accessibility_tree":
                client.send("Accessibility.enable")
            page.client = client  # type: ignore
            print(f"✅ CDP client configured for {page.url}")

    def _get_obs(self) -> dict[str, Observation]:
        obs = self.observation_handler.get_observation(
            self.page, self.get_page_client(self.page)
        )
        return obs

    def _get_obs_metadata(self) -> dict[str, ObservationMetadata]:
        metadata = self.observation_handler.get_observation_metadata()
        return metadata

    @beartype
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Reset the environment.
        :param options: options for the environment. The current supported options are:
            - "storage_state": the storage state of the browser. It is a file path to a json file.
        """
        super().reset(seed=seed, options=options)
        if self.reset_finished:
            self.context_manager.__exit__()

        if options is not None and "config_file" in options:
            config_file = Path(options["config_file"])
            if config_file.exists():
                self.setup(config_file=config_file)
            else:
                raise ValueError(f"Config file {config_file} does not exist.")
        else:
            self.setup()
        self.reset_finished = True

        if self.sleep_after_execution > 0:
            # Use uniform to support float ranges
            time.sleep(random.uniform(max(0, self.sleep_after_execution - 1), self.sleep_after_execution + 1))
            
        images = self.modify_page()

        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()
        info = {
            "page": DetachedPage(self.page.url, ""),
            "fail_error": "",
            "observation_metadata": observation_metadata,
            "images": images,
        } 

        return (observation, info)

    def save_trace(self, trace_path: str | Path) -> None:
        if self.save_trace_enabled:
            self.context.tracing.stop(path=trace_path)

    def close(self) -> None:
        if self.reset_finished:
            self.context_manager.__exit__()

    def step(
        self, action: Action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")

        success = False
        # Store current page count before action
        pages_before = len(self.context.pages)
        current_url_before = self.page.url
        print(f"🔍 Before action: {pages_before} tabs, current URL: {current_url_before}")
        
        fail_error = ""
        try:
            returned_page = execute_action(
                action,
                self.page,
                self.context,
                self.observation_handler.action_processor,
            )
            success = True
        except Exception as e:
            fail_error = str(e)
            raise e

        # Wait for potential new tabs to open (JavaScript async operations)
        time.sleep(1.0)
        
        # Debug: Check all tabs and their URLs
        print(f"🔍 All tabs after action:")
        for i, page in enumerate(self.context.pages):
            print(f"  Tab {i}: {page.url}")
        
        # Check for new tabs opened by JavaScript and focus on the newest one
        pages_after = len(self.context.pages)
        print(f"🔍 After action: {pages_after} tabs (was {pages_before})")
        
        if pages_after > pages_before:
            # New tab(s) opened - focus on the most recent one
            newest_page = self.context.pages[-1]
            # Apply stealth on new tab if available
            if _stealth_sync is not None:
                try:
                    _stealth_sync(newest_page)
                except Exception as e:
                    print(f"[stealth] Failed to apply on new tab: {e}")
            newest_page.bring_to_front()
            
            # Set up client for the new page (same as original pages)
            if not hasattr(newest_page, 'client'):
                client = newest_page.context.new_cdp_session(newest_page)
                if self.text_observation_type == "accessibility_tree":
                    client.send("Accessibility.enable")
                newest_page.client = client  # type: ignore
            
            self.page = newest_page
            print(f"🔄 NEW TAB DETECTED and focused: {newest_page.url if newest_page.url != 'about:blank' else 'blank page'}")
        else:
            # No new tabs created, check what execute_action returned
            print(f"🔍 execute_action returned page: {returned_page.url}")
            
            # Find which tab the returned page corresponds to
            returned_tab_index = -1
            for i, page in enumerate(self.context.pages):
                if page == returned_page:
                    returned_tab_index = i
                    break
            
            print(f"🔍 Returned page is Tab {returned_tab_index}: {returned_page.url}")
            
            # If the returned page is different from current page, switch to it
            if returned_page != self.page and returned_page.url != current_url_before:
                returned_page.bring_to_front()
                # Apply stealth on returned page if available
                if _stealth_sync is not None:
                    try:
                        _stealth_sync(returned_page)
                    except Exception as e:
                        print(f"[stealth] Failed to apply on returned page: {e}")
                self._ensure_page_has_client(returned_page)  # Ensure client before switching
                self.page = returned_page
                print(f"🔄 SWITCHED to returned page: {returned_page.url}")
            else:
                # No new tabs, use the page returned by execute_action
                print(f"🔄 No new tabs detected, using returned page")
                # Apply stealth on returned page if available
                if _stealth_sync is not None:
                    try:
                        _stealth_sync(returned_page)
                    except Exception as e:
                        print(f"[stealth] Failed to apply on returned page: {e}")
                self._ensure_page_has_client(returned_page)  # Ensure client before using
                self.page = returned_page
                print(f"🔍 Returned page URL: {returned_page.url}")
                
                # Wait additional time for potential page changes
                time.sleep(0.5)
                current_url_after_wait = self.page.url
                print(f"🔍 Page URL after additional wait: {current_url_after_wait}")
                
                # Check if page URL changed significantly (navigation within same tab)
                if self.page.url != current_url_before:
                    print(f"🔄 Page navigated: {current_url_before} → {self.page.url}")
                    self.page.bring_to_front()  # Ensure focus
                else:
                    print(f"🔄 Same page: {self.page.url}")
                    # Force page refresh to check if there are pending navigations
                    try:
                        self.page.wait_for_load_state("networkidle", timeout=2000)
                        final_url = self.page.url
                        if final_url != current_url_before:
                            print(f"🔄 Delayed navigation detected: {current_url_before} → {final_url}")
                    except:
                        print("🔍 No delayed navigation detected")

        # hard sleep TODO[shuyanzh] suboptimal, may need to check network
        if self.sleep_after_execution > 0:
            time.sleep(self.sleep_after_execution)

        # Ensure current page has client before using it
        self._ensure_page_has_client(self.page)

        images = self.modify_page()
        
        print(f"🔍 Getting observation from page: {self.page.url}")
        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()
        print(f"🔍 Observation text length: {len(str(observation))}")

        info = {
            "page": DetachedPage(self.page.url, self.page.content()),
            "fail_error": fail_error,
            "observation_metadata": observation_metadata,
            "images": images,
        }
        
        msg = (
            observation,
            float(success),  # reward
            False,  # terminated
            False,  # truncated
            info,
        )
        return msg

    def modify_page(self):
        self.page.wait_for_timeout(500)
        try:
            self.page.evaluate(remove_id_script)
        except:
            pass
        
        # Skip screenshot entirely - not needed for text-only observation
        raw_image = ""
        
        self.page.evaluate(mix_marker_script)
        self.page.wait_for_timeout(100)
        
        # get all clickable elements
        start_id = 0
        elem_items, start_id = self.page.evaluate(get_rect_script, {
            "selector": ".possible-clickable-element",
            "startIndex": start_id
        })
        
        # get ocr items
        ocr_items = []
        # ocr_items = page.evaluate(canva_handler_script)
        # svg_items, _ = page.evaluate(get_rect_script, {"selector": "svg", "startIndex": -1})
        # ocr_items = ocr_items + svg_items
        # ocr_items, start_id = get_canva_images(ocr_items, img_bytes, start_id)
        
        items = elem_items + ocr_items
        
        # mark our own labels and get the images
        items = self.page.evaluate(label_marker_script, items)
        # Skip marked screenshot entirely - not needed for text-only observation
        marked_image = ""
        
        self.page.evaluate(remove_label_mark_script)
        
        return {
            "raw_image": raw_image,
            "marked_image": marked_image,
        }
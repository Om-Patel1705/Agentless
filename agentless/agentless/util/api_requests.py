import time
from typing import Dict, Union

import anthropic
import openai
import tiktoken


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def create_gemini_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "",
    model: str = "gemini-2.0-flash"
) -> Dict:
    if isinstance(message, list):
        prompt = "\n".join([m["content"] for m in message])
    else:
        prompt = message.strip()
    config = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    return config

# def request_gemini_engine(config, logger, max_retries=40, timeout=100):
#     import os
#     import time
#     import requests

#     gemini_api_key = os.environ.get("GEMINI_API_KEY")
#     if gemini_api_key is None:
#         raise ValueError("GEMINI_API_KEY is not set. Please set it in your environment.")
    
#     endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
#     headers = {
#         "Content-Type": "application/json",
#         "Accept": "application/json"
#     }
#     retries = 0
#     ret = None
#     while ret is None and retries < max_retries:
#         try:
#             response = requests.post(endpoint, headers=headers, json=config, timeout=timeout)
#             response.raise_for_status()
#             ret = response.json()
#         except requests.exceptions.HTTPError as e:
#             if response.status_code == 429:
#                 wait_time = 2 ** retries  # exponential backoff
#                 logger.error("Rate limit exceeded. Waiting for %s seconds", wait_time)
#                 time.sleep(wait_time)
#             else:
#                 logger.error("Error calling Gemini API: %s", e)
#                 time.sleep(5)
#             retries += 1
#     return ret

def request_gemini_engine(config, logger, model_name, max_retries=40, timeout=100): # Added 'model_name' parameter
    import os
    import time
    import requests
    import sys # Added for sys.exit

    # --- FIX 1: Check for both common environment variables ---
    gemini_api_key = os.environ.get("GEMINI_API_KEY") # Check standard first
    if gemini_api_key is None:
        gemini_api_key = os.environ.get("GEMINI_API_KEY") # Check alternative
    logger.info(gemini_api_key)
    if gemini_api_key is None:
        logger.error("API Key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        # Raising an error or exiting might be better than just logging
        # raise ValueError("API Key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        sys.exit("API Key not found. Exiting.") # Exit more forcefully

    # --- FIX 2: Use the provided 'model_name' in the endpoint URL ---
    # Use the correct API version prefix for the models ('v1beta' or 'v1')
    # Generally 'models/' followed by model name is standard for generateContent
    # Check Google documentation for the exact endpoint format if this still fails.
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"
    logger.info(f"Attempting to contact Gemini endpoint: {endpoint.split('?')[0]}?key=...") # Log endpoint without key

    headers = {
        "Content-Type": "application/json",
        # Removed "Accept" header, often not needed and sometimes causes issues
    }
    retries = 0
    ret = None
    while ret is None and retries < max_retries:
        try:
            logger.info(f"Making Gemini API request (Attempt {retries + 1})...")
            response = requests.post(endpoint, headers=headers, json=config, timeout=timeout)
            # Log status code and reason for debugging
            logger.info(f"Gemini API response status: {response.status_code} {response.reason}")
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            ret = response.json()
            # --- Add check for empty or error response from Gemini ---
            if not ret or "candidates" not in ret or not ret["candidates"]:
                 logger.warning(f"Gemini API returned success status ({response.status_code}) but response is empty or missing candidates.")
                 logger.debug(f"Full Gemini JSON response: {ret}")
                 # Decide if you want to retry or return None/empty
                 # For now, let's treat it as a failure state to potentially retry or signal issues
                 ret = None # Reset ret to potentially retry if error is transient
                 # Or you could return a specific marker: return {"error": "Empty response"}
                 raise requests.exceptions.RequestException("Empty or invalid response structure from Gemini")


        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error calling Gemini API: {e.response.status_code} {e.response.reason}")
            # Log the response body if available, might contain error details
            try:
                error_details = e.response.json()
                logger.error(f"Error details from API: {error_details}")
            except ValueError: # Handle cases where response body is not JSON
                 logger.error(f"Raw error response: {e.response.text}")

            if e.response.status_code == 429: # Rate limit
                wait_time = min(2 ** retries, 60) # Exponential backoff up to 60s
                logger.warning(f"Rate limit exceeded. Waiting for {wait_time} seconds")
                time.sleep(wait_time)
            elif e.response.status_code == 400: # Bad Request (often invalid model or config)
                 logger.error("Bad Request (400). Check model name and request config.")
                 # No point retrying a 400 error usually
                 return {"error": "Bad Request", "details": error_details if 'error_details' in locals() else e.response.text}
            elif e.response.status_code == 403: # Forbidden (API key issue?)
                 logger.error("Forbidden (403). Check API key permissions or billing.")
                 return {"error": "Forbidden", "details": error_details if 'error_details' in locals() else e.response.text}
            elif e.response.status_code >= 500: # Server error
                 logger.warning("Server error (5xx). Waiting and retrying...")
                 time.sleep(5 + retries) # Simple backoff for server errors
            else: # Other client errors
                 logger.error(f"Unhandled Client Error ({e.response.status_code}).")
                 return {"error": f"Client Error {e.response.status_code}", "details": error_details if 'error_details' in locals() else e.response.text}
            # Reset ret to None to force retry (unless it was a non-retriable error like 400/403)
            ret = None

        except requests.exceptions.RequestException as e:
            # Includes connection errors, timeouts, etc.
            logger.error(f"Network or Request Error calling Gemini API: {e}")
            wait_time = min(2 ** retries, 30) # Exponential backoff up to 30s for network issues
            logger.warning(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            ret = None # Reset ret to force retry

        except Exception as e:
            # Catch any other unexpected errors during the request/response handling
            logger.error(f"Unexpected error during Gemini request: {e}", exc_info=True) # Log traceback
            wait_time = min(2 ** retries, 30)
            logger.warning(f"Waiting for {wait_time} seconds after unexpected error...")
            time.sleep(wait_time)
            ret = None # Reset ret to force retry

        retries += 1

    if ret is None:
        logger.error(f"Failed to get response from Gemini API after {max_retries} retries.")
        return {"error": "Max retries exceeded"} # Return an error indicator

    # Log successful response structure for debugging
    logger.info("Successfully received response from Gemini.")
    logger.debug(f"Gemini response structure: {list(ret.keys()) if isinstance(ret, dict) else 'Not a dict'}")

    return ret
def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = openai.OpenAI(base_url=base_url)

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
    tools: list = None,
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

    if tools:
        config["tools"] = tools

    return config


def request_anthropic_engine(
    config, logger, max_retries=40, timeout=500, prompt_cache=False
):
    ret = None
    retries = 0

    client = anthropic.Anthropic()

    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                config["messages"][0]["content"][0]["cache_control"] = {
                    "type": "ephemeral"
                }
                ret = client.beta.prompt_caching.messages.create(**config)
            else:
                ret = client.messages.create(**config)
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            time.sleep(10 * retries)
        retries += 1

    return ret

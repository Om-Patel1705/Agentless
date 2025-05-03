import json
from abc import ABC, abstractmethod
from typing import List

from agentless.util.api_requests import (
    create_anthropic_config,
    create_chatgpt_config,
    request_anthropic_engine,
    request_chatgpt_engine,
)

from agentless.util.api_requests import create_gemini_config, request_gemini_engine

class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        logger,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
    ) -> None:
        logger.info("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.logger = logger
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

# class GeminiChatDecoder(DecoderBase):
#     def __init__(self, name: str, logger, **kwargs) -> None:
#         super().__init__(name, logger, **kwargs)

#     def codegen(self, message: str, num_samples: int = 1, prompt_cache: bool = False) -> List[dict]:
#         import os
#         gemini_key = os.environ.get("GEMINI_API_KEY")
#         if gemini_key is None:
#             raise ValueError("GEMINI_API_KEY is not set. Please set it in your environment.")
#         # Build Gemini-specific config (see step 3 below for a sample function)
#         from agentless.util.api_requests import create_gemini_config, request_gemini_engine
#         config = create_gemini_config(
#             message=message,
#             max_tokens=self.max_new_tokens,
#             temperature=self.temperature,
#             batch_size=1,
#             model=self.name,
#         )
#         ret = request_gemini_engine(config, self.logger,model_name=self.name)
#         if ret:
#             responses = [ret.get("response", "")]
#             completion_tokens = ret.get("completion_tokens", 0)
#             prompt_tokens = ret.get("prompt_tokens", 0)
#         else:
#             responses = [""]
#             completion_tokens = 0
#             prompt_tokens = 0

#         trajs = [{
#             "response": responses[0],
#             "usage": {
#                 "completion_tokens": completion_tokens,
#                 "prompt_tokens": prompt_tokens,
#             },
#         }]
#         for response in responses[1:]:
#             trajs.append({
#                 "response": response,
#                 "usage": {
#                     "completion_tokens": 0,
#                     "prompt_tokens": 0,
#                 },
#             })
#         return trajs

#     def is_direct_completion(self) -> bool:
#         return False
class GeminiChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)
        # Log the specific model being initialized for clarity
        self.logger.info(f"GeminiChatDecoder initialized with model name: {self.name}")

    def codegen(self, message: str, num_samples: int = 1, prompt_cache: bool = False) -> List[dict]:
        # Gemini API (at least the basic generateContent) doesn't directly support num_samples > 1 in one call easily.
        # We will only generate 1 sample regardless of num_samples for simplicity here.
        # If multiple samples are strictly needed, you'd need multiple API calls.
        if num_samples > 1:
            self.logger.warning(f"Gemini backend in this implementation currently only supports num_samples=1. Requested {num_samples}, generating 1.")

        # Note: API key check is now primarily handled within request_gemini_engine
        # gemini_key = os.environ.get("GOOGLE_API_KEY") # Use GOOGLE_API_KEY as standard
        # if gemini_key is None:
        #    gemini_key = os.environ.get("GEMINI_API_KEY")
        # if gemini_key is None:
        #    self.logger.error("API Key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        #    # Return an empty/error trajectory
        #    return [{"response": "", "usage": {"completion_tokens": 0, "prompt_tokens": 0}, "error": "API Key missing"}]


        config = create_gemini_config(
            message=message,
            max_tokens=self.max_new_tokens, # Note: Gemini uses safety settings, not strict max_tokens in the body often
            temperature=self.temperature,
            batch_size=1, # Not used by create_gemini_config currently
            model=self.name, # Passed but not used by create_gemini_config currently
        )

        # --- FIX 1: Pass self.name (model_name) to the engine ---
        ret = request_gemini_engine(config, self.logger, model_name=self.name)

        # --- FIX 2: Correctly parse the Gemini API response ---
        response_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        error_info = None

        try:
            if ret and "candidates" in ret and ret["candidates"]:
                # Get text from the first candidate
                first_candidate = ret["candidates"][0]
                if "content" in first_candidate and "parts" in first_candidate["content"] and first_candidate["content"]["parts"]:
                    response_text = first_candidate["content"]["parts"][0].get("text", "")
                    self.logger.info(f"--- Gemini Response Text Received ---\n{response_text}\n-----------------------------------")
                else:
                    error_info = "Response structure missing content/parts/text."
                    self.logger.warning(f"{error_info} Full candidate: {first_candidate}")


                # Attempt to get token counts (structure might vary or be absent)
                if "usageMetadata" in ret:
                    prompt_tokens = ret["usageMetadata"].get("promptTokenCount", 0)
                    # Gemini often reports candidatesTokenCount (plural) or just total
                    completion_tokens = ret["usageMetadata"].get("candidatesTokenCount", ret["usageMetadata"].get("completionTokenCount", 0))
                    total_tokens = ret["usageMetadata"].get("totalTokenCount", 0)
                    # If completion is missing but total and prompt are present, calculate it
                    if completion_tokens == 0 and total_tokens > 0 and prompt_tokens > 0:
                         completion_tokens = total_tokens - prompt_tokens
                else:
                     self.logger.warning("usageMetadata not found in Gemini response.")

            elif ret and "error" in ret:
                 # Handle errors returned structurally from request_gemini_engine
                 error_info = f"API Error: {ret.get('error')} - {ret.get('details', 'No details')}"
                 self.logger.error(error_info)
            else:
                 # Handle cases where ret is None or empty
                 error_info = "Received None or empty response from request_gemini_engine."
                 self.logger.error(error_info)

        except Exception as e:
            error_info = f"Error parsing Gemini response: {e}"
            self.logger.error(error_info, exc_info=True) # Log traceback
            # Ensure response_text is empty if parsing fails badly
            response_text = ""


        # Create the trajectory structure
        traj = {
            "response": response_text.strip(), # Strip whitespace
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                # Add total tokens if needed elsewhere
                # "total_tokens": total_tokens
            },
        }
        # Add error info if present
        if error_info:
             traj["error"] = error_info


        # Return as a list containing the single trajectory
        return [traj]

    def is_direct_completion(self) -> bool:
        return False # Gemini is chat/instruction-based
class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )
        ret = request_chatgpt_engine(config, self.logger)
        if ret:
            responses = [choice.message.content for choice in ret.choices]
            completion_tokens = ret.usage.completion_tokens
            prompt_tokens = ret.usage.prompt_tokens
        else:
            responses = [""]
            completion_tokens = 0
            prompt_tokens = 0

        # The nice thing is, when we generate multiple samples from the same input (message),
        # the input tokens are only charged once according to openai API.
        # Therefore, we assume the request cost is only counted for the first sample.
        # More specifically, the `prompt_tokens` is for one input message,
        # and the `completion_tokens` is the sum of all returned completions.
        # Therefore, for the second and later samples, the cost is zero.
        trajs = [
            {
                "response": responses[0],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        ]
        for response in responses[1:]:
            trajs.append(
                {
                    "response": response,
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                    },
                }
            )
        return trajs

    def is_direct_completion(self) -> bool:
        return False


class AnthropicChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    _STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for editing files
* State is persistent across command calls and discussions with the user

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

    _USER_REPLY_EDIT_MESSAGE = """File is successfully edited"""

    tools = [
        {
            "name": "str_replace_editor",
            "description": _STR_REPLACE_EDITOR_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "description": "Full path to file, e.g. `folder/file.py`.",
                        "type": "string",
                    },
                    "old_str": {
                        "description": "Required parameter containing the string in `path` to replace.",
                        "type": "string",
                    },
                    "new_str": {
                        "description": "Optional parameter containing the new string (if not given, no string will be added).",
                        "type": "string",
                    },
                },
                "required": ["path", "old_str"],
            },
        }
    ]

    MAX_CODEGEN_ITERATIONS = 10

    # specialized codegen with tool
    def codegen_w_tool(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        def _build_response_and_extract(response, messages, iter):
            json_response = response.to_dict()

            contains_tool = False
            # formulate the messages
            json_response.pop("id")
            json_response.pop("model")
            json_response.pop("stop_reason")
            json_response.pop("stop_sequence")
            json_response.pop("type")
            json_response.pop("usage")

            messages.append(json_response)

            response_content = []

            for json_message in json_response["content"]:
                if json_message["type"] == "tool_use":
                    contains_tool = True
                    # each tool use requires a response
                    response_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": json_message["id"],
                            "content": self._USER_REPLY_EDIT_MESSAGE,
                        }
                    )

            if contains_tool:
                messages.append(
                    {
                        "role": "user",
                        "content": response_content,
                    }
                )
            else:
                if iter == 0:
                    # if the first iteration does not contain the tool, likely the model is doing some CoT for debugging
                    # append encouraging message
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please generate editing commands to fix the issue",
                                }
                            ],
                        }
                    )
                    contains_tool = True

            return messages, contains_tool

        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            self.logger.info(f" === Generating ====")
            # initialized the traj
            traj = {
                "response": [],
                "usage": {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "cache_creation_token": 0,
                    "cache_read_input_tokens": 0,
                },
            }

            # create the initial config and messages
            messages = [
                {"role": "user", "content": [{"type": "text", "text": message}]}
            ]

            for iteration in range(self.MAX_CODEGEN_ITERATIONS):
                config = create_anthropic_config(
                    message=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    batch_size=1,
                    model=self.name,
                    tools=self.tools,
                )
                ret = request_anthropic_engine(
                    config,
                    self.logger,
                    prompt_cache=True,  # prompt cache should be always true as we at least should query twice
                )

                if ret:
                    # add the response to the traj
                    traj["response"].append([reply.to_dict() for reply in ret.content])

                    # pretty dump the response
                    for reply in ret.content:
                        self.logger.info(json.dumps(reply.to_dict(), indent=2))

                    # update the usage
                    traj["usage"]["completion_tokens"] += ret.usage.output_tokens
                    traj["usage"]["prompt_tokens"] += ret.usage.input_tokens
                    traj["usage"][
                        "cache_creation_token"
                    ] += ret.usage.cache_creation_input_tokens
                    traj["usage"][
                        "cache_read_input_tokens"
                    ] += ret.usage.cache_read_input_tokens

                    messages, contains_tool = _build_response_and_extract(
                        ret, messages, iteration
                    )

                    if not contains_tool:
                        break
                else:
                    assert (
                        False
                    ), "No response from the engine"  # this should not happen

            if ret:
                trajs.append(traj)
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_anthropic_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_anthropic_engine(
                config, self.logger, prompt_cache=prompt_cache
            )

            if ret:
                trajs.append(
                    {
                        "response": ret.content[0].text,
                        "usage": {
                            "completion_tokens": ret.usage.output_tokens,
                            "prompt_tokens": ret.usage.input_tokens,
                            "cache_creation_token": 0
                            if not prompt_cache
                            else ret.usage.cache_creation_input_tokens,
                            "cache_read_input_tokens": 0
                            if not prompt_cache
                            else ret.usage.cache_read_input_tokens,
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class DeepSeekChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_chatgpt_engine(
                config, self.logger, base_url="https://api.deepseek.com"
            )
            if ret:
                trajs.append(
                    {
                        "response": ret.choices[0].message.content,
                        "usage": {
                            "completion_tokens": ret.usage.completion_tokens,
                            "prompt_tokens": ret.usage.prompt_tokens,
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


def make_model(
    model: str,
    backend: str,
    logger,
    batch_size: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.0,
):
    if backend == "openai":
        return OpenAIChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "anthropic":
        return AnthropicChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "gemini":
        return GeminiChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "deepseek":
        return DeepSeekChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise NotImplementedError

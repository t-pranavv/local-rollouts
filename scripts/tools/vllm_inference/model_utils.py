import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json

from tools import parse_tool_block, is_valid_json


class InvalidToolCallFormatException(Exception):
    """Exception raised for invalid tool call formatting."""

    def __init__(self, message: str = "Invalid tool call formatting, please follow the system instructions"):
        super().__init__(message)


class ToolCallEmptyException(Exception):
    """Exception raised when a tool call is empty."""

    def __init__(
        self,
        message: str = "No tool call found between {tool_call_start} {tool_call_end} tags.",
        tool_call_start: str = "<tool_call>",
        tool_call_end: str = "</tool_call>",
    ):
        message = message.format(tool_call_start=tool_call_start, tool_call_end=tool_call_end)
        super().__init__(message)


class BaseModelUtils(ABC):
    """Base class for model-specific utilities"""

    @abstractmethod
    def get_think_tokens(self) -> tuple[str, str]:
        """Return start and end tokens for thinking"""
        pass

    @abstractmethod
    def get_tool_call_tokens(self) -> tuple[str, str]:
        """Return start and end tokens for tool calls"""
        pass

    @abstractmethod
    def get_tool_response_tokens(self) -> tuple[str, str]:
        """Return start and end tokens for tool responses"""
        pass

    @abstractmethod
    def get_final_answer_tokens(self) -> tuple[str, str]:
        """Return start and end tokens for final answer"""
        pass

    @abstractmethod
    def get_generation_prompt(self, **kwargs) -> str:
        """Return the generation prompt for the model"""
        pass

    @abstractmethod
    def get_message_join_token(self) -> str:
        """Return the token used to join messages"""
        pass

    @abstractmethod
    def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True, **kwargs) -> str:
        """Apply model-specific chat template to messages"""
        pass

    def wrap_thinking(self, content: str) -> str:
        """Wrap content in thinking tokens"""
        start, end = self.get_think_tokens()
        return f"{start}\n{content}\n{end}"

    def wrap_tool_call(self, response: Any) -> str:
        """Wrap tool call in appropriate tokens"""
        start, end = self.get_tool_call_tokens()
        return (
            f"{start}\n{json.dumps(response, ensure_ascii=False) if not isinstance(response, str) else response}\n{end}"
        )

    def parse_tool_call(self, response: str) -> List[Dict]:
        """Parse tool calls from the model response"""
        if self.validate_tool_tags(response):
            return [
                {"tool_category": "tool", "tool_subcategory": "tool", "codeblock": self.extract_tool_call(response)}
            ]
        else:
            return [
                {
                    "tool_category": "exception",
                    "tool_subcategory": "exception",
                    "codeblock": str(InvalidToolCallFormatException()),
                }
            ]

    def validate_tool_tags(self, response: str) -> bool:
        """Validate that tool tags are correctly formatted in the response"""
        tool_call_start, tool_call_end = self.get_tool_call_tokens()
        pattern = re.compile(rf"{re.escape(tool_call_start)}.*?{re.escape(tool_call_end)}", re.DOTALL)
        return bool(pattern.search(response))

    def extract_tool_call(self, response: str) -> str:
        """Extract tool call from the response"""
        tool_call_start, tool_call_end = self.get_tool_call_tokens()
        pattern = re.compile(rf"{re.escape(tool_call_start)}(.*?){re.escape(tool_call_end)}", re.DOTALL)
        match = pattern.search(response)
        return match.group(1) if match else ""

    def strip_think_blocks(self, response: str) -> str:
        """Strip thinking blocks from the response"""
        think_start, think_end = self.get_think_tokens()
        pattern = re.compile(rf"{re.escape(think_start)}.*?{re.escape(think_end)}", re.DOTALL)
        return pattern.sub("", response).strip()

    def parse_tool_response(self, tool_responses: List[Dict]) -> str:
        """Parse tool response"""
        start, end = self.get_tool_response_tokens()
        return f"{start}\n{json.dumps(tool_responses, ensure_ascii=False) if not isinstance(tool_responses, str) else tool_responses}\n{end}"

    def add_tool_response(self, messages: str, tool_responses: List[Dict]) -> str:
        """Add tool response to messages"""
        tool_response_str = self.parse_tool_response(tool_responses)
        return f"{messages}\n{tool_response_str}"


class PhiThinkUtils(BaseModelUtils):
    """Utilities for Phi Think models"""

    def get_think_tokens(self) -> tuple[str, str]:
        return "<think>", "</think>"

    def get_tool_call_tokens(self) -> tuple[str, str]:
        return "<tool_call>", "</tool_call>"

    def get_tool_response_tokens(self) -> tuple[str, str]:
        return "<tool_result>", "</tool_result>"

    def get_final_answer_tokens(self) -> tuple[str, str]:
        return "<answer>", "</answer>"

    def get_generation_prompt(self, thinking=True, **kwargs) -> str:
        """Return the generation prompt for Phi Think model"""
        if thinking:
            return "<|im_start|>assistant<|im_sep>" + self.get_think_tokens()[0]
        else:
            return "<|im_start|>assistant<|im_sep>"

    def get_message_join_token(self) -> str:
        return ""

    def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True, **kwargs) -> str:
        """Apply Phi Think chat template"""
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                formatted_messages.append(f"<|im_start|>system<|im_sep|>{content}<|im_end|>")
            elif role == "user":
                formatted_messages.append(f"<|im_start|>user<|im_sep|>{content}<|im_end|>")
            elif role == "assistant":
                formatted_messages.append(f"<|im_start|>assistant<|im_sep|>{content}<|im_end|>")

        result = f"{self.get_message_join_token()}".join(formatted_messages)
        if add_generation_prompt:
            result += self.get_generation_prompt(**kwargs)
        return result

    def parse_tool_call(self, response: str) -> List[Dict]:
        """Parse tool call from Qwen Think model response"""
        start = self.get_think_tokens()[0]
        tool_call_start, tool_call_end = self.get_tool_call_tokens()
        response = self.strip_think_blocks(f"{start}{response}")
        tool_call = super().parse_tool_call(response)[0]
        if tool_call["tool_category"] != "exception":
            if tool_call["codeblock"].strip() == "":
                return [
                    {
                        "tool_category": "exception",
                        "tool_subcategory": "exception",
                        "codeblock": str(
                            ToolCallEmptyException(tool_call_start=tool_call_start, tool_call_end=tool_call_end)
                        ),
                    }
                ]
            else:
                # Parse the tool block
                return parse_tool_block(tool_call["codeblock"])
        else:
            # If there is an exception, return it as is
            return [tool_call]

    def add_tool_response(self, messages: str, tool_responses: List[Dict]) -> str:
        """Add tool response to messages"""
        start = self.get_think_tokens()[0]
        new_prompt = super().add_tool_response(messages, tool_responses)
        return f"{new_prompt}\n{start}"


class QwenThinkModelUtils(PhiThinkUtils):

    def get_message_join_token(self) -> str:
        return "\n"

    def get_generation_prompt(self, thinking=True, **kwargs) -> str:
        """Return the generation prompt for Qwen Think model"""
        if thinking:
            return "<|im_start|>assistant\n" + self.get_think_tokens()[0]
        else:
            return "<|im_start|>assistant\n"

    def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True, **kwargs) -> str:
        """Apply Qwen Think chat template"""
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                formatted_messages.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted_messages.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_messages.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        result = f"{self.get_message_join_token()}".join(formatted_messages) + "\n"
        if add_generation_prompt:
            result += self.get_generation_prompt(**kwargs)
        return result


class Phi4V1MSRIDataUtils(PhiThinkUtils):
    """Utilities for Phi-4 Think msri rl data format"""

    def parse_tool_response(self, tool_responses: List[Dict]) -> str:
        """Parse tool response for Phi-4 Think msri data format"""
        start, end = self.get_tool_response_tokens()
        tool_response = tool_responses[0]
        success, stdout, stderr, exception = (
            tool_response.get("success", True),
            tool_response.get("stdout", ""),
            tool_response.get("stderr", ""),
            tool_response.get("exception", ""),
        )
        if stdout.strip() != "":
            is_valid = is_valid_json(json.dumps(stdout.strip()))
            content = is_valid["content"]
            if is_valid["is_valid"]:
                tool_response_str = f"{content}"
            else:
                tool_response_str = f"Compiled successfully. Output: \n{stdout}"
        elif success:
            tool_response_str = "Compiled successfully. Output empty likely because print statements are missing."
        elif stderr.strip() != "":
            tool_response_str = f"Execution failed: \n{stderr}"
        elif exception.strip() != "":
            tool_response_str = f"Execution failed: \n{exception}"
        elif not success:
            tool_response_str = "Execution failed: Failed without any output. Please check the code."
        else:
            raise ValueError("No valid response content found")
        return f"{start}\n{tool_response_str}\n{end}"


class QwenV1MSRIDataUtils(QwenThinkModelUtils):

    def parse_tool_response(self, tool_responses: List[Dict]) -> str:
        """Parse tool response for Qwen Think msri data format"""
        start, end = self.get_tool_response_tokens()
        tool_response = tool_responses[0]
        success, stdout, stderr, exception = (
            tool_response.get("success", True),
            tool_response.get("stdout", ""),
            tool_response.get("stderr", ""),
            tool_response.get("exception", ""),
        )
        if stdout.strip() != "":
            is_valid = is_valid_json(json.dumps(stdout.strip()))
            content = is_valid["content"]
            if is_valid["is_valid"]:
                tool_response_str = f"{content}"
            else:
                tool_response_str = f"Compiled successfully. Output: \n{stdout}"
        elif success:
            tool_response_str = "Compiled successfully. Output empty likely because print statements are missing."
        elif stderr.strip() != "":
            tool_response_str = f"Execution failed: \n{stderr}"
        elif exception.strip() != "":
            tool_response_str = f"Execution failed: \n{exception}"
        elif not success:
            tool_response_str = "Execution failed: Failed without any output. Please check the code."
        else:
            raise ValueError("No valid response content found")
        return f"{start}\n{tool_response_str}\n{end}"


class Phi4V1TentDataUtils(PhiThinkUtils):
    """Utilities for Phi-4 Think v1 tent data format"""

    def parse_tool_response(self, tool_responses: List[Dict]) -> str:
        """Parse tool response for Phi-4 Think v1 tent data format"""
        start, end = self.get_tool_response_tokens()
        tool_response = tool_responses[0]
        success, stdout, stderr, exception = (
            tool_response.get("success", True),
            tool_response.get("stdout", ""),
            tool_response.get("stderr", ""),
            tool_response.get("exception", ""),
        )

        if stdout.strip() != "" or stderr.strip() != "":
            tool_response_str = ""
            if stdout.strip() != "":
                tool_response_str += f"[stdout]\n{stdout}\n"
            elif success:
                tool_response_str += (
                    "Compiled successfully. [stdout] empty likely because print statements are missing.\n"
                )
            if stderr.strip() != "":
                tool_response_str += f"[stderr]\n{stderr}\n"
            tool_response_str = tool_response_str.strip()
        elif success:
            tool_response_str = "Compiled successfully. [stdout] empty likely because print statements are missing."
        elif exception.strip() != "":
            tool_response_str = f"\u274c Exception:\n{exception}"
        elif not success:
            tool_response_str = "Execution failed: Failed without any output. Please check the code."
        else:
            raise ValueError("No valid response content found")
        return f"{start}\n{tool_response_str}\n{end}"


class QwenV1TentDataUtils(QwenThinkModelUtils):

    def parse_tool_response(self, tool_responses: List[Dict]) -> str:
        """Parse tool response for Qwen Think v1 tent data format"""
        start, end = self.get_tool_response_tokens()
        tool_response = tool_responses[0]
        success, stdout, stderr, exception = (
            tool_response.get("success", True),
            tool_response.get("stdout", ""),
            tool_response.get("stderr", ""),
            tool_response.get("exception", ""),
        )

        if stdout.strip() != "" or stderr.strip() != "":
            tool_response_str = ""
            if stdout.strip() != "":
                tool_response_str += f"[stdout]\n{stdout}\n"
            elif success:
                tool_response_str += (
                    "Compiled successfully. [stdout] empty likely because print statements are missing.\n"
                )
            if stderr.strip() != "":
                tool_response_str += f"[stderr]\n{stderr}\n"
            tool_response_str = tool_response_str.strip()
        elif success:
            tool_response_str = "Compiled successfully. [stdout] empty likely because print statements are missing."
        elif exception.strip() != "":
            tool_response_str = f"\u274c Exception:\n{exception}"
        elif not success:
            tool_response_str = "Execution failed: Failed without any output. Please check the code."
        else:
            raise ValueError("No valid response content found")
        return f"{start}\n{tool_response_str}\n{end}"


class ModelUtilsFactory:
    """Factory class to get appropriate model utils"""

    _utils_map = {
        "phi-think": PhiThinkUtils,
        "qwen-think": QwenThinkModelUtils,
        "phi4_v1_msri_data": Phi4V1MSRIDataUtils,
        "qwen_v1_msri_data": QwenV1MSRIDataUtils,
        "phi4_v1_tent_data": Phi4V1TentDataUtils,
        "qwen_v1_tent_data": QwenV1TentDataUtils,
    }

    @classmethod
    def get_utils(cls, model_name: str) -> BaseModelUtils:
        """Get appropriate utils based on model name"""
        model_lower = model_name.lower()

        for key, utils_class in cls._utils_map.items():
            if key in model_lower:
                return utils_class()

        raise ValueError(f"No utils found for model: {model_name}")

    @classmethod
    def register_utils(cls, model_key: str, utils_class: type[BaseModelUtils]):
        """Register new model utils"""
        cls._utils_map[model_key] = utils_class

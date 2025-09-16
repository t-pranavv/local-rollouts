import re
import json
from typing import List, Dict
from textwrap import dedent


class UnknownToolBlockFormatException(Exception):
    """Exception raised when the tool block format is unknown."""

    support_formats = ["JSON", "MarkdownCode"]

    def __init__(self, message: str = "Unknown tool block format found inside tool tags. Supported formats are: ["):
        message += ", ".join(UnknownToolBlockFormatException.support_formats) + "]."
        message += " Please follow the system instructions."
        super().__init__(message)


class InvalidJsonCodeBlockFormatException(Exception):
    """Exception raised when the JSON code block format is invalid."""

    def __init__(
        self,
        message: str = "Invalid JSON code block format found inside tool tags. Please follow the system instructions.",
    ):
        super().__init__(message)


class MultipleMarkdownCodeBlocksException(Exception):
    """Exception raised when multiple Markdown code blocks are found."""

    def __init__(self, message: str = "Multiple Markdown code blocks found inside tool tags, please provide only one."):
        super().__init__(message)


class EmptyMarkdownCodeBlockException(Exception):
    """Exception raised when an empty Markdown code block is found."""

    def __init__(
        self, message: str = "Empty Markdown code block found inside tool tags, please provide a valid code block."
    ):
        super().__init__(message)


class UnknownToolException(Exception):
    """Exception raised when an unknown tool is encountered."""

    supported_tools = ["CodeInterpreter"]

    def __init__(self, message: str = "Unknown tool found inside tool tags. Supported tools are: ["):
        message += ", ".join(UnknownToolException.supported_tools) + "]."
        message += " Please follow the system instructions."
        super().__init__(message)


def parse_tool_block(tool_call: str) -> List[Dict]:
    block_type = find_block_type(tool_call)
    if block_type == "json":
        return extract_json_codeblock(tool_call)
    elif block_type == "markdown_code":
        return extract_markdown_codeblock(tool_call)
    else:
        return [
            {
                "tool_category": "exception",
                "tool_subcategory": "exception",
                "codeblock": str(UnknownToolBlockFormatException()),
            }
        ]


def find_block_type(response: str) -> str:
    response = response.strip()
    # TODO: support single line `...` format as valid markdown code
    if response.startswith("```") and response.endswith("```"):
        return "markdown_code"
    if response[0] in "{[" and response[-1] in "}]":
        if is_valid_json(response)["is_valid"]:
            return "json"
    return "unknown"


def extract_markdown_codeblock(response: str) -> List[Dict]:
    pattern = re.compile(r"^\s*```\s*([a-zA-Z0-9_+-]+)\s*\n(.*?)\s*```\s*", re.DOTALL | re.MULTILINE)
    markdown_code = pattern.findall(response)
    markdown_code = [
        {
            "tool_category": "code_interpreter",
            "tool_subcategory": dedent(code_language),
            "codeblock": dedent(code_content).strip(),
        }
        for code_language, code_content in markdown_code
        if code_content.strip()
    ]
    if len(markdown_code) == 0:
        return [
            {
                "tool_category": "exception",
                "tool_subcategory": "exception",
                "codeblock": str(EmptyMarkdownCodeBlockException()),
            }
        ]
    elif len(markdown_code) == 1:
        return markdown_code
    else:
        return [
            {
                "tool_category": "exception",
                "tool_subcategory": "exception",
                "codeblock": str(MultipleMarkdownCodeBlocksException()),
            }
        ]


def extract_json_codeblock(response: str) -> List[Dict]:
    is_valid = is_valid_json(response)
    content = is_valid["content"]
    if is_valid["is_valid"]:
        if not isinstance(content, list):
            content = [content]

        codeblocks = []
        for block in content:
            if "name" not in block or "arguments" not in block:
                codeblocks.append(
                    {
                        "tool_category": "exception",
                        "tool_subcategory": "exception",
                        "codeblock": str(
                            InvalidJsonCodeBlockFormatException(
                                "Invalid JSON code block format found inside tool tags: 'name' or 'arguments' missing"
                            )
                        ),
                    }
                )
            else:
                tool_category = get_tool_category(block["name"])
                if tool_category["tool_category"] == "unknown":
                    codeblocks.append(
                        {
                            "tool_category": "exception",
                            "tool_subcategory": "exception",
                            "codeblock": str(UnknownToolException()),
                        }
                    )
                else:
                    # TODO: add json-schema checks for arguments to match the input schema
                    codeblocks.append({**tool_category, "codeblock": block})
        return codeblocks
    else:
        return [{"tool_category": "exception", "tool_subcategory": "exception", "codeblock": content}]


def is_valid_json(response: str) -> Dict:
    try:
        content = json.loads(response)
        return {"is_valid": True, "content": content}
    except json.JSONDecodeError as e:
        msg = e.msg
        snippet = repr(e.doc[e.pos - 20 : e.pos + 20])
        message_lines = ["JSON parsing failed for tool call:"]
        message_lines.append(f"  Message : {msg}")
        message_lines.append(f"  Snippet : {snippet}")
        error_message = "\n".join(message_lines)
        return {"is_valid": False, "content": error_message}


def get_tool_category(tool_name: str) -> Dict:
    # Only python code interpreter is supported for now
    return {"tool_category": "code_interpreter", "tool_subcategory": "python"}

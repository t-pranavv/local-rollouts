import os
import json
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict
from textwrap import dedent

from phigen.utils.prompt_render import JinjaPromptTemplate
from phigen.datagen.shared_data import load_file
from phigen.logging import logger

from .tool_utils import UnknownToolException, UnknownToolBlockFormatException
from .azure_code_interpreter import get_azure_dynamic_sessions_client
from .local_code_interpreter import start_local_ci_client

current_directory = Path(__file__).resolve().parent


class UnsupportedCodeInterpreterToolException(Exception):
    """Exception raised when the Code Interpreter tool is not supported."""

    support_code_interpreters = ["Python"]

    def __init__(self, message: str = "Unsupported Code Interpreter tool found. Supported tools are: ["):
        message += ", ".join(UnsupportedCodeInterpreterToolException.support_code_interpreters) + "]."
        message += " Please follow the system instructions."
        super().__init__(message)


class ToolExecutor:

    if os.environ.get("CODE_INTERPRETER", "LOCAL") == "AZURE":
        CODE_INTERPRETER = get_azure_dynamic_sessions_client(
            pool_base_name=os.environ.get("CODE_INTERPRETER_POOL_BASE_NAME", "python-code-interpreter-pool"),
            num_pools=int(os.environ.get("CODE_INTERPRETER_NUM_POOLS", 10)),
            region=os.environ.get("CODE_INTERPRETER_REGION", "westus2"),
            subscription_id=os.environ.get("CODE_INTERPRETER_SUBSCRIPTION_ID", "d4fe558f-6660-4fe7-99ec-ae4716b5e03f"),
            resource_group=os.environ.get("CODE_INTERPRETER_RESOURCE_GROUP", "reasoning_tools"),
            mi_client_id=os.environ.get("CODE_INTERPRETER_MI_CLIENT_ID", "b32444ac-27e2-4f36-ab71-b664f6876f00"),
        )
    else:
        CODE_INTERPRETER = start_local_ci_client(
            host=os.environ.get("CODE_INTERPRETER_HOST", "localhost"),
            port=int(os.environ.get("CODE_INTERPRETER_PORT", 6989)),
            num_pools=int(os.environ.get("CODE_INTERPRETER_NUM_POOLS", 10)),
            log_level=os.environ.get("CODE_INTERPRETER_LOG_LEVEL", "warning"),
        )

    CLASS_NAME_TO_ENV_FILE = {
        "GorillaFileSystem": str(current_directory / "bfcl_envs/gorilla_file_system.py"),
        "MathAPI": str(current_directory / "bfcl_envs/math_api.py"),
        "MessageAPI": str(current_directory / "bfcl_envs/message_api.py"),
        "TwitterAPI": str(current_directory / "bfcl_envs/posting_api.py"),
        "TicketAPI": str(current_directory / "bfcl_envs/ticket_api.py"),
        "TradingBot": str(current_directory / "bfcl_envs/trading_bot.py"),
        "TravelAPI": str(current_directory / "bfcl_envs/travel_booking.py"),
        "VehicleControlAPI": str(current_directory / "bfcl_envs/vehicle_control.py"),
    }

    CLASS_NAME_TO_ENV_DEPS = {
        "GorillaFileSystem": {"files": [str(current_directory / "bfcl_envs/long_context.py")]},
        "MathAPI": {"files": []},
        "MessageAPI": {"files": []},
        "TwitterAPI": {"files": []},
        "TicketAPI": {"files": []},
        "TradingBot": {"files": [str(current_directory / "bfcl_envs/long_context.py")]},
        "TravelAPI": {"files": [str(current_directory / "bfcl_envs/long_context.py")]},
        "VehicleControlAPI": {"files": [str(current_directory / "bfcl_envs/long_context.py")]},
    }

    INVOLVED_CLASS_TO_FUNC_DOC_PATH = {
        "GorillaFileSystem": str(current_directory / "bfcl_tools/gorilla_file_system.jsonl"),
        "MathAPI": str(current_directory / "bfcl_tools/math_api.jsonl"),
        "MessageAPI": str(current_directory / "bfcl_tools/message_api.jsonl"),
        "TwitterAPI": str(current_directory / "bfcl_tools/posting_api.jsonl"),
        "TicketAPI": str(current_directory / "bfcl_tools/ticket_api.jsonl"),
        "TradingBot": str(current_directory / "bfcl_tools/trading_bot.jsonl"),
        "TravelAPI": str(current_directory / "bfcl_tools/travel_booking.jsonl"),
        "VehicleControlAPI": str(current_directory / "bfcl_tools/vehicle_control.jsonl"),
    }

    STATELESS_CLASSES = ["MathAPI"]

    FUNCTION_CALLING_PROMPT = JinjaPromptTemplate(
        template_path=str(current_directory.parent / "prompts/function_calling.jinja")
    )

    @classmethod
    async def execute_tool_call_async(cls, tool_call: List[Dict], **kwargs) -> List[Dict]:
        tool_response = []
        event_loop = asyncio.get_event_loop()
        call_handlers = []
        tmp_files = []
        tool_call_timeouts = kwargs.get("tool_call_timeouts", {})
        for i, call in enumerate(tool_call):
            if call["tool_category"] == "exception":
                tool_response.append({"index": i, "exception": call["codeblock"], "success": False})
            else:
                try:
                    tool_category, tool_subcategory, codeblock = (
                        call["tool_category"],
                        call["tool_subcategory"],
                        call["codeblock"],
                    )
                    if tool_category == "code_interpreter":
                        if tool_subcategory == "python":
                            session_id = kwargs.get("session_id", None)
                            timeout = tool_call_timeouts.get("code_interpreter", {}).get("python", 60)
                            if isinstance(codeblock, dict):
                                involved_classes = kwargs.get("involved_classes", [])
                                initial_config = kwargs.get("initial_config", {})
                                long_context = kwargs.get("long_context", False)

                                env_files_to_upload, deps_files_to_upload, packages_to_install = [], [], []
                                for class_name in involved_classes:
                                    if class_name in cls.CLASS_NAME_TO_ENV_FILE:
                                        env_files_to_upload.append(cls.CLASS_NAME_TO_ENV_FILE[class_name])
                                        if class_name in cls.CLASS_NAME_TO_ENV_DEPS:
                                            deps_files_to_upload.append(
                                                (class_name, cls.CLASS_NAME_TO_ENV_DEPS[class_name]["files"])
                                            )
                                            packages_to_install.append(
                                                (class_name, cls.CLASS_NAME_TO_ENV_DEPS[class_name].get("packages", []))
                                            )

                                files_to_upload = [
                                    {"local_file": file, "remote_file": os.path.basename(file)}
                                    for file in env_files_to_upload
                                ]

                                for class_name, files in deps_files_to_upload:
                                    files_to_upload.extend(
                                        {"local_file": file, "remote_file": os.path.basename(file)} for file in files
                                    )

                                for class_name, packages in packages_to_install:
                                    if not packages:
                                        continue
                                    f = tempfile.NamedTemporaryFile(delete=False, delete_on_close=True, mode="w")
                                    tmp_files.append(f)
                                    f.write("\n".join(packages))
                                    f.flush()
                                    files_to_upload.append(
                                        {"local_file": f.name, "remote_file": f"{class_name}_requirements.txt"}
                                    )

                                f = tempfile.NamedTemporaryFile(delete=False, delete_on_close=True, mode="w")
                                tmp_files.append(f)
                                files_to_upload.append({"local_file": f.name, "remote_file": "__init__.py"})

                                # dedup files_to_upload based on local_file paths
                                files_to_upload = list({f["local_file"]: f for f in files_to_upload}.values())

                                code = cls.FUNCTION_CALLING_PROMPT.create(
                                    {
                                        "code_block": codeblock,
                                        "class_name_to_env_file": cls.CLASS_NAME_TO_ENV_FILE,
                                        "stateless_classes": cls.STATELESS_CLASSES,
                                        "initial_config": initial_config,
                                        "involved_classes": involved_classes,
                                        "long_context": long_context,
                                    }
                                )
                                code = dedent(code)
                                call_handlers.append(
                                    event_loop.create_task(
                                        cls.CODE_INTERPRETER.execute_code(
                                            (code, i, session_id, files_to_upload, None), timeout=timeout
                                        )
                                    )
                                )
                            elif isinstance(codeblock, str):
                                call_handlers.append(
                                    event_loop.create_task(
                                        cls.CODE_INTERPRETER.execute_code(
                                            (codeblock, i, session_id, None, None), timeout=timeout
                                        )
                                    )
                                )
                            else:
                                tool_response.append(
                                    {"index": i, "exception": str(UnknownToolBlockFormatException()), "success": False}
                                )
                        else:
                            tool_response.append(
                                {
                                    "index": i,
                                    "exception": str(UnsupportedCodeInterpreterToolException()),
                                    "success": False,
                                }
                            )
                    else:
                        tool_response.append({"index": i, "exception": str(UnknownToolException()), "success": False})
                except Exception as e:
                    tool_response.append({"index": i, "exception": str(e), "success": False})
        call_responses = await asyncio.gather(*call_handlers)
        tool_response.extend(call_responses)
        [tmp_file.close() for tmp_file in tmp_files]
        return sorted(tool_response, key=lambda x: x["index"])

    @classmethod
    def construct_tools_from_involved_classes(cls, involved_classes: List[str]) -> str:
        try:
            tools = []
            for class_name in involved_classes:
                func_doc = load_file(cls.INVOLVED_CLASS_TO_FUNC_DOC_PATH[class_name], show_progress=False)
                for func in func_doc:
                    func["description"] = func["description"].split("Tool description: ", 1)[1]
                func_doc = [json.dumps(func) for func in func_doc]
                tools.extend(func_doc)
            return "\n".join(tools)
        except Exception as e:
            logger().error(f"Error constructing tools from involved classes: {e}")
            return ""

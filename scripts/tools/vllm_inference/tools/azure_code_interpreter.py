import time
import json
import uuid
import random
import hashlib
import asyncio
from typing import Dict, Tuple, Any, List, Optional
from azure.identity import ManagedIdentityCredential
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool

from phigen.logging import logger


def token_provider(mi_client_id: str):
    """Token provider for Azure Dynamic Sessions using Managed Identity"""
    resource_scope = "https://dynamicsessions.io/.default"
    credential = ManagedIdentityCredential(client_id=mi_client_id)
    token = credential.get_token(resource_scope).token
    return token


class AzureDynamicSessionsClient:
    def __init__(
        self, session_pool_urls: List[str], token_provider_func, max_concurrent: int = 100, max_retries: int = 3
    ):
        self.session_pool_urls = [url.rstrip("/") for url in session_pool_urls]
        self.token_provider_func = token_provider_func
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries

    def get_session_pool_url_for_session(self, session_id: str) -> str:
        """Get consistent session pool URL for a given session ID using hash-based selection"""
        # Use hash of session_id to consistently map to the same pool
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        pool_index = hash_value % len(self.session_pool_urls)
        return self.session_pool_urls[pool_index]

    def get_session_pool_url_for_index(self, index: int) -> str:
        """Round-robin selection of session pool (kept for backward compatibility)"""
        return self.session_pool_urls[index % len(self.session_pool_urls)]

    def create_repl_tool(self, session_pool_url: str, session_id: str) -> SessionsPythonREPLTool:
        """Create a new REPL tool with the specified session ID"""
        return SessionsPythonREPLTool(
            name="Python Code Interpreter",
            pool_management_endpoint=session_pool_url,
            session_id=session_id,
            access_token_provider=self.token_provider_func,
        )

    def _handle_file_uploads(
        self, repl_tool: SessionsPythonREPLTool, upload_files: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Handle file upload operations
        Required: local_file and remote_file must be specified in upload_files
        """
        file_results = []
        if upload_files:
            for file_spec in upload_files:
                try:
                    local_file = file_spec.get("local_file")
                    remote_file = file_spec.get("remote_file")

                    if not remote_file:
                        raise ValueError("remote_file must be specified for upload")

                    if isinstance(local_file, str):
                        # local_file is a file path
                        metadata = repl_tool.upload_file(local_file_path=local_file, remote_file_path=remote_file)
                    elif hasattr(local_file, "read"):
                        # local_file is BinaryIO data
                        metadata = repl_tool.upload_file(data=local_file, remote_file_path=remote_file)
                    else:
                        raise ValueError("local_file must be a file path (str) or BinaryIO data")

                    file_results.append(
                        {
                            "local_file": str(local_file) if isinstance(local_file, str) else "<BinaryIO>",
                            "remote_file": remote_file,
                            "status": "success",
                            "metadata": metadata.__dict__ if hasattr(metadata, "__dict__") else str(metadata),
                        }
                    )
                except Exception as e:
                    file_results.append(
                        {
                            "local_file": str(file_spec.get("local_file", "<unknown>")),
                            "remote_file": file_spec.get("remote_file", "<unknown>"),
                            "status": "error",
                            "error": str(e),
                        }
                    )
        return file_results

    def _handle_file_downloads(
        self, repl_tool: SessionsPythonREPLTool, download_files: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Handle file download operations
        Required: remote_file must be specified, local_file is optional.
        """
        file_results = []
        if download_files:
            for file_spec in download_files:
                try:
                    local_file = file_spec.get("local_file")
                    remote_file = file_spec.get("remote_file")

                    if not remote_file:
                        raise ValueError("remote_file must be specified for download")

                    if local_file:
                        # Download to local file path
                        data = repl_tool.download_file(remote_file_path=remote_file, local_file_path=local_file)
                        file_results.append(
                            {
                                "local_file": local_file,
                                "remote_file": remote_file,
                                "status": "success",
                                "data": None,  # File saved to disk
                            }
                        )
                    else:
                        # Return BinaryIO data
                        data = repl_tool.download_file(remote_file_path=remote_file)
                        file_results.append(
                            {
                                "local_file": None,
                                "remote_file": remote_file,
                                "status": "success",
                                "data": data,  # BinaryIO data
                            }
                        )
                except Exception as e:
                    file_results.append(
                        {
                            "local_file": file_spec.get("local_file"),
                            "remote_file": file_spec.get("remote_file", "<unknown>"),
                            "status": "error",
                            "error": str(e),
                            "data": None,
                        }
                    )
        return file_results

    async def execute_code_single(
        self,
        code: str,
        session_pool_url: str,
        session_identifier: str,
        index: int,
        upload_files: Optional[List[Dict[str, Any]]] = None,
        download_files: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute code using LangChain's SessionsPythonREPLTool (single attempt)"""
        start_time = time.time()

        try:
            # Create a new REPL tool with the specific session ID
            repl_tool = self.create_repl_tool(session_pool_url, session_identifier)

            # Handle file operations
            loop = asyncio.get_event_loop()
            file_results = {}
            # Running in executor to avoid blocking the event loop
            file_results["uploads"] = await loop.run_in_executor(
                None, self._handle_file_uploads, repl_tool, upload_files
            )

            result = await repl_tool.arun(code)
            result_dict = json.loads(result) if isinstance(result, str) else result
            # Running in executor to avoid blocking the event loop
            file_results["downloads"] = await loop.run_in_executor(
                None, self._handle_file_downloads, repl_tool, download_files
            )

            # Parse the result - SessionsPythonREPLTool returns a string
            success = True
            stdout = ""
            stderr = ""
            if result_dict["stdout"].strip():
                stdout = result_dict["stdout"].strip()
            if result_dict["stderr"].strip():
                success = all(("Error" not in result, "Exception" not in result, "Traceback" not in result))
                stderr = result_dict["stderr"].strip()

            return {
                "index": index,
                "code": code,
                "session_identifier": session_identifier,
                "execution_time": round(time.time() - start_time, 2),
                "success": success,
                "raw_result": result,
                "stdout": stdout,
                "stderr": stderr,
                "session_pool": session_pool_url,
                "file_operations": file_results,
            }
        except Exception as e:
            raise e

    async def execute_code_single_with_retry(
        self,
        code: str,
        index: int,
        session_identifier: str | None = None,
        upload_files: Optional[List[Dict[str, Any]]] = None,
        download_files: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute code with retry logic for 409 conflicts"""
        last_error = None

        # Determine the primary session pool based on session_identifier
        has_session_identifier = session_identifier is not None
        if has_session_identifier:
            primary_session_pool_url = self.get_session_pool_url_for_session(session_identifier)
        else:
            # Fallback to random session pool if no session identifier is provided
            session_identifier = str(uuid.uuid4())
            primary_session_pool_url = self.get_session_pool_url_for_session(session_identifier)

        # Use the primary session pool for first attempt
        session_pool_url = primary_session_pool_url
        for attempt in range(self.max_retries):
            try:
                if attempt > 0 and not has_session_identifier:
                    # Generate new session identifier for subsequent attempts if not user provided
                    session_identifier = str(uuid.uuid4())
                    session_pool_url = self.get_session_pool_url_for_session(session_identifier)

                result = await self.execute_code_single(
                    code, session_pool_url, session_identifier, index, upload_files, download_files
                )
                return result
            except Exception as e:
                error_str = str(e)
                last_error = e

                # Backoff for 409 conflict error
                if "409" in error_str and "Conflict" in error_str:
                    logger().debug(
                        f"Attempt {attempt + 1}: 409 Conflict for index {index}, retrying with different pool..."
                    )

                    # Exponential backoff with jitter
                    wait_time = (5**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break

        # If all retries failed, return error result
        return {
            "index": index,
            "code": code,
            "session_identifier": session_identifier,
            "execution_time": 0,
            "success": False,
            "stdout": "",
            "stderr": f"Max retries exceeded. Last error: {str(last_error)}",
            "session_pool": session_pool_url,
            "file_operations": {"uploads": [], "downloads": []},
        }

    async def execute_code(self, code_block: Tuple, timeout: int = 10) -> Dict:
        """Execute a single code snippet with retry logic"""
        code, index, session_identifier, upload_files, download_files = code_block
        try:
            return await asyncio.wait_for(
                self.execute_code_single_with_retry(
                    code,
                    index,
                    session_identifier=session_identifier,
                    upload_files=upload_files,
                    download_files=download_files,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return {
                "index": index,
                "exception": "Code Execution Timed Out, possible infinite loops in the code",
                "success": False,
            }

    async def execute_code_batch(self, code_batch: List[Tuple], timeout: int = 10) -> List[Dict]:
        """Execute a batch of code snippets concurrently with retry logic"""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def execute_with_semaphore(code, index, session_identifier, upload_files, download_files):
            try:
                async with semaphore:
                    return await asyncio.wait_for(
                        self.execute_code_single_with_retry(
                            code,
                            index,
                            session_identifier=session_identifier,
                            upload_files=upload_files,
                            download_files=download_files,
                        ),
                        timeout=timeout,
                    )
            except asyncio.TimeoutError:
                return {
                    "index": index,
                    "exception": "Code Execution Timed Out, possible infinite loops in the code",
                    "success": False,
                }
            except Exception as e:
                return {"index": index, "exception": str(e), "success": False}

        tasks = [execute_with_semaphore(*code) for code in code_batch]
        return await asyncio.gather(*tasks)


def get_azure_dynamic_sessions_client(
    pool_base_name: str,
    num_pools: int,
    region: str,
    subscription_id: str,
    resource_group: str,
    mi_client_id: str,
    max_concurrent: int = 600,
) -> AzureDynamicSessionsClient:
    """Get an instance of AzureDynamicSessionsClient with configured session pools"""
    session_pool_urls = [
        f"https://{region}.dynamicsessions.io/subscriptions/{subscription_id}/resourceGroups/{resource_group}/sessionPools/{pool_base_name}-{i}"
        for i in range(1, num_pools + 1)
    ]
    token_provider_func = lambda: token_provider(mi_client_id)
    return AzureDynamicSessionsClient(session_pool_urls, token_provider_func, max_concurrent)

import os
import signal
import sys
import json
import time
import traceback
import threading
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from multiprocessing import Process, Queue

import streamlit as st

VLLM_INF_DIR = Path(__file__).resolve().parent.parent
if str(VLLM_INF_DIR) not in sys.path:
    sys.path.insert(0, str(VLLM_INF_DIR))

from run_inference_on_vllm_server import VLLMLocalInferenceFlow, VLLMLocalResponseAgent, VLLMLocalToolResponseAgent
from model_utils import ModelUtilsFactory, BaseModelUtils
from phigen.client.models.model_types import ModelKind, ModelMetadata
from phigen.local_inference.vllm_local_client import _setup_vllm_local_client
from phigen.datagen.local_dataset import local_dataset, temp_dataset

__all__ = ["VLLMLocalResponseAgent", "VLLMLocalToolResponseAgent"]


def set_vllm_env(
    base_ip: str, base_port: int, api_key: str, num_servers: int, heartbeat: int = 10, max_wait: int = 3600
) -> None:
    os.environ["VLLM_LOCAL_BASE_URL"] = f"http://{base_ip}:{{port}}/v1"
    os.environ["VLLM_LOCAL_BASE_PORT"] = str(base_port)
    os.environ["VLLM_LOCAL_API_KEY"] = str(api_key)
    os.environ["VLLM_LOCAL_NUM_INSTANCES"] = str(num_servers)
    os.environ["VLLM_LOCAL_SERVER_HEARTBEAT"] = str(heartbeat)
    os.environ["VLLM_LOCAL_MAX_WAIT_TIME_FOR_HEARTBEAT"] = str(max_wait)


def write_pasted_prompt_jsonl(prompt_text: str, dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = dest_dir / "pasted_prompt.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in load_items(prompt_text):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(jsonl_path)


def save_uploaded_file(upload, dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / upload.name
    with dest_path.open("wb") as f:
        f.write(upload.read())
    return str(dest_path)


def run_flow(args: argparse.Namespace):
    set_vllm_env(
        base_ip=args.base_ip,
        base_port=args.base_port,
        api_key=args.api_key,
        num_servers=args.num_servers,
        heartbeat=args.server_heartbeat,
        max_wait=args.max_wait_time_for_vllm_heartbeat,
    )

    # Initialize the local vLLM client target
    _setup_vllm_local_client(
        str(args.served_model_name),
        getattr(ModelKind, args.api_type.upper()),
        ModelMetadata(user_token="user", asst_token="assistant", organization="openai"),
    )

    # Prepare dataset
    filename = Path(args.prompts_file).stem
    if args.phigen_dataset == "local":
        dataset = local_dataset(f"{filename}_output_dataset", dir=args.output_dir)
    else:
        dataset = temp_dataset()

    # Track files before
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {p for p in out_dir.glob("**/*") if p.is_file()}

    # Run the Flow
    VLLMLocalInferenceFlow.run_flow(
        dataset=dataset,
        data_kwargs={"args": args, "remove_cols": ["idx"]},
        run_kwargs={"multiprocess": False, "ordered": True, "num_worker_procs": 1},
    )

    # Diff files after
    after = {p for p in out_dir.glob("**/*") if p.is_file()}
    new_files = sorted(list(after - before))

    return {"output_dir": str(out_dir), "new_files": [str(p) for p in new_files]}


def _mp_run_flow(queue: "Queue", args_dict: Dict[str, Any]):  # pragma: no cover - background worker
    """
    Worker executed in a separate process.
    """
    try:
        args = argparse.Namespace(**args_dict)
        res = run_flow(args)
        queue.put({"status": "ok", "result": res, "args": args_dict})
    except Exception:
        queue.put({"status": "error", "error": traceback.format_exc(), "args": args_dict})


def load_items(pasted_text: str) -> List[Dict[str, Any]]:
    obj = json.loads(pasted_text)
    if isinstance(obj, list):
        items = [x for x in obj if isinstance(x, dict)]
    elif isinstance(obj, dict):
        items = [obj]
    else:
        st.warning("Top-level JSON should be an object or array of objects. Falling back to JSONL parse.")
        items = []
    return items


def _normalize_weird_fenced_tags(text: str, utils: BaseModelUtils) -> str:
    """
    Normalize cases where tags accidentally appear inside triple backticks.

    Example: ```<tool_call>```python  -> <tool_call>\n```python
    """
    import re

    parts: List[Tuple[str, str]] = []
    parts.append(utils.get_tool_call_tokens())
    parts.append(utils.get_tool_response_tokens())
    try:
        parts.append(utils.get_think_tokens())
    except Exception:
        pass
    try:
        parts.append(utils.get_final_answer_tokens())
    except Exception:
        pass
    if hasattr(utils, "usermessage_tokens"):
        try:  # pragma: no cover - optional
            parts.append(utils.usermessage_tokens())  # type: ignore[attr-defined]
        except Exception:
            pass

    out = text
    for s_tok, e_tok in parts:
        out = re.sub(rf"```\s*{re.escape(s_tok)}\s*```([A-Za-z0-9_+\-]*)", rf"{s_tok}\n```\1", out)
        out = re.sub(rf"```\s*{re.escape(e_tok)}", rf"```\n{e_tok}", out)
    return out


def linearize_assistant_segments(raw: str, utils: BaseModelUtils) -> List[Dict[str, Any]]:
    text = raw or ""
    text = _normalize_weird_fenced_tags(text, utils)
    try:
        think_start, think_end = utils.get_think_tokens()
    except Exception:
        think_start, think_end = "", ""
    tool_start, tool_end = utils.get_tool_call_tokens()
    result_start, result_end = utils.get_tool_response_tokens()
    try:
        fa_start, fa_end = utils.get_final_answer_tokens()
    except Exception:
        fa_start, fa_end = "", ""
    if hasattr(utils, "usermessage_tokens"):
        try:  # pragma: no cover
            user_start, user_end = utils.usermessage_tokens()  # type: ignore[attr-defined]
        except Exception:
            user_start, user_end = "", ""
    else:
        user_start, user_end = "", ""

    segments: List[Dict[str, Any]] = []
    fa_s_idx = text.rfind(fa_start) if fa_start else -1
    fa_e_idx = text.find(fa_end, fa_s_idx + len(fa_start)) if fa_s_idx != -1 and fa_end else -1

    def add_segment(seg_type: str, start_idx: int, end_idx: int):
        if start_idx < 0 or end_idx <= start_idx:
            return
        content = text[start_idx:end_idx]
        seg: Dict[str, Any] = {"type": seg_type, "start": start_idx, "end": end_idx, "content": content}
        if seg_type == "tool_call":
            try:
                wrapped = f"{tool_start}{content}{tool_end}"
                if hasattr(utils, "parse_tool_call"):
                    seg["parsed_tool"] = utils.parse_tool_call(wrapped)  # type: ignore[attr-defined]
                else:
                    raise AttributeError("parse_tool_call not available on utils")
            except Exception as e:
                seg["parsed_tool"] = [
                    {"tool_category": "exception", "tool_subcategory": "exception", "codeblock": str(e)}
                ]
        segments.append(seg)

    idx = 0
    text_len = len(text)
    while think_start and think_end and idx < text_len:
        ts = text.find(think_start, idx)
        if ts == -1:
            break
        te = text.find(think_end, ts + len(think_start))
        if te == -1:
            break
        add_segment("think", ts + len(think_start), te)
        idx = te + len(think_end)

        next_think_ts = text.find(think_start, idx)
        search_end = next_think_ts if next_think_ts != -1 else text_len
        tcs = text.find(tool_start, idx, search_end) if tool_start else -1
        if tcs != -1:
            tce = text.find(tool_end, tcs + len(tool_start), search_end)
            if tce != -1:
                add_segment("tool_call", tcs + len(tool_start), tce)
                after_tool = tce + len(tool_end)
                trs = text.find(result_start, after_tool, search_end) if result_start else -1
                if trs != -1:
                    tre = text.find(result_end, trs + len(result_start), search_end)
                    if tre != -1:
                        add_segment("tool_result", trs + len(result_start), tre)
                        idx = tre + len(result_end)
                        continue
                idx = after_tool
                continue
        idx = search_end if search_end > idx else idx + 1

    # user segments
    if user_start and user_end:
        u_idx = 0
        while u_idx < text_len:
            us = text.find(user_start, u_idx)
            if us == -1:
                break
            ue = text.find(user_end, us + len(user_start))
            if ue == -1:
                break
            add_segment("user", us + len(user_start), ue)
            u_idx = ue + len(user_end)

    # final answer
    if fa_s_idx != -1 and fa_e_idx != -1:
        add_segment("final_answer", fa_s_idx + len(fa_start), fa_e_idx)
    else:
        # fallback treat trailing text
        last_end = 0
        if segments:
            last_end = max(s["end"] for s in segments)
        if last_end < len(text):
            trailing = text[last_end:]
            if trailing.strip():
                add_segment("final_answer", last_end, len(text))

    segments.sort(key=lambda d: d["start"])
    return segments


def render_assistant_timeline(raw: str, utils: BaseModelUtils, expand_mode: str = "auto"):
    segs = linearize_assistant_segments(raw, utils)
    if not segs:
        st.code(raw or "", language="text")
        return
    # Per-type running counters (nice labels) and expansion logic
    counters: Dict[str, int] = {}
    for seg in segs:
        t = seg["type"]
        counters[t] = counters.get(t, 0) + 1
        idx = counters[t]
        base_label = t.replace("_", " ")
        label = (
            f"{base_label.title()} #{idx}"
            if t != "final_answer"
            else ("Final Answer" if idx == 1 else f"Final Answer #{idx}")
        )
        if expand_mode == "all":
            expanded_flag = True
        elif expand_mode == "none":
            expanded_flag = False
        else:
            expanded_flag = t == "final_answer"
        with st.expander(label, expanded=expanded_flag):
            if seg["type"] == "tool_call":
                st.markdown("**Tool Call**")
                parsed = seg.get("parsed_tool") or []
                for call in parsed:
                    st.caption(f"{call.get('tool_category','?')} / {call.get('tool_subcategory','?')}")
                    codeblock = call.get("codeblock")
                    if isinstance(codeblock, (dict, list)):
                        st.json(codeblock)
                    elif isinstance(codeblock, str):
                        try:
                            st.json(json.loads(codeblock))
                        except Exception:
                            st.code(codeblock, language="python")
                    else:
                        st.write(codeblock)
            else:
                st.code(seg["content"], language="text")


def render_assistant_view(raw: str, utils: BaseModelUtils, show_raw: bool = False):
    """Inline generation breakdown with global expand/collapse buttons (no raw text panel)."""
    scope_id = str(abs(hash(raw or "")))
    expand_mode = st.session_state.get(f"gb_mode_{scope_id}", "auto")
    # Controls row
    ctrl_col, btn_expand, btn_collapse = st.columns([10, 1, 1])
    with btn_expand:
        if st.button("🔽", help="Expand all segments", key=f"expand_all_{scope_id}"):
            expand_mode = "all"
    with btn_collapse:
        if st.button("🔼", help="Collapse all segments", key=f"collapse_all_{scope_id}"):
            expand_mode = "none"
    st.session_state[f"gb_mode_{scope_id}"] = expand_mode
    render_assistant_timeline(raw, utils, expand_mode=expand_mode)


def extract_assistant_text(record: Dict[str, Any]) -> List[str]:
    """
    Return list of assistant generation strings found in a record.
    """
    gens: List[str] = []
    # completions list
    if isinstance(record.get("completions"), list):
        gens.extend([str(c) for c in record["completions"] if isinstance(c, (str, bytes))])
    # messages conversation
    if isinstance(record.get("messages"), list):
        for m in record["messages"]:
            if isinstance(m, dict) and m.get("role") == "assistant":
                content = m.get("content")
                if isinstance(content, str):
                    gens.append(content)
    # direct response fields
    for key in ["final_answer", "assistant_response", "response", "output"]:
        val = record.get(key)
        if isinstance(val, str):
            gens.append(val)
    # dedupe preserving order
    seen = set()
    out = []
    for g in gens:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def app():
    st.set_page_config(page_title="vLLM Tool-Use Inference", layout="wide")
    st.title("vLLM Interactive Tool-Use Inference")
    st.caption("Interactive UI that runs the same flow as run_inference_on_vllm_server.py")

    base_ip = os.environ.get("VLLM_DP_MASTER_IP", "localhost")
    base_port = int(os.environ.get("PORT", 9000))
    api_key = os.environ.get("API_KEY", "key")
    num_servers = 1
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "vllm-local-phi-4")
    model_utils_name = os.environ.get("MODEL_UTILS", "phi-think")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "microsoft/Phi-4-reasoning")
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", 32768))
    system_message = os.environ.get("SYSMSG", "None")
    output_dir = os.environ.get("EXPERIMENT_OUTPUT_DIR", f"outputs/streamlit")
    output_dir = os.path.join(output_dir, f"{int(time.time())}")

    # Initialize session state for background inference
    if "inference_proc" not in st.session_state:
        st.session_state.inference_proc = None  # type: ignore
    if "inference_queue" not in st.session_state:
        st.session_state.inference_queue = None  # type: ignore
    if "inference_result" not in st.session_state:
        st.session_state.inference_result = None  # type: ignore
    if "inference_error" not in st.session_state:
        st.session_state.inference_error = None  # type: ignore
    if "inference_args" not in st.session_state:
        st.session_state.inference_args = None  # type: ignore
    if "last_poll" not in st.session_state:
        st.session_state.last_poll = 0.0

    running = st.session_state.inference_proc is not None and st.session_state.inference_proc.is_alive()

    st.markdown(
        """
        <style>
        button[data-baseweb="button"][kind="primary"] {
            background-color: #d32f2f !important;
            border: 1px solid #b71c1c !important;
            color: #ffffff !important;
        }
        button[data-baseweb="button"][kind="primary"]:hover {
            background-color: #b71c1c !important;
            border-color: #8e0000 !important;
        }
        button[data-baseweb="button"][kind="primary"][disabled] {
            background-color: #f5b7b1 !important;
            border-color: #e6aaaa !important;
            color: #555 !important;
            opacity: 0.85 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Run")
        uploaded = st.file_uploader("Upload JSON or JSONL", type=["json", "jsonl"])
        text = st.text_area(
            "Or Paste JSON here",
            value="",
            height=180,
            placeholder="Paste a single JSON object, an array of objects",
        )
        num_samples_to_generate = st.number_input("num_samples_to_generate", min_value=1, max_value=32, value=1, step=1)
        prompt_field = st.text_input("Prompt field (for JSONL)", value="prompt")
        output_dir = st.text_input("Output directory", value=output_dir)
        run_button = st.button(
            "Run Inference",
            disabled=running,
            help="Disabled while an inference run is active",
            type="primary",
        )
        # Allow user to cancel an active inference run
        if running:
            if st.button("Cancel Inference", type="secondary", help="Terminate the running background process"):
                proc = st.session_state.get("inference_proc")
                try:
                    if proc is not None and hasattr(proc, "is_alive") and proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=0.5)
                except Exception:
                    pass
                # Mark as cancelled
                st.session_state.inference_error = "Cancelled by user"
                st.session_state.inference_result = None
                st.session_state.inference_proc = None
                st.session_state.inference_queue = None
                st.rerun()

        st.header("Server")
        base_ip = st.text_input("Base IP", value=base_ip)
        base_port = st.number_input("Base Port", min_value=8001, value=base_port, step=1)
        api_key = st.text_input("API Key", value=api_key)
        num_servers = st.number_input("Number of servers", min_value=1, value=num_servers, step=1)
        served_model_name = st.text_input("Served model name", value=served_model_name)

        st.header("Generation")
        phigen_dataset = st.selectbox("PhiGen Dataset Type", options=["temp", "local"], index=0)
        tokenizer_path = st.text_input("Tokenizer path", value=tokenizer_path)
        model_utils_keys = list(getattr(ModelUtilsFactory, "_utils_map", {}).keys())
        model_utils_name = st.selectbox(
            "Model utils name",
            options=model_utils_keys,
            index=model_utils_keys.index(model_utils_name) if model_utils_name in model_utils_keys else 0,
        )
        max_model_seq_len = st.number_input("Max model seq len", min_value=16384, value=max_model_len, step=8192)
        max_tokens = st.text_input("Max tokens", value="None")
        max_tokens = None if max_tokens.strip().lower() == "none" else int(max_tokens)
        system_message = st.text_input("System message", value=system_message)
        system_message = None if system_message.strip().lower() == "none" else system_message
        temperature = st.number_input("temperature", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
        top_p = st.number_input("top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        top_k = st.number_input("top_k", min_value=-1, max_value=2048, value=50, step=1)
        thinking_model = st.checkbox("Thinking Model", value=True)
        skip_special_tokens = st.checkbox("skip_special_tokens", value=False)

        st.header("Agent")
        api_type = st.selectbox("API Type", options=["completion", "chat"], index=0)
        agent_cls = st.selectbox(
            "Agent class",
            options=["VLLMLocalToolResponseAgent", "VLLMLocalResponseAgent"],
            index=0,
        )
        max_tool_call_steps = st.number_input("Max tool call steps", min_value=1, value=5, step=1)
        tool_call_timeouts = st.text_area(
            "Tool call timeouts",
            value="""{"code_interpreter": {"python": 200}}""",
            height=80,
            help="Timeouts per tool in seconds.",
        )
        num_worker_procs = st.number_input("Number of worker processes", min_value=1, value=32, step=1)
        generate_multi_each_step = st.checkbox("Generate multiple responses each step", value=False)

        st.divider()
        st.subheader("Server Control")
        confirm_shutdown = st.checkbox(
            "Confirm shutdown", value=False, help="Enable to allow the shutdown button below."
        )
        if st.button("🛑 Shutdown Streamlit Server", type="secondary", disabled=not confirm_shutdown):
            st.warning("Shutting down server…")
            # Mark a shutdown flag so we don't restart background work
            st.session_state["__shutdown_requested__"] = True

            def _graceful_cleanup():
                # Stop any running inference subprocess
                try:
                    proc = st.session_state.get("inference_proc")
                    if proc is not None and hasattr(proc, "is_alive") and proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=1)
                except Exception:
                    pass
                # Give Streamlit a brief moment to send final UI update
                time.sleep(0.2)
                # Try graceful exit first
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(0.4)
                except Exception:
                    pass
                # If still alive, force exit (Streamlit sometimes traps SIGTERM on some platforms)
                try:
                    os._exit(0)  # noqa: PLR1722
                except Exception:
                    pass

            try:
                threading.Thread(target=_graceful_cleanup, daemon=True).start()
            except Exception as e:
                st.error(f"Failed to initiate shutdown: {e}")
            # Prevent any further script sections from scheduling new work
            st.stop()

    @st.fragment(run_every=10)
    def poll():
        nonlocal running, output_dir
        if run_button and not running:
            try:
                generation_config = {
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "skip_special_tokens": bool(skip_special_tokens),
                }
            except Exception as e:
                st.error("Invalid generation config or tool timeouts JSON.")
                st.exception(e)
                st.stop()

            output_dir = Path(output_dir)
            if uploaded:
                prompts_file = save_uploaded_file(uploaded, output_dir)
            elif text and text.strip():
                prompts_file = write_pasted_prompt_jsonl(text.strip(), output_dir)
            else:
                st.warning("Please provide input via file upload or text input.")
                st.stop()

            # Build args for the Flow
            args = argparse.Namespace()
            args.prompts_file = prompts_file
            args.base_ip = base_ip
            args.base_port = int(base_port)
            args.api_key = api_key
            args.num_servers = int(num_servers)
            args.server_heartbeat = 10
            args.max_wait_time_for_vllm_heartbeat = 3600
            args.served_model_name = served_model_name
            args.phigen_dataset = phigen_dataset
            args.tokenizer_path = tokenizer_path
            args.model_utils_name = model_utils_name
            args.max_model_seq_len = max_model_seq_len
            args.max_tokens = max_tokens
            args.system_message = system_message
            args.thinking_model = thinking_model
            args.skip_special_tokens = skip_special_tokens
            args.api_type = api_type
            args.agent_cls = globals()[agent_cls]
            args.max_tool_call_steps = max_tool_call_steps
            args.tool_call_timeouts = tool_call_timeouts
            args.num_worker_procs = int(num_worker_procs)
            args.generate_multi_each_step = generate_multi_each_step
            args.generation_config = generation_config
            args.num_samples_to_generate = int(num_samples_to_generate)
            args.prompt_field = prompt_field
            args.output_dir = output_dir

            # Launch background process
            args_dict = vars(args)
            q: Queue = Queue()
            p = Process(target=_mp_run_flow, args=(q, args_dict), daemon=True)
            p.start()
            st.session_state.inference_proc = p
            st.session_state.inference_queue = q
            st.session_state.inference_result = None
            st.session_state.inference_error = None
            st.session_state.inference_args = args_dict
            st.rerun()

        if running:
            # attempt non-blocking queue read
            q = st.session_state.inference_queue  # type: ignore[assignment]
            if q is not None:
                try:  # pragma: no cover - UI interaction
                    msg = q.get_nowait()
                except Exception:
                    msg = None
                if msg:
                    if msg.get("status") == "ok":
                        st.session_state.inference_result = msg.get("result")
                    else:
                        st.session_state.inference_error = msg.get("error")
                    proc = st.session_state.inference_proc
                    try:
                        if proc is not None:
                            proc.join(timeout=0.1)
                    except Exception:  # pragma: no cover
                        pass
                    st.session_state.inference_proc = None
                    st.rerun()
                else:
                    st.info("Inference running ...")

        # Render results if available
        if st.session_state.inference_error:
            st.error("Inference flow failed ...")
            st.code(st.session_state.inference_error, language="text")
        if st.session_state.inference_result:
            result = st.session_state.inference_result
            st.success("Inference complete")
            st.write(f"Output directory: {result['output_dir']}")
            new_files = result["new_files"]
            if not new_files:
                st.info("No new files detected. Check the output directory for dataset artifacts.")
                return

            st.subheader("Download Outputs")
            for f in new_files:
                st.write(f"- {os.path.basename(f)}")
            for f in new_files:
                try:
                    with open(f, "rb") as fh:
                        st.download_button(label=f"Download {Path(f).name}", data=fh, file_name=Path(f).name)
                except Exception:
                    pass

            # Attempt to load records from first JSONL file(s)
            jsonl_files = [p for p in new_files if p.endswith(".jsonl")]
            if not jsonl_files:
                st.info("No JSONL outputs to display inline.")
                return

            # Concatenate limited lines for performance
            records: List[Dict[str, Any]] = []
            max_lines = 200  # safety cap
            for jf in jsonl_files:
                try:
                    with open(jf, "r", encoding="utf-8") as fh:
                        for i, line in enumerate(fh):
                            if i >= max_lines:
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                if isinstance(obj, dict):
                                    records.append(obj)
                            except Exception:
                                continue
                except Exception:
                    continue
            if not records:
                st.info("No parsable JSON records found in output JSONL files.")
                return

            st.subheader("Generation View")
            utils = ModelUtilsFactory.get_utils(str(model_utils_name))
            rec_index = 0
            if len(records) > 1:
                rec_index = st.slider("Record", 1, len(records), value=1) - 1
            record = records[rec_index]
            st.markdown(f"**Record {rec_index+1}/{len(records)}**")
            if "generation_prompt" in record:
                with st.expander("Generation Prompt", expanded=True):
                    st.code(record["generation_prompt"], language="text")

            gens = extract_assistant_text(record)
            if gens:
                tab_labels = [f"Gen {i+1}" for i in range(len(gens))] + ["Raw Record"]
                tabs = st.tabs(tab_labels)
                for i, t in enumerate(tabs):
                    if i < len(gens):
                        with t:
                            # Provide per-generation breakdown + raw toggle
                            render_assistant_view(gens[i], utils)
                    else:
                        with t:
                            st.json(record, expanded=False)
            else:
                st.info("No assistant generations detected in record.")

    poll()


if __name__ == "__main__":
    app()

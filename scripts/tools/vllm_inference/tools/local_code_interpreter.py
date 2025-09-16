import os
import sys
import time
import uuid
import asyncio
import base64
import json
import fcntl
import random
import tempfile
import requests
import subprocess
import signal
from textwrap import dedent
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import contextmanager
from typing import Optional, List, Tuple, Dict, Any
import shlex
import socket
import atexit

from phigen.logging import logger

app = FastAPI()

PYBOX_DIR = os.getenv("PYBOX_DIR", os.path.join(os.environ.get("HOME", "/tmp"), ".pybox"))
os.makedirs(PYBOX_DIR, exist_ok=True)
MAX_SESSIONS_FILE = os.path.join(PYBOX_DIR, ".pybox_max_sessions")
LOCK_FILE = os.path.join(PYBOX_DIR, ".pybox_lock")
DEFAULT_MAX_SESSIONS = int(os.getenv("PYBOX_MAX_SESSIONS", "300"))
BASE_IMAGE = os.getenv("PYBOX_BASE_IMAGE", "python3.12-slim-pybox:160825")
SERVER_PORT = 8765
SEVER_CODE_PORT = 65599
DOCKER_LABEL_KEY = "pybox"
DOCKER_LABEL_VAL = "true"
RUNTIME = os.getenv("PYBOX_RUNTIME", "apptainer").strip().lower()  # docker | apptainer
APPTAINER_IMAGE = os.getenv("PYBOX_APPTAINER_IMAGE", "")  # path to .sif image
APPTAINER_DOCKER_ARCHIVE = os.getenv(
    "PYBOX_APPTAINER_DOCKER_ARCHIVE", os.getenv("PYBOX_DOCKER_ARCHIVE", "")
)  # docker archive tar for apptainer build
DOCKER_IMAGE_TAR = os.getenv(
    "PYBOX_DOCKER_IMAGE_TAR", os.getenv("PYBOX_DOCKER_ARCHIVE", "")
)  # docker image tar for docker runtime
WHEELS_DIR = os.getenv("PYBOX_WHEELS_DIR", "/workspace/wheels")
OFFLINE_MODE = os.getenv("PYBOX_OFFLINE", "0") == "1"
APPTAINER_BIN = os.getenv("PYBOX_APPTAINER_BIN", "apptainer")  # or singularity
APPTAINER_FORCE_FALLBACK = os.getenv("PYBOX_APPTAINER_FORCE_FALLBACK", "0") == "1"  # force exec fallback (no instances)
_docker_image_loaded = False
APPTAINER_FALLBACK_MODE = False  # set True at runtime if instance start unsupported (e.g. hidepid)

# shared pip cache to speed installs across containers
PIP_CACHE_DIR = os.path.join(PYBOX_DIR, "pip-cache")
os.makedirs(PIP_CACHE_DIR, exist_ok=True)

# Override the list with PYBOX_COMMON_PIP_PACKAGES
COMMON_PIP_PACKAGES_DEFAULT = (
    "numpy pandas scipy scikit-learn matplotlib pillow requests beautifulsoup4 lxml pyarrow tqdm pydantic"
)
PIP_INDEX_URL = os.environ.get("PYBOX_PIP_INDEX_URL", "").strip()
PIP_EXTRA_ARGS = os.environ.get("PYBOX_PIP_EXTRA_ARGS", "").strip()
try:
    PIP_INSTALL_TIMEOUT = float(os.environ.get("PYBOX_PIP_INSTALL_TIMEOUT", "300"))
except Exception:
    PIP_INSTALL_TIMEOUT = 300.0


def run_cmd(cmd: List[str], timeout: Optional[float] = None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@contextmanager
def file_lock(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _docker_run_args_from_env() -> List[str]:
    extra = os.getenv("PYBOX_DOCKER_RUN_ARGS", "").strip()
    return shlex.split(extra) if extra else []


def _is_apptainer() -> bool:
    return RUNTIME == "apptainer"


def _apptainer_use_instances() -> bool:
    """Whether we should attempt to use instances (persistent daemon) for Apptainer.
    Falls back to exec-only mode if instances fail (e.g. hidepid constraint) or forced via env.
    """
    return _is_apptainer() and (not APPTAINER_FALLBACK_MODE) and (not APPTAINER_FORCE_FALLBACK)


def _ensure_apptainer_image() -> str:
    """Return path to apptainer image, building from docker archive if needed.
    We only attempt a build once per process; errors raise HTTPException."""
    if not _is_apptainer():
        return ""
    if APPTAINER_IMAGE and os.path.exists(APPTAINER_IMAGE):
        return APPTAINER_IMAGE
    # attempt build from docker archive if provided
    if APPTAINER_DOCKER_ARCHIVE and os.path.exists(APPTAINER_DOCKER_ARCHIVE):
        target = APPTAINER_IMAGE or os.path.join(PYBOX_DIR, "pybox_base.sif")
        # build only if not exists
        if not os.path.exists(target):
            res = run_cmd([APPTAINER_BIN, "build", target, f"docker-archive://{APPTAINER_DOCKER_ARCHIVE}"])
            if res.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Failed to build Apptainer image: {res.stderr.strip()}")
        return target
    raise HTTPException(
        status_code=500, detail="Apptainer image not found; set PYBOX_APPTAINER_IMAGE or PYBOX_DOCKER_ARCHIVE"
    )


def _apptainer_fallback_sessions() -> List[Tuple[str, datetime]]:
    sessions_root = os.path.join(PYBOX_DIR, "sessions")
    out: List[Tuple[str, datetime]] = []
    if not os.path.isdir(sessions_root):
        return out
    for d in os.listdir(sessions_root):
        session_dir = os.path.join(sessions_root, d)
        if not os.path.isdir(session_dir):
            continue
        name = f"pybox_{d}"
        try:
            stat = os.stat(session_dir)
            dt = datetime.utcfromtimestamp(stat.st_mtime)
        except Exception:
            dt = datetime.utcnow()
        out.append((name, dt))
    return out


def _remove_session_dir(session_id: str):
    try:
        sessions_root = os.path.join(PYBOX_DIR, "sessions")
        path = os.path.join(sessions_root, session_id)
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
            try:
                os.rmdir(path)
            except Exception:
                pass
    except Exception:
        pass


# New: resolve which packages to install
def _get_common_pip_packages() -> List[str]:
    override = os.environ.get("PYBOX_COMMON_PIP_PACKAGES", "").strip()
    if override:
        if override == "NO INSTALL":
            logger().debug("PYBOX_COMMON_PIP_PACKAGES set to 'NO INSTALL'; skipping common package install.")
            return []
        return shlex.split(override)
    return shlex.split(COMMON_PIP_PACKAGES_DEFAULT)


def _install_common_packages(cname: str):
    pkgs = _get_common_pip_packages()
    if not pkgs:
        logger().debug("No common packages requested; skipping pip install.")
        return
    quoted_pkgs = " ".join(shlex.quote(p) for p in pkgs)
    wheels_available = os.path.isdir(WHEELS_DIR) and any(f.endswith(".whl") for f in os.listdir(WHEELS_DIR) or [])
    if OFFLINE_MODE or wheels_available:
        pip_template = "{python} -m pip install --no-index --find-links /opt/pybox/wheels {pkgs}"
    else:
        index_part = f" -i {shlex.quote(PIP_INDEX_URL)}" if PIP_INDEX_URL else ""
        extra_part = f" {PIP_EXTRA_ARGS}" if PIP_EXTRA_ARGS else ""
        pip_template = (
            "{python} -m pip install --upgrade --disable-pip-version-check pip setuptools wheel && "
            f"{{python}} -m pip install --disable-pip-version-check{index_part}{extra_part} {{pkgs}}"
        )
    if _is_apptainer():
        if _apptainer_use_instances():
            pip_cmd = pip_template.format(python="python", pkgs=quoted_pkgs)
            res = run_cmd(
                [APPTAINER_BIN, "exec", f"instance://{cname}", "bash", "-lc", pip_cmd], timeout=PIP_INSTALL_TIMEOUT
            )
        else:
            session_id = cname.replace("pybox_", "", 1)
            session_dir = os.path.join(PYBOX_DIR, "sessions", session_id)
            venv_dir = os.path.join(session_dir, ".venv")
            os.makedirs(session_dir, exist_ok=True)
            binds = [f"{PIP_CACHE_DIR}:/root/.cache/pip", f"{session_dir}:/mnt/data"]
            if os.path.isdir(WHEELS_DIR):
                binds.append(f"{WHEELS_DIR}:/opt/pybox/wheels:ro")
            bind_args = []
            for b in binds:
                host_path = b.split(":", 1)[0]
                if host_path and os.path.exists(host_path):
                    bind_args.extend(["--bind", b])
            image_path = _ensure_apptainer_image()
            # Bootstrap venv if needed
            bootstrap = "python - <<'PY'\nimport os,venv\nvd='/mnt/data/.venv'\nif not os.path.exists(vd):\n    venv.EnvBuilder(with_pip=True).create(vd)\nprint('VENV_READY')\nPY"
            run_cmd([APPTAINER_BIN, "exec", *bind_args, image_path, "bash", "-lc", bootstrap])
            venv_python = os.path.join(venv_dir, "bin", "python")
            python_exec = venv_python if os.path.exists(venv_python) else "python"
            pip_cmd = pip_template.format(python=python_exec, pkgs=quoted_pkgs)
            res = run_cmd(
                [APPTAINER_BIN, "exec", *bind_args, image_path, "bash", "-lc", pip_cmd], timeout=PIP_INSTALL_TIMEOUT
            )
    else:
        pip_cmd = pip_template.format(python="python", pkgs=quoted_pkgs)
        res = run_cmd(["docker", "exec", "-i", cname, "bash", "-lc", pip_cmd], timeout=PIP_INSTALL_TIMEOUT)
    if res.returncode != 0:
        logger().warning(f"Common package install failed in {cname} (rc={res.returncode}): {res.stderr.strip()}")
    else:
        logger().debug(f"Installed common packages in {cname}: {pkgs}")


def _docker_ps_pybox(only_running=True) -> List[Tuple[str, str]]:
    """Return list of (name, id) for our containers only."""
    if _is_apptainer() and _apptainer_use_instances():
        # apptainer instances list (no running vs all distinction)
        result = subprocess.run([APPTAINER_BIN, "instance", "list"], capture_output=True, text=True)
        out = []
        if result.returncode == 0:
            for line in result.stdout.splitlines()[1:]:  # skip header
                parts = line.split()
                if not parts:
                    continue
                name = parts[0]
                if name.startswith("pybox_"):
                    out.append((name, name))
        return out
    elif _is_apptainer():
        # fallback mode sessions
        return [(name, name) for name, _ in _apptainer_fallback_sessions()]
    else:
        if only_running:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"label={DOCKER_LABEL_KEY}={DOCKER_LABEL_VAL}",
                    "--format",
                    "{{.Names}}\t{{.ID}}",
                ],
                capture_output=True,
                text=True,
            )
        else:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--all",
                    "--filter",
                    f"label={DOCKER_LABEL_KEY}={DOCKER_LABEL_VAL}",
                    "--format",
                    "{{.Names}}\t{{.ID}}",
                ],
                capture_output=True,
                text=True,
            )
        out = []
        for line in result.stdout.strip().split("\n"):
            if line:
                name, cid = line.split("\t", 1)
                out.append((name, cid))
        return out


def _docker_created_iso(cid: str) -> Optional[datetime]:
    if _is_apptainer() and _apptainer_use_instances():
        # Apptainer: we don't have per-instance creation time directly; fallback to current time
        return datetime.utcnow()
    res = subprocess.run(["docker", "inspect", "-f", "{{.Created}}", cid], capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.strip():
        return None
    s = res.stdout.strip()  # e.g., 2025-08-12T10:20:30.123456789Z
    s = s.rstrip("Z")
    # trim excessive nanoseconds for fromisoformat
    if "." in s:
        head, tail = s.split(".", 1)
        tail = (tail + "000000")[:6]  # microseconds
        s = f"{head}.{tail}"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def list_pybox_containers(only_running=True) -> List[Tuple[str, datetime]]:
    """Return list of (name, created_at) sorted oldest first, only ours."""
    entries = []
    if _is_apptainer() and not _apptainer_use_instances():
        entries.extend(_apptainer_fallback_sessions())
    else:
        for name, cid in _docker_ps_pybox(only_running=only_running):
            dt = _docker_created_iso(cid)
            entries.append((name, dt))
    # ensure consistent sortable key; place unknown dt last
    return sorted(entries, key=lambda x: (x[1] is None, x[1] or datetime.max))


def session_exists(session_id: str):
    cname = f"pybox_{session_id}"
    return any(name == cname for name, _ in list_pybox_containers())


def _install_server_script(cname: str):
    """Copy a tiny persistent eval server into the container."""
    server_py = dedent(
        f"""
        import os, socket, threading, json, traceback, sys, io, contextlib, errno

        SOCK = "/mnt/data/.pybox.sock"

        # ensure no stale socket file
        try:
            os.remove(SOCK)
        except OSError:
            pass

        # Set working directory so relative paths land under /mnt/data
        try:
            os.makedirs("/mnt/data", exist_ok=True)
            os.chdir("/mnt/data")
        except Exception:
            pass

        ns = {{"__name__": "__main__"}}  # persistent namespace
        ns_lock = threading.Lock()       # serialize code execution

        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.bind(SOCK)
        s.listen(64)

        def handle(conn):
            try:
                buf = b""
                while True:
                    try:
                        chunk = conn.recv({SEVER_CODE_PORT})
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk

                # Ignore empty probes (readiness checks)
                if not buf:
                    return

                try:
                    req = json.loads(buf.decode())
                except Exception:
                    # error reply; ignore if peer already closed
                    try:
                        conn.sendall(json.dumps({{"ok": False, "error": "bad_request"}}).encode())
                    except Exception:
                        pass
                    return

                code = req.get("code", "")
                out, err = io.StringIO(), io.StringIO()
                ok = True
                tb = None
                
                # prevent interleaving in the shared ns
                with ns_lock, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    try:
                        compiled = compile(code, "<pybox>", "exec")
                        exec(compiled, ns)
                    except Exception:
                        ok = False
                        tb = traceback.format_exc()

                resp = {{"ok": ok, "stdout": out.getvalue(), "stderr": err.getvalue()}}
                if not ok:
                    resp["traceback"] = tb
                try:
                    conn.sendall(json.dumps(resp).encode())
                except Exception:
                    pass
            finally:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass

        def main():
            try:
                while True:
                    try:
                        conn, _ = s.accept()
                    except socket.timeout:
                        # periodic timeout; keep loop responsive
                        continue
                    except OSError as e:
                        err = getattr(e, "errno", None)
                        if err in (errno.EINTR, errno.EAGAIN):
                            continue
                        break
                    threading.Thread(target=handle, args=(conn,), daemon=True).start()
            finally:
                try:
                    s.close()
                except Exception:
                    pass
                try:
                    os.remove(SOCK)
                except Exception:
                    pass

        if __name__ == "__main__":
            main()
    """
    ).strip()

    if _is_apptainer() and _apptainer_use_instances():
        write_cmd = [
            APPTAINER_BIN,
            "exec",
            f"instance://{cname}",
            "bash",
            "-lc",
            "cat > /pybox_server.py <<'PY'\n" + server_py + "\nPY",
        ]
    else:
        write_cmd = [
            "docker",
            "exec",
            "-i",
            cname,
            "bash",
            "-lc",
            "cat > /pybox_server.py <<'PY'\n" + server_py + "\nPY",
        ]
    res = run_cmd(write_cmd)
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Failed to install server: {res.stderr.strip()}")


def _server_is_running(cname: str) -> bool:
    check_py = dedent(
        f"""
        import socket, sys
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            s.connect("/mnt/data/.pybox.sock")
            s.close()
            sys.exit(0)
        except Exception:
            sys.exit(1)
    """
    ).strip()
    if _is_apptainer() and _apptainer_use_instances():
        res = run_cmd(
            [APPTAINER_BIN, "exec", f"instance://{cname}", "bash", "-lc", "python - <<'PY'\n" + check_py + "\nPY"]
        )
    else:
        res = run_cmd(["docker", "exec", "-i", cname, "bash", "-lc", "python - <<'PY'\n" + check_py + "\nPY"])
    return res.returncode == 0


def _start_server(cname: str):
    # In exec-fallback mode for Apptainer there is no persistent server
    if _is_apptainer() and not _apptainer_use_instances():
        return
    if not _server_is_running(cname):
        if _is_apptainer():
            res = run_cmd(
                [
                    APPTAINER_BIN,
                    "exec",
                    f"instance://{cname}",
                    "bash",
                    "-lc",
                    "nohup python /pybox_server.py >/dev/null 2>&1 &",
                ]
            )
        else:
            res = run_cmd(["docker", "exec", "-d", cname, "python", "/pybox_server.py"])
        if res.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Failed to start server: {res.stderr.strip()}")


def _fallback_paths(session_id: str) -> Dict[str, str]:
    sd = os.path.join(PYBOX_DIR, "sessions", session_id)
    return {
        "session_dir": sd,
        "sock": os.path.join(sd, ".pybox.sock"),
        "pid": os.path.join(sd, ".pybox_daemon.pid"),
        "server_py": os.path.join(sd, "pybox_server.py"),
        "log": os.path.join(sd, ".pybox_daemon.log"),
    }


def _apptainer_bind_args_for_session(session_dir: str) -> List[str]:
    binds = [f"{session_dir}:/mnt/data", f"{PIP_CACHE_DIR}:/root/.cache/pip"]
    if os.path.isdir(WHEELS_DIR):
        binds.append(f"{WHEELS_DIR}:/opt/pybox/wheels:ro")
    bind_args: List[str] = []
    for b in binds:
        host_path = b.split(":", 1)[0]
        if host_path and os.path.exists(host_path):
            bind_args.extend(["--bind", b])
    return bind_args


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _fallback_daemon_running(session_id: str, timeout: float = 0.5) -> bool:
    paths = _fallback_paths(session_id)
    image_path = _ensure_apptainer_image()
    bind_args = _apptainer_bind_args_for_session(paths["session_dir"])
    check_py = dedent(
        f"""
        import socket, sys
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout({timeout})
        try:
            s.connect('/mnt/data/.pybox.sock')
            s.close()
            sys.exit(0)
        except Exception:
            sys.exit(1)
    """
    ).strip()

    res = run_cmd([APPTAINER_BIN, "exec", *bind_args, image_path, "python", "-c", check_py])
    return res.returncode == 0


def _fallback_daemon_proc_alive(session_id: str) -> bool:
    """Check if the recorded daemon process for this session is alive on the host."""
    paths = _fallback_paths(session_id)
    try:
        if os.path.isfile(paths["pid"]):
            pid_s = open(paths["pid"]).read().strip() or "0"
            pid = int(pid_s) if pid_s.isdigit() else 0
            return pid > 0 and _process_alive(pid)
    except Exception:
        return False
    return False


def _start_fallback_daemon(session_id: str, wait_seconds: float = 5.0):
    paths = _fallback_paths(session_id)
    os.makedirs(paths["session_dir"], exist_ok=True)

    # Early exit if a daemon is already alive; don't spawn a second one.
    if _fallback_daemon_proc_alive(session_id):
        deadline = time.time() + min(wait_seconds, 2.0)
        while time.time() < deadline:
            if _fallback_daemon_running(session_id, timeout=0.15):
                return
            time.sleep(0.1)
        return

    # write server script inside session dir
    server_py = dedent(
        f"""
        import os, socket, threading, json, traceback, sys, io, contextlib, errno, time, signal

        SOCK = "/mnt/data/.pybox.sock"

        # ensure no stale socket file
        try:
            os.remove(SOCK)
        except OSError:
            pass

        # Set working directory so relative paths land under /mnt/data
        try:
            os.makedirs("/mnt/data", exist_ok=True)
            os.chdir("/mnt/data")
        except Exception:
            pass

        # Make per-session venv packages importable
        sp_root = "/mnt/data/.venv/lib"
        if os.path.isdir(sp_root):
            for root, dirs, files in os.walk(sp_root):
                if root.endswith("site-packages") and root not in sys.path:
                    sys.path.insert(0, root)

        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.bind(SOCK)
        s.listen(64)

        def _shutdown(signum, frame):
            try:
                s.close()
            except Exception:
                pass
        try:
            signal.signal(signal.SIGTERM, _shutdown)
            signal.signal(signal.SIGINT, _shutdown)
        except Exception:
            pass

        ns = {{"__name__": "__main__"}}
        ns_lock = threading.Lock()

        def handle(conn):
            try:
                buf = b""
                while True:
                    try:
                        chunk = conn.recv({SEVER_CODE_PORT})
                    except socket.timeout:
                        # Keep waiting for peer to finish sending
                        continue
                    if not chunk:
                        break
                    buf += chunk

                # Ignore empty probes
                if not buf:
                    return

                try:
                    req = json.loads(buf.decode())
                    code = req.get("code", "")
                except Exception:
                    try:
                        conn.sendall(json.dumps({{"ok": False, "error": "bad_request"}}).encode())
                    except Exception:
                        pass
                    return

                out, err = io.StringIO(), io.StringIO()
                ok, tb = True, None
                with ns_lock, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    try:
                        compiled = compile(code, "<pybox>", "exec")
                        exec(compiled, ns)
                    except Exception:
                        ok = False
                        tb = traceback.format_exc()

                resp = {{"ok": ok, "stdout": out.getvalue(), "stderr": err.getvalue()}}
                if not ok:
                    resp["traceback"] = tb
                try:
                    conn.sendall(json.dumps(resp).encode())
                except Exception:
                    pass
            finally:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass

        try:
            while True:
                try:
                    conn, _ = s.accept()
                except socket.timeout:
                    # Periodic timeout so we can detect closed listener and exit
                    try:
                        # If socket fileno is invalid, listener is closed; exit
                        if s.fileno() < 0:
                            break
                    except Exception:
                        break
                    continue
                except OSError as e:
                    err = getattr(e, "errno", None)
                    if err in (errno.EINTR, errno.EAGAIN):
                        continue           # transient, retry
                    if err in (errno.EBADF,):
                        break              # socket closed during shutdown
                    # Backoff slightly on other transient errors
                    time.sleep(0.05)
                    continue
                threading.Thread(target=handle, args=(conn,), daemon=True).start()
        finally:
            try:
                s.close()
            except Exception:
                pass
            # Cleanup socket file
            try:
                os.remove(SOCK)
            except Exception:
                pass
    """
    ).strip()

    with open(paths["server_py"], "w") as f:
        f.write(server_py)

    image_path = _ensure_apptainer_image()
    bind_args = _apptainer_bind_args_for_session(paths["session_dir"])
    # stream daemon logs to a per-session log
    log_f = open(paths["log"], "ab", buffering=0)

    _env = os.environ.copy()
    # Detach into its own process group so it survives the caller exiting
    proc = subprocess.Popen(
        [APPTAINER_BIN, "exec", *bind_args, image_path, "python", "/mnt/data/pybox_server.py"],
        stdout=log_f,
        stderr=log_f,
        env=_env,
        preexec_fn=os.setsid,  # Linux only
    )
    with open(paths["pid"], "w") as f:
        f.write(str(proc.pid))

    # wait for readiness
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise HTTPException(status_code=500, detail="Fallback daemon exited prematurely.")
        if _fallback_daemon_running(session_id, timeout=0.2):
            return
        time.sleep(0.1)

    # failed to become ready; try to terminate
    try:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
            proc.kill()
    except Exception:
        pass
    raise HTTPException(status_code=500, detail="Failed to start fallback daemon in time.")


def _stop_fallback_daemon(session_id: str):
    paths = _fallback_paths(session_id)
    try:
        if os.path.isfile(paths["pid"]):
            pid_s = open(paths["pid"]).read().strip() or "0"
            pid = int(pid_s) if pid_s.isdigit() else 0
            if pid > 0 and _process_alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                    # wait briefly
                    for _ in range(20):
                        if not _process_alive(pid):
                            break
                        time.sleep(0.1)
                    if _process_alive(pid):
                        os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
    except Exception:
        pass
    # cleanup socket/pid files
    for fp in (paths["sock"], paths["pid"]):
        try:
            os.remove(fp)
        except Exception:
            pass


def _send_code_to_server(cname: str, code: str, timeout: Optional[float] = None):
    """Send code to the persistent server and return parsed JSON response."""
    code_b64 = base64.b64encode(code.encode()).decode()
    client_py = dedent(
        f"""
        import socket, json, base64, sys, time
        code = base64.b64decode("{code_b64}".encode()).decode()
        SOCK = "/mnt/data/.pybox.sock"
        conn_timeout = {f"{timeout:.3f}" if timeout else "60.0"}
        recv_timeout = {f"{timeout:.3f}" if timeout else "None"}

        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if conn_timeout:
            s.settimeout(conn_timeout)
        try:
            s.connect(SOCK)
        except Exception:
            print(json.dumps({{"ok": False, "error": "connect_failed"}}), end="")
            sys.exit(0)
        if recv_timeout:
            s.settimeout(recv_timeout)
        s.sendall(json.dumps({{"code": code}}).encode())
        s.shutdown(socket.SHUT_WR)
        buf = b""
        while True:
            try:
                chunk = s.recv({SEVER_CODE_PORT})
                if not chunk:
                    break
                buf += chunk
            except socket.timeout:
                print(json.dumps({{"ok": False, "timeout": True}}), end="")
                sys.exit(0)
        print(buf.decode(), end="")
    """
    ).strip()

    if _is_apptainer() and not _apptainer_use_instances():
        # Ensure fallback daemon is running
        session_id = cname.replace("pybox_", "", 1)
        # Only start if both the socket isn’t accepting AND there’s no alive PID
        if not _fallback_daemon_running(session_id):
            if not _fallback_daemon_proc_alive(session_id):
                _start_fallback_daemon(session_id)
            else:
                # give the existing daemon a brief chance to accept after a transient blip
                time.sleep(0.2)

        image_path = _ensure_apptainer_image()
        paths = _fallback_paths(session_id)
        bind_args = _apptainer_bind_args_for_session(paths["session_dir"])

        try:
            res = run_cmd(
                [
                    APPTAINER_BIN,
                    "exec",
                    *bind_args,
                    image_path,
                    "bash",
                    "-lc",
                    "python - <<'PY'\n" + client_py + "\nPY",
                ],
                timeout=(timeout + 2) if timeout else None,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="subprocess.TimeoutExpired: Code Execution Timed Out, possible infinite loops in the code",
            )

        # If first attempt failed (e.g., daemon crashed), restart and retry once
        if res.returncode != 0 or not (res.stdout or "").strip():
            _start_fallback_daemon(session_id)
            try:
                res = run_cmd(
                    [
                        APPTAINER_BIN,
                        "exec",
                        *bind_args,
                        image_path,
                        "bash",
                        "-lc",
                        "python - <<'PY'\n" + client_py + "\nPY",
                    ],
                    timeout=(timeout + 2) if timeout else None,
                )
            except subprocess.TimeoutExpired:
                raise HTTPException(
                    status_code=504,
                    detail="subprocess.TimeoutExpired: Code Execution Timed Out, possible infinite loops in the code",
                )

        if res.returncode != 0:
            raise HTTPException(status_code=500, detail=res.stderr.strip() or "Execution failed via fallback daemon")

        try:
            return json.loads(res.stdout.strip() or "{}")
        except Exception:
            raise HTTPException(status_code=500, detail="Malformed response from fallback daemon")

    # Normal path (docker or apptainer instance)
    try:
        if _is_apptainer():
            res = run_cmd(
                [APPTAINER_BIN, "exec", f"instance://{cname}", "bash", "-lc", "python - <<'PY'\n" + client_py + "\nPY"],
                timeout=(timeout + 2) if timeout else None,
            )
        else:
            res = run_cmd(
                [
                    "docker",
                    "exec",
                    "-i",
                    cname,
                    "bash",
                    "-lc",
                    "python - <<'PY'\n" + client_py + "\nPY",
                ],
                timeout=(timeout + 2) if timeout else None,
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="subprocess.TimeoutExpired: Code Execution Timed Out, possible infinite loops in the code",
        )

    if res.returncode != 0:
        _start_server(cname)
        if _is_apptainer():
            res = run_cmd(
                [APPTAINER_BIN, "exec", f"instance://{cname}", "bash", "-lc", "python - <<'PY'\n" + client_py + "\nPY"],
                timeout=(timeout + 2) if timeout else None,
            )
        else:
            res = run_cmd(
                ["docker", "exec", "-i", cname, "bash", "-lc", "python - <<'PY'\n" + client_py + "\nPY"],
                timeout=(timeout + 2) if timeout else None,
            )
        if res.returncode != 0:
            raise HTTPException(status_code=500, detail=res.stderr.strip() or "Execution failed inside container")

    try:
        payload = json.loads(res.stdout.strip() or "{}")
    except Exception:
        raise HTTPException(status_code=500, detail="Malformed response from session server")
    return payload


def create_session(session_id: str):
    cname = f"pybox_{session_id}"
    if _is_apptainer():
        global APPTAINER_FALLBACK_MODE
        image_path = _ensure_apptainer_image()
        session_dir = os.path.join(PYBOX_DIR, "sessions", session_id)
        os.makedirs(session_dir, exist_ok=True)
        if _apptainer_use_instances():
            binds = [
                f"{PIP_CACHE_DIR}:/root/.cache/pip",
                f"{WHEELS_DIR}:/opt/pybox/wheels:ro",
                f"{session_dir}:/mnt/data",
            ]
            bind_args = []
            for b in binds:
                if b.split(":")[0] and os.path.exists(b.split(":")[0]):
                    bind_args.extend(["--bind", b])
            res = run_cmd([APPTAINER_BIN, "instance", "start", *bind_args, image_path, cname])
            if res.returncode != 0:
                stderr = (res.stderr or "").lower()
                if ("hidepid" in stderr) or ("permission denied" in stderr) or ("operation not permitted" in stderr):
                    logger().debug(
                        "Apptainer instance start failed due to host restrictions; switching to exec fallback mode (stateless pickle-based namespace)."
                    )
                    APPTAINER_FALLBACK_MODE = True
                else:
                    raise HTTPException(
                        status_code=500, detail=f"Failed to start apptainer instance: {res.stderr.strip()}"
                    )
            else:
                # Create working dir /mnt/data inside instance
                run_cmd(
                    [APPTAINER_BIN, "exec", f"instance://{cname}", "bash", "-lc", "mkdir -p /mnt/data"]
                )  # ignore rc
        if not _apptainer_use_instances():
            # ensure persistent venv exists
            binds = [f"{session_dir}:/mnt/data", f"{PIP_CACHE_DIR}:/root/.cache/pip"]
            if os.path.isdir(WHEELS_DIR):
                binds.append(f"{WHEELS_DIR}:/opt/pybox/wheels:ro")
            bind_args = []
            for b in binds:
                if b.split(":")[0] and os.path.exists(b.split(":")[0]):
                    bind_args.extend(["--bind", b])
            bootstrap = "python - <<'PY'\nimport os,venv\nvd='/mnt/data/.venv'\nif not os.path.exists(vd):\n    venv.EnvBuilder(with_pip=True).create(vd)\nprint('VENV_READY')\nPY"
            run_cmd([APPTAINER_BIN, "exec", *bind_args, image_path, "bash", "-lc", bootstrap])

            # start fallback daemon for this session (Unix socket server)
            _start_fallback_daemon(session_id)
    else:
        global _docker_image_loaded
        if OFFLINE_MODE and DOCKER_IMAGE_TAR and not _docker_image_loaded and os.path.exists(DOCKER_IMAGE_TAR):
            # attempt to load image tar (ignore failures if already loaded)
            load_res = run_cmd(["docker", "load", "-i", DOCKER_IMAGE_TAR])
            if load_res.returncode == 0:
                _docker_image_loaded = True
        cmd = (
            [
                "docker",
                "run",
                "-dit",
                "--name",
                cname,
                "--label",
                f"{DOCKER_LABEL_KEY}={DOCKER_LABEL_VAL}",
                "--label",
                f"{DOCKER_LABEL_KEY}.session={session_id}",
                "-w",
                "/mnt/data",
                "-v",
                f"{PIP_CACHE_DIR}:/root/.cache/pip:rw",
            ]
            + (["-v", f"{WHEELS_DIR}:/opt/pybox/wheels:ro"] if os.path.isdir(WHEELS_DIR) else [])
            + _docker_run_args_from_env()
            + [BASE_IMAGE, "python", "-i"]
        )
        res = run_cmd(cmd)
        if res.returncode != 0:
            raise HTTPException(status_code=500, detail=res.stderr.strip())

    _install_common_packages(cname)
    if _is_apptainer() and not _apptainer_use_instances():
        # No server script in fallback mode
        pass
    else:
        _install_server_script(cname)
        _start_server(cname)
    return cname


class UploadItem(BaseModel):
    path: str  # path inside the container; relative paths are resolved under /mnt/data
    content_b64: str  # base64-encoded file content
    makedirs: bool = True
    mode: Optional[int] = None  # optional chmod (e.g., 420 for 0o644)


class UploadRequest(BaseModel):
    session_id: str
    files: List[UploadItem]


class DownloadItem(BaseModel):
    path: str  # path inside the container


class DownloadRequest(BaseModel):
    session_id: str
    files: List[DownloadItem]


class SetupRequest(BaseModel):
    max_sessions: int = DEFAULT_MAX_SESSIONS


class ExecRequest(BaseModel):
    session_id: str
    code: str
    timeout: Optional[float] = None
    kill_on_timeout: bool = False


@app.post("/setup")
def setup_sessions(req: SetupRequest):
    def write(max_sessions: int):
        with file_lock(LOCK_FILE):
            with open(MAX_SESSIONS_FILE, "w") as f:
                f.write(str(max_sessions))

    if os.path.exists(MAX_SESSIONS_FILE):
        current_max_sessions = int(open(MAX_SESSIONS_FILE).read().strip())
        if current_max_sessions != req.max_sessions:
            write(req.max_sessions)
            return {"status": "updated", "max_sessions": req.max_sessions}
        else:
            raise HTTPException(status_code=400, detail="Already set up.")
    if req.max_sessions < 1:
        raise HTTPException(status_code=400, detail="max_sessions must be >= 1")

    write(req.max_sessions)
    return {"status": "ok", "max_sessions": req.max_sessions}


@app.post("/exec")
def execute_code(req: ExecRequest):
    # Use configured value if present; otherwise default to DEFAULT_MAX_SESSIONS
    if os.path.exists(MAX_SESSIONS_FILE):
        try:
            with open(MAX_SESSIONS_FILE) as f:
                max_sessions = int(f.read().strip())
                if max_sessions < 1:
                    max_sessions = DEFAULT_MAX_SESSIONS
        except Exception:
            max_sessions = DEFAULT_MAX_SESSIONS
    else:
        max_sessions = DEFAULT_MAX_SESSIONS

    # list/evict/create with a process lock to avoid races
    with file_lock(LOCK_FILE):
        if not session_exists(req.session_id):
            containers = list_pybox_containers(only_running=False)
            if len(containers) >= max_sessions:
                oldest_name, _ = containers[0]
                if _is_apptainer():
                    if _apptainer_use_instances():
                        run_cmd([APPTAINER_BIN, "instance", "stop", oldest_name])
                    else:
                        _stop_fallback_daemon(oldest_name.replace("pybox_", "", 1))
                        _remove_session_dir(oldest_name.replace("pybox_", "", 1))
                else:
                    run_cmd(["docker", "rm", "-f", oldest_name])
            create_session(req.session_id)

    cname = f"pybox_{req.session_id}"
    _start_server(cname)

    payload = _send_code_to_server(cname, req.code, timeout=req.timeout)
    # Handle timeout signaled by client script (server kept running)
    if payload.get("timeout"):
        if req.kill_on_timeout:
            if _is_apptainer():
                if _apptainer_use_instances():
                    run_cmd([APPTAINER_BIN, "instance", "stop", cname])
                else:
                    _remove_session_dir(req.session_id)
            else:
                run_cmd(["docker", "rm", "-f", cname])
        raise HTTPException(
            status_code=504, detail="HTTPException: Code Execution Timed Out, possible infinite loops in the code"
        )

    return {
        "stdout": payload.get("stdout", ""),
        "stderr": payload.get("stderr", ""),
        "returncode": 0 if payload.get("ok") else 1,
        "traceback": payload.get("traceback"),
    }


@app.post("/files/upload")
def upload_files(req: UploadRequest):
    # Ensure session exists (same policy as /exec), using configured/default max_sessions
    if os.path.exists(MAX_SESSIONS_FILE):
        try:
            with open(MAX_SESSIONS_FILE) as f:
                max_sessions = int(f.read().strip() or "0") or DEFAULT_MAX_SESSIONS
        except Exception:
            max_sessions = DEFAULT_MAX_SESSIONS
    else:
        max_sessions = DEFAULT_MAX_SESSIONS

    with file_lock(LOCK_FILE):
        if not session_exists(req.session_id):
            containers = list_pybox_containers()
            if len(containers) >= max_sessions:
                oldest_name, _ = containers[0]
                if _is_apptainer():
                    if _apptainer_use_instances():
                        run_cmd([APPTAINER_BIN, "instance", "stop", oldest_name])
                    else:
                        _remove_session_dir(oldest_name.replace("pybox_", "", 1))
                else:
                    run_cmd(["docker", "rm", "-f", oldest_name])
            create_session(req.session_id)

    cname = f"pybox_{req.session_id}"

    files_payload = [
        {"path": f.path, "content_b64": f.content_b64, "makedirs": f.makedirs, "mode": f.mode} for f in req.files
    ]
    payload_str = json.dumps(files_payload)

    code = dedent(
        """
        import os, json, base64, sys
        root = '/mnt/data'
        items = json.loads(sys.stdin.buffer.read().decode())
        results = []
        for it in items:
            p = it.get('path','')
            if not p:
                results.append({'path': p, 'ok': False, 'error': 'missing_path'})
                continue
            if not p.startswith('/'):
                p = os.path.join(root, p)
            try:
                if it.get('makedirs', True):
                    d = os.path.dirname(p)
                    if d:
                        os.makedirs(d, exist_ok=True)
                data = base64.b64decode((it.get('content_b64') or '').encode())
                with open(p, 'wb') as fh:
                    fh.write(data)
                mode = it.get('mode', None)
                if mode is not None:
                    os.chmod(p, int(mode))
                results.append({'path': p, 'ok': True, 'bytes': len(data)})
            except Exception as e:
                results.append({'path': p, 'ok': False, 'error': str(e)})
        print(json.dumps({'ok': True, 'results': results}))
    """
    ).strip()

    if _is_apptainer():
        if _apptainer_use_instances():
            res = subprocess.run(
                [APPTAINER_BIN, "exec", f"instance://{cname}", "python", "-c", code],
                input=payload_str,
                capture_output=True,
                text=True,
            )
        else:
            # exec fallback: run ephemeral container with binds
            session_dir = os.path.join(PYBOX_DIR, "sessions", req.session_id)
            os.makedirs(session_dir, exist_ok=True)
            image_path = _ensure_apptainer_image()
            binds = [f"{session_dir}:/mnt/data", f"{PIP_CACHE_DIR}:/root/.cache/pip"]
            if os.path.isdir(WHEELS_DIR):
                binds.append(f"{WHEELS_DIR}:/opt/pybox/wheels:ro")
            bind_args = []
            for b in binds:
                host_path = b.split(":", 1)[0]
                if host_path and os.path.exists(host_path):
                    bind_args.extend(["--bind", b])
            res = subprocess.run(
                [APPTAINER_BIN, "exec", *bind_args, image_path, "python", "-c", code],
                input=payload_str,
                capture_output=True,
                text=True,
            )
    else:
        res = subprocess.run(
            ["docker", "exec", "-i", cname, "python", "-c", code], input=payload_str, capture_output=True, text=True
        )
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail=res.stderr.strip() or "Upload failed in container")
    try:
        payload = json.loads(res.stdout.strip())
    except Exception:
        raise HTTPException(status_code=500, detail="Malformed response from upload handler")
    return payload


@app.post("/files/download")
def download_files(req: DownloadRequest):
    # For download, require session exists; do not auto-create
    if not session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    cname = f"pybox_{req.session_id}"

    files_payload = [{"path": f.path} for f in req.files]
    payload_str = json.dumps(files_payload)

    code = dedent(
        """
        import os, json, base64, sys
        root = '/mnt/data'
        items = json.loads(sys.stdin.buffer.read().decode())
        results = []
        for it in items:
            p = it.get('path','')
            if not p:
                results.append({'path': p, 'ok': False, 'error': 'missing_path'})
                continue
            if not p.startswith('/'):
                p = os.path.join(root, p)
            try:
                if not os.path.exists(p):
                    results.append({'path': p, 'ok': True, 'exists': False})
                    continue
                with open(p, 'rb') as fh:
                    data = fh.read()
                results.append({
                    'path': p,
                    'ok': True,
                    'exists': True,
                    'size': len(data),
                    'content_b64': base64.b64encode(data).decode()
                })
            except Exception as e:
                results.append({'path': p, 'ok': False, 'error': str(e)})
        print(json.dumps({'ok': True, 'results': results}))
    """
    ).strip()

    if _is_apptainer():
        if _apptainer_use_instances():
            res = subprocess.run(
                [APPTAINER_BIN, "exec", f"instance://{cname}", "python", "-c", code],
                input=payload_str,
                capture_output=True,
                text=True,
            )
        else:
            session_dir = os.path.join(PYBOX_DIR, "sessions", req.session_id)
            if not os.path.isdir(session_dir):
                raise HTTPException(status_code=404, detail="Session not found")
            image_path = _ensure_apptainer_image()
            binds = [f"{session_dir}:/mnt/data", f"{PIP_CACHE_DIR}:/root/.cache/pip"]
            if os.path.isdir(WHEELS_DIR):
                binds.append(f"{WHEELS_DIR}:/opt/pybox/wheels:ro")
            bind_args = []
            for b in binds:
                host_path = b.split(":", 1)[0]
                if host_path and os.path.exists(host_path):
                    bind_args.extend(["--bind", b])
            res = subprocess.run(
                [APPTAINER_BIN, "exec", *bind_args, image_path, "python", "-c", code],
                input=payload_str,
                capture_output=True,
                text=True,
            )
    else:
        res = subprocess.run(
            ["docker", "exec", "-i", cname, "python", "-c", code], input=payload_str, capture_output=True, text=True
        )
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail=res.stderr.strip() or "Download failed in container")
    try:
        payload = json.loads(res.stdout.strip())
    except Exception:
        raise HTTPException(status_code=500, detail="Malformed response from download handler")
    return payload


@app.post("/reset")
def reset_sessions():
    containers = [name for name, _ in list_pybox_containers(only_running=False)]
    for cname in containers:
        if _is_apptainer():
            if _apptainer_use_instances():
                run_cmd([APPTAINER_BIN, "instance", "stop", cname])
            else:
                sid = cname.replace("pybox_", "", 1)
                _stop_fallback_daemon(sid)
                _remove_session_dir(sid)
        else:
            run_cmd(["docker", "rm", "-f", cname])
    if os.path.exists(MAX_SESSIONS_FILE):
        os.remove(MAX_SESSIONS_FILE)
    return {"status": "reset", "removed": containers}


@app.post("/reset_session/{session_id}")
def reset_session(session_id: str):
    cname = f"pybox_{session_id}"
    if _is_apptainer():
        if _apptainer_use_instances():
            run_cmd([APPTAINER_BIN, "instance", "stop", cname])
        else:
            _stop_fallback_daemon(session_id)
            _remove_session_dir(session_id)
    else:
        run_cmd(["docker", "rm", "-f", cname])
    return {"status": "reset", "removed": cname}


class PyboxClient:
    def __init__(self, base_url: str = "http://localhost:6989", max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries

    def setup(self, max_sessions: int = DEFAULT_MAX_SESSIONS):
        r = requests.post(f"{self.base_url}/setup", json={"max_sessions": max_sessions})
        self._raise_for_status(r)
        return r.json()

    async def execute_code_single_with_retry(
        self,
        code: str,
        index: int,
        session_identifier: str,
        timeout: Optional[float] = None,
        kill_on_timeout: bool = False,
        upload_files: Optional[List[Dict[str, Any]]] = None,
        download_files: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        last_error = None
        session_identifier = session_identifier or str(uuid.uuid4())
        for attempt in range(self.max_retries):
            start_time = time.time()
            try:
                file_results = {}
                file_results["uploads"] = await loop.run_in_executor(
                    None, self._handle_file_uploads, session_identifier, {"files_to_upload": upload_files}
                )
                payload = {
                    "session_id": session_identifier,
                    "code": code,
                    "timeout": timeout,
                    "kill_on_timeout": kill_on_timeout,
                }
                payload = {k: v for k, v in payload.items() if v is not None}

                # avoid blocking the event loop
                resp = await loop.run_in_executor(None, lambda: requests.post(f"{self.base_url}/exec", json=payload))
                self._raise_for_status(resp)
                result_dict = resp.json()

                file_results["downloads"] = await loop.run_in_executor(
                    None, self._handle_file_downloads, session_identifier, {"files_to_download": download_files}
                )

                success = True
                stdout = (result_dict.get("stdout") or "").strip()
                stderr = (result_dict.get("stderr") or "").strip()
                traceback = (result_dict.get("traceback") or "").strip()  # safe for None
                return_code = result_dict.get("returncode", 1)

                if stderr or traceback or return_code != 0:
                    success = False
                    stderr = stderr + "\n" + traceback if traceback else stderr

                return {
                    "index": index,
                    "code": code,
                    "session_identifier": session_identifier,
                    "execution_time": round(time.time() - start_time, 2),
                    "success": success,
                    "raw_result": result_dict,
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": return_code,
                    "file_operations": file_results,
                }
            except requests.exceptions.HTTPError as e:
                last_error = str(e)
                if "Code Execution Timed Out" in last_error:
                    return {
                        "index": index,
                        "exception": "HTTPError: Code Execution Timed Out, possible infinite loops in the code",
                        "success": False,
                    }
                elif attempt < self.max_retries - 1:
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)

        return {
            "index": index,
            "exception": f"Failed to execute code after {self.max_retries} attempts: {last_error}",
            "success": False,
        }

    async def execute_code(self, code_block: Tuple, timeout: int = 200) -> Dict:
        code, index, session_identifier, upload_files, download_files = code_block
        try:
            return await asyncio.wait_for(
                self.execute_code_single_with_retry(
                    code,
                    index,
                    session_identifier=session_identifier,
                    timeout=timeout,
                    upload_files=upload_files,
                    download_files=download_files,
                ),
                timeout=(20 + timeout),
            )
        except asyncio.TimeoutError:
            return {
                "index": index,
                "exception": "asyncio.TimeoutError: Code Execution Timed Out, possible infinite loops in the code",
                "success": False,
            }

    def _handle_file_uploads(self, session_identifier: str, uploads):
        """
        Accepts either:
          - dict form: {"files_to_upload": [{"local_file": "...", "remote_file": "..."}, ...]}
          - legacy list form: [{"path": "...", content_b64|content|local_path, ...}]
        """
        if not uploads:
            return {"ok": True, "results": []}

        files_spec = []
        if isinstance(uploads, dict) and "files_to_upload" in uploads:
            items = uploads.get("files_to_upload") or []
            for it in items:
                local_path = it.get("local_file")
                if not local_path:
                    files_spec.append({"path": "", "ok": False, "error": "missing local_file"})
                    continue
                if not os.path.exists(local_path):
                    files_spec.append({"path": "", "ok": False, "error": f"local_file not found: {local_path}"})
                    continue
                remote_path = it.get("remote_file") or os.path.basename(local_path)
                with open(local_path, "rb") as fh:
                    content_b64 = base64.b64encode(fh.read()).decode()
                files_spec.append(
                    {
                        "path": remote_path,
                        "content_b64": content_b64,
                        "makedirs": True,
                        "mode": it.get("mode"),
                    }
                )
        else:
            # Legacy list shape
            files_spec = []
            for item in uploads or []:
                path = item.get("path")
                if not path:
                    files_spec.append({"path": "", "ok": False, "error": "missing_path"})
                    continue
                if "content_b64" in item and item["content_b64"] is not None:
                    content_b64 = item["content_b64"]
                elif "content" in item and item["content"] is not None:
                    content_b64 = base64.b64encode(item["content"].encode("utf-8")).decode()
                elif "local_path" in item and item["local_path"]:
                    with open(item["local_path"], "rb") as fh:
                        content_b64 = base64.b64encode(fh.read()).decode()
                else:
                    files_spec.append({"path": path, "ok": False, "error": "no content provided"})
                    continue
                files_spec.append(
                    {
                        "path": path,
                        "content_b64": content_b64,
                        "makedirs": bool(item.get("makedirs", True)),
                        "mode": item.get("mode"),
                    }
                )

        if any(("ok" in f and f.get("ok") is False) for f in files_spec):
            return {"ok": False, "results": files_spec}

        r = requests.post(f"{self.base_url}/files/upload", json={"session_id": session_identifier, "files": files_spec})
        self._raise_for_status(r)
        return r.json()

    def _handle_file_downloads(self, session_identifier: str, downloads):
        """
        Accepts either:
          - dict form: {"files_to_download": [{"remote_file": "...", "local_file": "..."} , ...]}
              - local_file optional; if omitted, content is returned in-memory (base64)
          - legacy list form: [{"path": "...", dest?: "...", as_text?: bool}]
        """
        if not downloads:
            return {"ok": True, "results": []}

        save_plan = []
        if isinstance(downloads, dict) and "files_to_download" in downloads:
            items = downloads.get("files_to_download") or []
            for it in items:
                remote_path = it.get("remote_file")
                if not remote_path:
                    save_plan.append(("", None, None))
                    continue
                local_dest = it.get("local_file")
                save_plan.append((remote_path, local_dest, None))
        else:
            for it in downloads or []:
                remote_path = it.get("path")
                if not remote_path:
                    save_plan.append(("", None, None))
                    continue
                local_dest = it.get("dest")
                as_text = it.get("as_text")
                save_plan.append((remote_path, local_dest, as_text))

        req_files = [{"path": rp} for (rp, _, _) in save_plan if rp]
        r = requests.post(
            f"{self.base_url}/files/download", json={"session_id": session_identifier, "files": req_files}
        )
        self._raise_for_status(r)
        payload = r.json()

        by_path = {res["path"]: res for res in payload.get("results", []) if "path" in res}
        out_results = []
        for remote_path, local_dest, as_text in save_plan:
            if not remote_path:
                out_results.append({"path": "", "ok": False, "error": "missing remote path"})
                continue

            abs_remote = remote_path if remote_path.startswith("/") else os.path.join("/mnt/data", remote_path)
            res = by_path.get(abs_remote) or by_path.get(remote_path)
            if not res:
                out_results.append({"path": remote_path, "ok": False, "error": "not_returned"})
                continue
            if not res.get("ok"):
                out_results.append(res)
                continue

            if res.get("exists"):
                data_b64 = res.get("content_b64", "")
                data = base64.b64decode(data_b64.encode()) if data_b64 else b""
                if local_dest:
                    os.makedirs(os.path.dirname(local_dest) or ".", exist_ok=True)
                    with open(local_dest, "wb") as fh:
                        fh.write(data)
                    out_results.append({"path": remote_path, "exists": True, "saved_to": local_dest, "size": len(data)})
                else:
                    if as_text:
                        try:
                            text = data.decode("utf-8")
                            out_results.append({"path": remote_path, "exists": True, "text": text, "size": len(data)})
                        except Exception as e:
                            out_results.append(
                                {
                                    "path": remote_path,
                                    "exists": True,
                                    "error": f"utf8_decode_failed: {e}",
                                    "size": len(data),
                                }
                            )
                    else:
                        out_results.append(
                            {"path": remote_path, "exists": True, "content_b64": data_b64, "size": len(data)}
                        )
            else:
                out_results.append({"path": remote_path, "exists": False})

        return {"ok": True, "results": out_results}

    def reset(self):
        r = requests.post(f"{self.base_url}/reset")
        self._raise_for_status(r)
        return r.json()

    def reset_session(self, session_id: str):
        r = requests.post(f"{self.base_url}/reset_session/{session_id}")
        self._raise_for_status(r)
        return r.json()

    @staticmethod
    def _raise_for_status(r: requests.Response):
        if r.status_code >= 400:
            try:
                error_detail = r.json().get("detail", r.text)
            except ValueError:
                error_detail = r.text
            raise requests.exceptions.HTTPError(f"HTTP {r.status_code}: {error_detail}", response=r)
        return r


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _is_pybox_server(host: str, port: int) -> bool:
    """
    Probe /openapi.json and check for our known endpoints.
    Returns True iff the server at host:port looks like this FastAPI app.
    """
    try:
        resp = requests.get(f"http://{host}:{port}/openapi.json", timeout=1.0)
        if resp.status_code != 200:
            return False
        data = resp.json()
        paths = (data or {}).get("paths", {})
        # Core endpoints this app serves
        required = {"/setup", "/exec", "/files/upload", "/files/download"}
        return required.issubset(set(paths.keys()))
    except Exception:
        return False


def start_local_ci_client(
    host: str, port: int, num_pools: int, log_level: str, max_sessions: int = DEFAULT_MAX_SESSIONS
) -> Optional[PyboxClient]:
    """
    Start the FastAPI app with uvicorn and return a ready PyboxClient.
    If another instance is already running on host:port and matches our API, reuse it.
    """
    # Reuse if server is already up and it's our API
    if _port_open(host, port):
        if _is_pybox_server(host, port):
            logger().debug(f"Detected existing local code interpreter at {host}:{port}; reusing it.")
            client = PyboxClient(base_url=f"http://{host}:{port}", max_retries=3)
            try:
                client.setup(max_sessions=max_sessions)
            except requests.exceptions.HTTPError as e:
                if "Already set up" in str(e):
                    logger().debug("Local code interpreter already set up.")
                else:
                    logger().warning(f"Failed to set up local code interpreter client: {e}")
                    return None
            except Exception as e:
                logger().error(f"Failed to talk to existing server: {e}")
                return None
            return client
        else:
            # Port occupied by something else; do not start another uvicorn
            logger().error(f"Port {host}:{port} is in use by a different service; not starting uvicorn.")
            return None

    # Otherwise, start a fresh uvicorn process
    module_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "local_code_interpreter:app",
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        str(num_pools),
        "--log-level",
        log_level,
        "--app-dir",
        module_dir,
    ]
    proc = subprocess.Popen(cmd)

    def _stop_proc():
        if proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass

    atexit.register(_stop_proc)

    # Wait for readiness (up to ~10s)
    start_t = time.time()
    last_err = None
    while time.time() - start_t < 10:
        if proc.poll() is not None:
            logger().error(f"uvicorn exited early with code {proc.returncode}")
            return None
        if _port_open(host, port):
            # Verify it's our API; if not, treat as failure
            if _is_pybox_server(host, port):
                break
            else:
                last_err = "different service detected on target port"
        time.sleep(0.1)
    else:
        _stop_proc()
        logger().error(f"uvicorn failed to start within timeout: {last_err}")
        return None

    client = PyboxClient(base_url=f"http://{host}:{port}", max_retries=3)
    client._uvicorn_proc = proc  # type: ignore[attr-defined]

    try:
        client.setup(max_sessions=max_sessions)
    except requests.exceptions.HTTPError as e:
        if "Already set up" in str(e):
            logger().debug("Local code interpreter client already set up.")
        else:
            logger().warning(f"Failed to set up local code interpreter client: {e}")
            _stop_proc()
            return None
    except Exception as e:
        logger().error(f"Failed to set up local code interpreter client: {e}")
        _stop_proc()
        return None
    return client


if __name__ == "__main__":
    CODE_INTERPRETER = start_local_ci_client(
        host=os.environ.get("CODE_INTERPRETER_HOST", "localhost"),
        port=int(os.environ.get("CODE_INTERPRETER_PORT", 6989)),
        num_pools=int(os.environ.get("CODE_INTERPRETER_NUM_POOLS", 10)),
        log_level=os.environ.get("CODE_INTERPRETER_LOG_LEVEL", "warning"),
    )
    if CODE_INTERPRETER:
        logger().info("Local code interpreter client started successfully.")
    else:
        logger().error("Failed to start local code interpreter client.")

    f = tempfile.NamedTemporaryFile(delete=False, delete_on_close=True, mode="w")
    f.write("This is a test file for the local code interpreter.")
    f.flush()

    files_to_upload = [{"local_file": f.name, "remote_file": f"my_data_file.txt"}]
    file_to_download = [{"remote_file": "downloaded_data.txt", "local_file": "downloads/downloaded_data.txt"}]

    code = dedent(
        """
        import os
        import json
        # print the current working directory
        print("Current working directory:", os.getcwd())
        # read the file we uploaded
        with open("my_data_file.txt", "r") as f:
            data = f.read()
        print("Data from uploaded file:", data)
        # return some JSON data
        result = {"status": "success", "message": "Code executed successfully", "data": data}
        print(json.dumps(result))
        
        with open("downloaded_data.txt", "w") as f:
            f.write("This is the content of the downloaded file.")
    """
    )
    if CODE_INTERPRETER:
        result = asyncio.run(
            CODE_INTERPRETER.execute_code_single_with_retry(
                code, 0, "test_session", timeout=200, upload_files=files_to_upload, download_files=file_to_download
            )
        )
        logger().info(f"Execution result: {result}")
        CODE_INTERPRETER.reset()
        logger().info("Local code interpreter client reset.")
        CODE_INTERPRETER.reset_session("test_session")
        logger().info("Test session reset.")

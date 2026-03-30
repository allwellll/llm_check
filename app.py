import concurrent.futures
import json
import os
import re
import selectors
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
DEFAULT_PROBLEMS = [3743, 3501, 3486, 3435, 3389]
LEETCODE_CLI_VERSION = "0.4.3"
DEFAULT_THINKING_EFFORT = "high"
MODEL_CONNECT_TIMEOUT = 10
MODEL_READ_TIMEOUT = 45
MODEL_STREAM_MAX_SECONDS = 120
CODEX_EXEC_IDLE_TIMEOUT = 300
CODEX_EXEC_MAX_SECONDS = 720
INSTALL_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS: dict[str, "JobState"] = {}


@dataclass
class ProblemResult:
    problem_id: int
    problem_name: str = ""
    starter_file: str = ""
    sample_passed: bool = False
    submit_passed: bool = False
    status: str = "pending"
    error: str = ""
    sample_output: str = ""
    submit_output: str = ""
    model_output_preview: str = ""
    runtime: str = ""
    memory: str = ""


@dataclass
class JobState:
    job_id: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    logs: list[str] = field(default_factory=list)
    results: list[ProblemResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    current_step: str = ""
    current_step_started_at: float = 0.0
    current_problem_id: int | None = None
    current_reasoning: str = ""
    reasoning_events: list[str] = field(default_factory=list)
    problem_reasoning: dict[str, list[str]] = field(default_factory=dict)
    problem_titles: dict[str, str] = field(default_factory=dict)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        self.updated_at = time.time()

    def set_step(self, step: str) -> None:
        self.current_step = step
        self.current_step_started_at = time.time()
        self.updated_at = time.time()

    def start_problem_reasoning(self, problem_id: int, problem_name: str = "") -> None:
        key = str(problem_id)
        self.current_problem_id = problem_id
        if key not in self.problem_reasoning:
            self.problem_reasoning[key] = []
        if problem_name:
            self.problem_titles[key] = problem_name
        self.current_reasoning = self.problem_reasoning[key][-1] if self.problem_reasoning[key] else ""
        self.reasoning_events = self.problem_reasoning[key][-20:]
        self.updated_at = time.time()

    def set_problem_title(self, problem_id: int, problem_name: str) -> None:
        if problem_name:
            self.problem_titles[str(problem_id)] = problem_name
            self.updated_at = time.time()

    def add_reasoning(self, message: str) -> None:
        message = clip_text(message, 4000)
        self.current_reasoning = message
        key = str(self.current_problem_id) if self.current_problem_id is not None else "unknown"
        bucket = self.problem_reasoning.setdefault(key, [])
        if not bucket or bucket[-1] != message:
            bucket.append(message)
        self.reasoning_events = bucket[-20:]
        self.updated_at = time.time()


app = FastAPI(title="LLM LeetCode Benchmark")


def clip_text(value: str, limit: int = 3000) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...<truncated>..."


def set_job(job: JobState) -> None:
    with JOBS_LOCK:
        JOBS[job.job_id] = job


def get_job(job_id: str) -> JobState:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def cargo_bin_dir() -> Path:
    return Path("/root/.cargo/bin")


def resolve_executable(name: str) -> str | None:
    cargo_path = cargo_bin_dir() / name
    if cargo_path.exists():
        return str(cargo_path)
    return shutil.which(name)


def build_exec_env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = f"{cargo_bin_dir()}:{env.get('PATH', '')}"
    return env


def resolve_codex_cli() -> str | None:
    return shutil.which("codex")


def root_codex_dir() -> Path:
    return Path("/root/.codex")


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def ensure_leetcode_cli_installed(job: JobState | None = None) -> Path:
    binary = resolve_executable("leetcode")
    if binary:
        return Path(binary)

    with INSTALL_LOCK:
        binary = resolve_executable("leetcode")
        if binary:
            return Path(binary)

        def maybe_log(message: str) -> None:
            if job:
                job.log(message)

        def install_rustup_toolchain() -> str:
            maybe_log("系统 Rust 版本过旧，切换到 rustup stable 工具链。")
            rustup = run_command(
                [
                    "bash",
                    "-lc",
                    "curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable",
                ],
                timeout=2400,
            )
            if rustup.returncode != 0:
                raise RuntimeError(
                    "Rust 安装失败:\n"
                    f"{clip_text(rustup.stderr or rustup.stdout, 4000)}"
                )
            cargo = resolve_executable("cargo")
            if not cargo:
                raise RuntimeError("Rust 安装完成后仍未找到 cargo。")
            return cargo

        def ensure_cli_source_checkout() -> Path:
            checkout_dir = Path(f"/tmp/leetcode-cli-v{LEETCODE_CLI_VERSION}")
            if checkout_dir.exists():
                return checkout_dir
            maybe_log(f"拉取 leetcode-cli 源码标签 v{LEETCODE_CLI_VERSION}。")
            clone = run_command(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    f"v{LEETCODE_CLI_VERSION}",
                    "https://github.com/clearloop/leetcode-cli",
                    str(checkout_dir),
                ],
                timeout=1800,
            )
            if clone.returncode != 0:
                raise RuntimeError(
                    "拉取 leetcode-cli 源码失败:\n"
                    f"{clip_text((clone.stderr or '') + (clone.stdout or ''), 4000)}"
                )
            return checkout_dir

        maybe_log("未检测到 leetcode-cli，开始安装依赖。")
        run_command(["apt-get", "update"], timeout=1800)
        install_result = run_command(
            [
                "apt-get",
                "install",
                "-y",
                "build-essential",
                "cargo",
                "rustc",
                "pkg-config",
                "libdbus-1-dev",
                "libsqlite3-dev",
                "curl",
            ],
            timeout=1800,
        )
        if install_result.returncode != 0:
            raise RuntimeError(
                "apt 依赖安装失败:\n"
                f"{clip_text(install_result.stderr or install_result.stdout, 4000)}"
            )

        cargo = resolve_executable("cargo")
        if not cargo:
            cargo = install_rustup_toolchain()

        source_dir = ensure_cli_source_checkout()
        maybe_log("通过 cargo 安装 leetcode-cli。")
        cargo_install = run_command(
            [cargo, "install", "--path", str(source_dir), "--locked"],
            env=build_exec_env(Path("/root")),
            timeout=3600,
        )
        combined_output = (cargo_install.stdout or "") + "\n" + (cargo_install.stderr or "")
        if cargo_install.returncode != 0 and "edition2024" in combined_output:
            cargo = install_rustup_toolchain()
            maybe_log("使用 rustup 工具链重新安装 leetcode-cli。")
            cargo_install = run_command(
                [cargo, "install", "--path", str(source_dir), "--locked"],
                env=build_exec_env(Path("/root")),
                timeout=3600,
            )
            combined_output = (cargo_install.stdout or "") + "\n" + (cargo_install.stderr or "")
        if cargo_install.returncode != 0:
            raise RuntimeError(
                "leetcode-cli 安装失败:\n"
                f"{clip_text(combined_output, 4000)}"
            )

    binary = resolve_executable("leetcode")
    if not binary:
        raise RuntimeError("leetcode-cli 安装完成后仍未找到可执行文件。")
    return Path(binary)


def write_leetcode_config(home: Path, csrf_token: str, session_token: str, language: str) -> None:
    config_dir = home / ".leetcode"
    code_dir = config_dir / "code"
    scripts_dir = config_dir / "scripts"
    code_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    config = f"""[code]
editor = "true"
lang = "{language}"
comment_problem_desc = true
comment_leading = "#"
test = true

[cookies]
csrf = "{csrf_token}"
session = "{session_token}"
site = "leetcode.com"

[storage]
cache = "Problems"
code = "code"
root = "~/.leetcode"
scripts = "scripts"
"""
    (config_dir / "leetcode.toml").write_text(config, encoding="utf-8")


def normalize_endpoint(api_url: str, api_type: str) -> str:
    api_url = api_url.strip().rstrip("/")
    normalized_type = normalize_api_type(api_type)
    if normalized_type == "chat_completion":
        return api_url if api_url.endswith("/chat/completions") else api_url + "/chat/completions"
    return api_url if api_url.endswith("/responses") else api_url + "/responses"


def normalize_api_type(api_type: str) -> str:
    value = api_type.strip().lower()
    aliases = {
        "codex": "responses",
        "responseapi": "responses",
        "response_api": "responses",
        "responsesapi": "responses",
        "response api": "responses",
        "responses": "responses",
        "chat_completion": "chat_completion",
        "chat completions": "chat_completion",
        "chat-completion": "chat_completion",
        "chatcompletion": "chat_completion",
    }
    return aliases.get(value, value)


def extract_cookie_value(raw_value: str, cookie_name: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""

    pattern = re.compile(rf"{re.escape(cookie_name)}=([^;\\s]+)")
    match = pattern.search(value)
    if match:
        return match.group(1).strip()

    if ";" in value:
        return value.split(";", 1)[0].strip()

    return value


def extract_text_from_payload(payload: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(payload, str):
        texts.append(payload)
    elif isinstance(payload, dict):
        if isinstance(payload.get("output_text"), str):
            texts.append(payload["output_text"])
        if isinstance(payload.get("text"), str):
            texts.append(payload["text"])
        for value in payload.values():
            texts.extend(extract_text_from_payload(value))
    elif isinstance(payload, list):
        for item in payload:
            texts.extend(extract_text_from_payload(item))
    return texts


def extract_model_text(api_type: str, data: dict[str, Any]) -> str:
    if api_type == "chat_completion":
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            texts = extract_text_from_payload(content)
            if texts:
                return "\n".join(part.strip() for part in texts if part.strip()).strip()

    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()

    output = data.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks).strip()

    texts = [part.strip() for part in extract_text_from_payload(data) if isinstance(part, str) and part.strip()]
    if texts:
        return "\n".join(texts).strip()
    raise RuntimeError(f"无法从模型响应中提取文本: {json.dumps(data)[:1200]}")


def strip_code_fences(content: str) -> str:
    text = content.strip()
    block = re.fullmatch(r"```[a-zA-Z0-9_+-]*\n(.*)\n```", text, re.DOTALL)
    if block:
        return block.group(1).strip() + "\n"
    return text + ("\n" if not text.endswith("\n") else "")


def extract_sse_delta(data: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    if data.get("type") == "response.output_text.delta" and isinstance(data.get("delta"), str):
        chunks.append(data["delta"])

    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    return chunks


def summarize_reasoning_payload(data: dict[str, Any]) -> str | None:
    event_type = data.get("type")
    if isinstance(event_type, str) and "reasoning" in event_type:
        texts = [part.strip() for part in extract_text_from_payload(data) if part.strip()]
        if texts:
            return clip_text(f"{event_type}: {' | '.join(texts)}", 1200)
        return clip_text(f"{event_type}: {json.dumps(data, ensure_ascii=False)}", 1200)

    item = data.get("item")
    if isinstance(item, dict) and item.get("type") == "reasoning":
        content = item.get("summary")
        if content:
            texts = [part.strip() for part in extract_text_from_payload(content) if part.strip()]
            if texts:
                return clip_text(f"{event_type or 'reasoning'}: {' | '.join(texts)}", 1200)
            return clip_text(
                f"{event_type or 'reasoning'}: {json.dumps(content, ensure_ascii=False)}",
                1200,
            )
        encrypted = item.get("encrypted_content")
        if isinstance(encrypted, str):
            return f"{event_type or 'reasoning'}: received encrypted reasoning payload ({len(encrypted)} chars)"
        return clip_text(f"{event_type or 'reasoning'}: {json.dumps(item, ensure_ascii=False)}", 1200)

    response = data.get("response")
    if isinstance(response, dict):
        reasoning = response.get("reasoning")
        if reasoning:
            effort = reasoning.get("effort")
            summary = reasoning.get("summary")
            if summary:
                texts = [part.strip() for part in extract_text_from_payload(summary) if part.strip()]
                if texts:
                    return clip_text(
                        f"{event_type or 'response'}: effort={effort or '-'} | {' | '.join(texts)}",
                        1200,
                    )
            if effort:
                return f"{event_type or 'response'}: reasoning effort={effort}"
            return clip_text(
                f"{event_type or 'response'}: {json.dumps(reasoning, ensure_ascii=False)}",
                1200,
            )

    return None


def extract_sse_text(
    response: requests.Response,
    reasoning_callback: Callable[[str], None] | None = None,
) -> str:
    chunks: list[str] = []
    payloads: list[dict[str, Any]] = []
    plain_errors: list[str] = []
    started_at = time.time()

    for raw_line in response.iter_lines(decode_unicode=True):
        if time.time() - started_at > MODEL_STREAM_MAX_SECONDS:
            raise RuntimeError(f"模型流式响应超时: {MODEL_STREAM_MAX_SECONDS}s")
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            if data_str:
                plain_errors.append(data_str)
            continue
        payloads.append(payload)
        reasoning_summary = summarize_reasoning_payload(payload)
        if reasoning_summary and reasoning_callback:
            reasoning_callback(reasoning_summary)
        chunks.extend(extract_sse_delta(payload))

    text = "".join(chunks).strip()
    if text:
        return text

    for payload in reversed(payloads):
        if isinstance(payload.get("response"), dict):
            try:
                return extract_model_text("responses", payload["response"])
            except Exception:  # noqa: BLE001
                pass
        try:
            return extract_model_text("chat_completion", payload)
        except Exception:  # noqa: BLE001
            continue

    if plain_errors:
        raise RuntimeError(clip_text("\n".join(plain_errors), 4000))

    raise RuntimeError("流式响应中未提取到文本。")


def send_model_request(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    reasoning_callback: Callable[[str], None] | None = None,
) -> str:
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=(MODEL_CONNECT_TIMEOUT, MODEL_READ_TIMEOUT),
            stream=True,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            f"模型请求超时: connect={MODEL_CONNECT_TIMEOUT}s read={MODEL_READ_TIMEOUT}s"
        ) from exc
    content_type = (response.headers.get("content-type") or "").lower()
    if response.status_code >= 400:
        body_preview = response.text[:4000]
        raise RuntimeError(f"模型请求失败: HTTP {response.status_code}\n{clip_text(body_preview, 4000)}")

    if "text/event-stream" in content_type:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                extract_sse_text,
                response,
                reasoning_callback=reasoning_callback,
            )
            try:
                return future.result(timeout=MODEL_STREAM_MAX_SECONDS + 5)
            except concurrent.futures.TimeoutError as exc:
                response.close()
                raise RuntimeError(f"模型流式响应超时: {MODEL_STREAM_MAX_SECONDS}s") from exc

    body_preview = response.text[:4000]
    try:
        data = json.loads(body_preview)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"模型响应不是合法 JSON: {clip_text(body_preview, 4000)}") from exc
    return extract_model_text("responses", data)


def seed_temp_codex_home(temp_home: Path) -> Path:
    source_dir = root_codex_dir()
    target_dir = temp_home / ".codex"
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["config.toml", "auth.json", "version.json", ".personality_migration"]:
        source_file = source_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, target_dir / filename)

    for dirname in ["rules", "skills", "memories"]:
        source_path = source_dir / dirname
        target_path = target_dir / dirname
        if source_path.exists() and source_path.is_dir():
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)

    return target_dir


def build_temp_codex_config(base_text: str, api_url: str, model: str, project_dir: Path) -> str:
    filtered_lines: list[str] = []
    skip_provider_block = False
    in_features_block = False
    features_block_seen = False
    fast_mode_injected = False

    for line in base_text.splitlines():
        stripped = line.strip()

        if stripped == "[model_providers.llm_check_temp]":
            skip_provider_block = True
            continue

        if skip_provider_block:
            if stripped.startswith("[") and stripped.endswith("]"):
                skip_provider_block = False
            else:
                continue

        if stripped == "[features]":
            features_block_seen = True
            in_features_block = True
            filtered_lines.append(line)
            if not fast_mode_injected:
                filtered_lines.append("fast_mode = true")
                fast_mode_injected = True
            continue

        if in_features_block and stripped.startswith("[") and stripped.endswith("]"):
            in_features_block = False

        if re.match(r"^(model_provider|model|model_reasoning_effort|openai_base_url)\s*=", stripped):
            continue

        if in_features_block and re.match(r"^fast_mode\s*=", stripped):
            continue

        filtered_lines.append(line)

    filtered_text = "\n".join(filtered_lines).strip()
    project_path = project_dir.as_posix().replace('"', '\\"')
    api_base_url = api_url.strip().rstrip("/").replace('"', '\\"')
    model_name = model.strip().replace('"', '\\"')

    prefix = "\n".join(
        [
            'model_provider = "llm_check_temp"',
            f'model = "{model_name}"',
            f'model_reasoning_effort = "{DEFAULT_THINKING_EFFORT}"',
        ]
    )
    feature_block = ""
    if not features_block_seen:
        feature_block = "\n".join(
            [
                "[features]",
                "fast_mode = true",
            ]
        )
    provider_block = "\n".join(
        [
            "[model_providers.llm_check_temp]",
            'name = "llm_check_temp"',
            f'base_url = "{api_base_url}"',
            'wire_api = "responses"',
            "requires_openai_auth = true",
            "",
            f'[projects."{project_path}"]',
            'trust_level = "trusted"',
        ]
    )

    parts = [prefix]
    if filtered_text:
        parts.append(filtered_text)
    if feature_block:
        parts.append(feature_block)
    parts.append(provider_block)
    return "\n\n".join(parts) + "\n"


def prepare_temp_codex_home(
    parent_dir: Path,
    api_url: str,
    api_key: str,
    model: str,
    project_dir: Path,
) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    temp_home = parent_dir / f"codex-home-{timestamp}-{uuid.uuid4().hex[:8]}"
    temp_home.mkdir(parents=True, exist_ok=False)
    codex_dir = seed_temp_codex_home(temp_home)

    base_config_path = codex_dir / "config.toml"
    base_config = base_config_path.read_text(encoding="utf-8") if base_config_path.exists() else ""
    base_config_path.write_text(
        build_temp_codex_config(base_config, api_url, model, project_dir),
        encoding="utf-8",
    )

    (codex_dir / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": api_key.strip()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return temp_home


def summarize_codex_event(event: dict[str, Any]) -> str | None:
    event_type = event.get("type")
    if event_type == "thread.started":
        return "codex: thread started"
    if event_type == "turn.started":
        return "codex: turn started"
    if event_type == "turn.completed":
        usage = event.get("usage") or {}
        output_tokens = usage.get("output_tokens")
        if output_tokens is not None:
            return f"codex: turn completed, output_tokens={output_tokens}"
        return "codex: turn completed"

    item = event.get("item")
    if not isinstance(item, dict):
        return None

    item_type = item.get("type")
    if item_type == "agent_message":
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            return clip_text(f"assistant: {text.strip()}", 1200)
        return "assistant: <empty>"
    if item_type == "command_execution":
        command = item.get("command") or "<unknown>"
        status = item.get("status") or "unknown"
        exit_code = item.get("exit_code")
        output = clip_text((item.get("aggregated_output") or "").strip(), 800)
        if status == "in_progress":
            return f"command running: {command}"
        if output:
            return (
                f"command {status} (exit={exit_code}): {command}\n"
                f"{output}"
            )
        return f"command {status} (exit={exit_code}): {command}"
    if item_type == "error":
        message = item.get("message") or "unknown error"
        return clip_text(f"codex error: {message}", 1200)

    return None


def call_model_with_codex_cli(
    home: Path,
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    reasoning_callback: Callable[[str], None] | None = None,
) -> str:
    codex_cli = resolve_codex_cli()
    if not codex_cli:
        raise RuntimeError("未找到 codex CLI，无法通过 codex exec 调用 codex 模式。")

    codex_parent = home / "codex-runtime"
    codex_parent.mkdir(parents=True, exist_ok=True)
    temp_home = prepare_temp_codex_home(codex_parent, api_url, api_key, model, home)
    output_file = temp_home / "codex-last-message.txt"
    command = [
        codex_cli,
        "exec",
        "-m",
        model.strip(),
        "--json",
        "--color",
        "never",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--ephemeral",
        "-o",
        str(output_file),
        prompt,
    ]
    env = os.environ.copy()
    env["HOME"] = str(temp_home)
    env["PATH"] = f"{cargo_bin_dir()}:{env.get('PATH', '')}"
    process: subprocess.Popen[str] | None = None
    last_agent_message = ""
    error_messages: list[str] = []
    final_message = ""
    started_at = time.time()
    last_output_at = started_at
    try:
        process = subprocess.Popen(
            command,
            cwd=str(home),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                now = time.time()
                if now - started_at > CODEX_EXEC_MAX_SECONDS:
                    process.kill()
                    raise RuntimeError(f"codex exec 总耗时超时: {CODEX_EXEC_MAX_SECONDS}s")
                if now - last_output_at > CODEX_EXEC_IDLE_TIMEOUT:
                    process.kill()
                    raise RuntimeError(f"codex exec 空闲超时: {CODEX_EXEC_IDLE_TIMEOUT}s")

                if process.poll() is not None:
                    ready = selector.select(timeout=0)
                    if not ready:
                        break
                else:
                    ready = selector.select(timeout=1)
                    if not ready:
                        continue

                raw_line = process.stdout.readline()
                if raw_line == "":
                    if process.poll() is not None:
                        break
                    continue
                last_output_at = time.time()
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    summary = clip_text(f"codex: {line}", 1200)
                    error_messages.append(summary)
                    if reasoning_callback:
                        reasoning_callback(summary)
                    continue

                summary = summarize_codex_event(event)
                if summary and reasoning_callback:
                    reasoning_callback(summary)

                item = event.get("item")
                if isinstance(item, dict):
                    if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                        last_agent_message = item["text"].strip()
                    if item.get("type") == "error" and isinstance(item.get("message"), str):
                        error_messages.append(item["message"].strip())

            return_code = process.wait(timeout=5)
        finally:
            selector.close()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if output_file.exists():
            final_message = output_file.read_text(encoding="utf-8").strip()
        shutil.rmtree(temp_home, ignore_errors=True)

    if return_code != 0:
        detail = final_message or last_agent_message or "\n".join(error_messages)
        raise RuntimeError(f"codex exec 失败 (exit={return_code})\n{clip_text(detail, 4000)}")

    if final_message:
        return final_message
    if last_agent_message:
        return last_agent_message
    if error_messages:
        raise RuntimeError(clip_text("\n".join(error_messages), 4000))
    raise RuntimeError("codex exec 未返回最终文本。")


def with_reasoning_variants(base_payload: dict[str, Any], api_type: str) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []

    if api_type == "responses":
        variants.append(base_payload | {"reasoning": {"effort": DEFAULT_THINKING_EFFORT}})
        variants.append(base_payload | {"reasoning_effort": DEFAULT_THINKING_EFFORT})
    else:
        variants.append(base_payload | {"reasoning_effort": DEFAULT_THINKING_EFFORT})
        variants.append(base_payload | {"reasoning": {"effort": DEFAULT_THINKING_EFFORT}})

    variants.append(base_payload)
    return variants


def call_model(
    home: Path,
    api_type: str,
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    reasoning_callback: Callable[[str], None] | None = None,
) -> str:
    normalized_type = normalize_api_type(api_type)
    if normalized_type == "responses":
        return call_model_with_codex_cli(
            home,
            api_url,
            api_key,
            model,
            prompt,
            reasoning_callback=reasoning_callback,
        )

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
    }
    system_prompt = (
        "You solve LeetCode problems. Return only the final source code file. "
        "Do not include markdown fences or explanations. Work under a hard 5-minute "
        "deadline. Do not run broad or exhaustive tests; if you verify anything, use "
        "only one or two tiny hand-crafted examples."
    )

    attempts: list[tuple[str, dict[str, Any]]] = []
    if normalized_type == "responses":
        responses_payload = {
            "model": model.strip(),
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "stream": True,
        }
        for payload in with_reasoning_variants(responses_payload, "responses"):
            attempts.append((normalize_endpoint(api_url, "responses"), payload))
    else:
        chat_payload = {
            "model": model.strip(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }
        for payload in with_reasoning_variants(chat_payload, "chat_completion"):
            attempts.append((normalize_endpoint(api_url, "chat_completion"), payload))

    last_error: Exception | None = None
    for endpoint, payload in attempts:
        try:
            return send_model_request(
                endpoint,
                headers,
                payload,
                reasoning_callback=reasoning_callback,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(str(last_error) if last_error else "模型请求失败")


def build_prompt(problem_id: int, starter_code: str) -> str:
    return f"""Solve LeetCode problem {problem_id} using Python 3.

Requirements:
- Return the complete final Python file only.
- Keep the LeetCode class name and method signatures valid.
- Add any imports your solution needs.
- Optimize for correctness first, and respect the problem constraints.
- Work under a hard 5-minute deadline and return the final code as soon as the solution is sound.
- Do not run large, exhaustive, randomized, or stress-test style validations.
- If you verify the logic, use only one or two tiny hand-crafted examples.
- Do not include markdown, explanations, or extra text.

Starter file:
```python
{starter_code}
```"""


def parse_problem_name(output: str) -> str:
    match = re.search(r"\[(\d+)\]\s+(.+?)\s+is on the run", output, re.DOTALL)
    return match.group(2).strip() if match else ""


def parse_runtime_and_memory(output: str) -> tuple[str, str]:
    runtime = ""
    memory = ""
    runtime_match = re.search(r"Runtime:\s*([^\n,]+)", output)
    memory_match = re.search(r"Memory Usage:\s*([^\n,]+)", output)
    if runtime_match:
        runtime = runtime_match.group(1).strip()
    if memory_match:
        memory = memory_match.group(1).strip()
    return runtime, memory


def leetcode_command_failed(run: subprocess.CompletedProcess[str]) -> bool:
    combined = ((run.stdout or "") + "\n" + (run.stderr or "")).lower()
    if run.returncode != 0:
        return True
    return "error:" in combined


def find_single_code_file(home: Path, language: str) -> Path:
    suffix = ".py" if language in {"python", "python3"} else ""
    code_dir = home / ".leetcode" / "code"
    matches = sorted(code_dir.glob(f"*{suffix}"))
    if not matches:
        raise RuntimeError("leetcode edit 后没有生成代码文件。")
    return matches[0]


def run_leetcode_command(
    home: Path,
    args: list[str],
    *,
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    binary = ensure_leetcode_cli_installed()
    env = build_exec_env(home)
    return run_command([str(binary), *args], env=env, timeout=timeout)


def ensure_problem_cache(job: JobState, home: Path) -> None:
    result = run_leetcode_command(home, ["data", "--update"], timeout=1800)
    if leetcode_command_failed(result):
        raise RuntimeError(
            "LeetCode 题库缓存更新失败:\n"
            f"{clip_text((result.stdout or '') + (result.stderr or ''), 4000)}"
        )
    job.log("LeetCode 题库缓存已更新。")


def benchmark_problem(
    job: JobState,
    home: Path,
    api_type: str,
    api_url: str,
    api_key: str,
    model_name: str,
    problem_id: int,
    language: str,
) -> ProblemResult:
    result = ProblemResult(problem_id=problem_id)
    job.start_problem_reasoning(problem_id)
    code_dir = home / ".leetcode" / "code"
    shutil.rmtree(code_dir, ignore_errors=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    job.set_step(f"题目 {problem_id}: 拉取题目模板")
    edit_run = run_leetcode_command(home, ["edit", str(problem_id), "--lang", language], timeout=1800)
    if edit_run.returncode != 0:
        raise RuntimeError(clip_text(edit_run.stderr or edit_run.stdout, 4000))

    result.problem_name = parse_problem_name(edit_run.stdout)
    job.set_problem_title(problem_id, result.problem_name)
    job.start_problem_reasoning(problem_id, result.problem_name)
    code_file = find_single_code_file(home, language)
    result.starter_file = str(code_file)
    starter_code = code_file.read_text(encoding="utf-8")
    prompt = build_prompt(problem_id, starter_code)

    job.log(f"题目 {problem_id} 开始请求模型。")
    job.set_step(f"题目 {problem_id}: 模型请求中，正在 reasoning")
    model_output = call_model(
        home,
        api_type,
        api_url,
        api_key,
        model_name,
        prompt,
        reasoning_callback=job.add_reasoning,
    )
    result.model_output_preview = clip_text(model_output, 2000)
    final_code = strip_code_fences(model_output)
    code_file.write_text(final_code, encoding="utf-8")

    job.log(f"题目 {problem_id} 开始运行 leetcode test。")
    job.set_step(f"题目 {problem_id}: 运行 leetcode test")
    test_run = run_leetcode_command(home, ["test", str(problem_id)], timeout=1800)
    result.sample_output = clip_text((test_run.stdout + "\n" + test_run.stderr).strip(), 4000)
    result.sample_passed = test_run.returncode == 0 and "Wrong Answer" not in test_run.stdout

    job.log(f"题目 {problem_id} 开始运行 leetcode exec。")
    job.set_step(f"题目 {problem_id}: 运行 leetcode exec")
    exec_run = run_leetcode_command(home, ["exec", str(problem_id)], timeout=1800)
    combined_exec = (exec_run.stdout + "\n" + exec_run.stderr).strip()
    result.submit_output = clip_text(combined_exec, 4000)
    result.submit_passed = exec_run.returncode == 0 and "Success" in exec_run.stdout
    result.runtime, result.memory = parse_runtime_and_memory(exec_run.stdout)
    result.status = "accepted" if result.submit_passed else "failed"
    return result


def run_benchmark_job(job: JobState, payload: dict[str, Any]) -> None:
    job.status = "running"
    job.set_step("准备启动任务")
    job.updated_at = time.time()

    try:
        api_type = normalize_api_type(payload["api_type"])
        api_url = payload["api_url"].strip()
        api_key = payload["api_key"].strip()
        model_name = payload["model_name"].strip()
        csrf_token = extract_cookie_value(payload["csrf_token"], "csrftoken")
        session_token = extract_cookie_value(payload["session_token"], "LEETCODE_SESSION")
        raw_problem_ids = payload.get("problem_ids", "")
        language = payload.get("language", "python3").strip() or "python3"

        if not all([api_type, api_url, api_key, model_name, csrf_token, session_token]):
            raise RuntimeError("API 和 LeetCode 所需字段不能为空。")

        problem_ids = [
            int(item.strip())
            for item in raw_problem_ids.split(",")
            if item.strip()
        ] or DEFAULT_PROBLEMS

        job.log("检查 leetcode-cli 环境。")
        job.set_step("检查 leetcode-cli 环境")
        ensure_leetcode_cli_installed(job)

        home = BASE_DIR / "runtime" / "jobs" / job.job_id / "home"
        if home.exists():
            shutil.rmtree(home)
        home.mkdir(parents=True, exist_ok=True)
        write_leetcode_config(home, csrf_token, session_token, language)
        job.set_step("同步 LeetCode 题库缓存")
        ensure_problem_cache(job, home)

        passed = 0
        for problem_id in problem_ids:
            job.log(f"开始处理题目 {problem_id}。")
            job.set_step(f"题目 {problem_id}: 准备开始")
            try:
                problem_result = benchmark_problem(
                    job,
                    home,
                    api_type,
                    api_url,
                    api_key,
                    model_name,
                    problem_id,
                    language,
                )
            except Exception as exc:  # noqa: BLE001
                problem_result = ProblemResult(
                    problem_id=problem_id,
                    status="error",
                    error=str(exc),
                )
                job.log(f"题目 {problem_id} 执行失败: {exc}")

            if problem_result.submit_passed:
                passed += 1
                job.log(f"题目 {problem_id} 提交通过。")
            elif problem_result.status != "error":
                job.log(f"题目 {problem_id} 未通过。")

            job.results.append(problem_result)
            job.updated_at = time.time()

        total = len(problem_ids)
        rate = round((passed / total) * 100, 2) if total else 0.0
        job.summary = {
            "passed": passed,
            "total": total,
            "pass_rate": rate,
            "problem_ids": problem_ids,
        }
        job.status = "finished"
        job.set_step("任务已完成")
        job.log(f"任务完成，通过率 {passed}/{total} = {rate}%。")
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = str(exc)
        job.set_step("任务失败")
        job.log(f"任务失败: {exc}")
    finally:
        job.updated_at = time.time()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {"default_problem_ids": ",".join(str(item) for item in DEFAULT_PROBLEMS)},
    )


@app.post("/api/jobs")
async def create_job(request: Request) -> JSONResponse:
    payload = await request.json()
    api_type = normalize_api_type(payload.get("api_type", ""))
    if api_type not in {"chat_completion", "responses"}:
        raise HTTPException(status_code=400, detail="api_type 仅支持 chat_completion 或 codex")
    payload["api_type"] = api_type

    job = JobState(job_id=uuid.uuid4().hex[:12])
    set_job(job)
    thread = threading.Thread(target=run_benchmark_job, args=(job, payload), daemon=True)
    thread.start()
    return JSONResponse({"job_id": job.job_id, "job_url": f"/jobs/{job.job_id}"})


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str) -> HTMLResponse:
    job = get_job(job_id)
    return TEMPLATES.TemplateResponse(request, "job.html", {"job": job})


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    job = get_job(job_id)
    return JSONResponse(asdict(job))


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/run")
async def legacy_run_redirect() -> RedirectResponse:
    return RedirectResponse("/", status_code=302)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=1024, reload=False)

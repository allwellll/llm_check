import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
DEFAULT_PROBLEMS = [3743, 3501, 3486, 3435, 3389]
LEETCODE_CLI_VERSION = "0.4.3"
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

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
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


def extract_sse_text(response: requests.Response) -> str:
    chunks: list[str] = []
    payloads: list[dict[str, Any]] = []

    for raw_line in response.iter_lines(decode_unicode=True):
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
            continue
        payloads.append(payload)
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

    raise RuntimeError("流式响应中未提取到文本。")


def send_model_request(endpoint: str, headers: dict[str, str], payload: dict[str, Any]) -> str:
    response = requests.post(endpoint, headers=headers, json=payload, timeout=300, stream=True)
    content_type = (response.headers.get("content-type") or "").lower()
    if response.status_code >= 400:
        body_preview = response.text[:4000]
        raise RuntimeError(f"模型请求失败: HTTP {response.status_code}\n{clip_text(body_preview, 4000)}")

    if "text/event-stream" in content_type:
        return extract_sse_text(response)

    body_preview = response.text[:4000]
    try:
        data = json.loads(body_preview)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"模型响应不是合法 JSON: {clip_text(body_preview, 4000)}") from exc
    return extract_model_text("responses", data)


def call_model(api_type: str, api_url: str, api_key: str, model: str, prompt: str) -> str:
    normalized_type = normalize_api_type(api_type)
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
    }
    system_prompt = (
        "You solve LeetCode problems. Return only the final source code file. "
        "Do not include markdown fences or explanations."
    )

    attempts: list[tuple[str, dict[str, Any]]] = []
    if normalized_type == "responses":
        attempts.append(
            (
                normalize_endpoint(api_url, "responses"),
                {
                    "model": model.strip(),
                    "instructions": system_prompt,
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt}],
                        }
                    ],
                    "stream": True,
                },
            )
        )
        attempts.append(
            (
                normalize_endpoint(api_url, "chat_completion"),
                {
                    "model": model.strip(),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": True,
                },
            )
        )
    else:
        attempts.append(
            (
                normalize_endpoint(api_url, "chat_completion"),
                {
                    "model": model.strip(),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": True,
                },
            )
        )

    last_error: Exception | None = None
    for endpoint, payload in attempts:
        try:
            return send_model_request(endpoint, headers, payload)
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
    if result.returncode != 0:
        raise RuntimeError(
            "LeetCode 题库缓存更新失败:\n"
            f"{clip_text(result.stderr or result.stdout, 4000)}"
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
    code_dir = home / ".leetcode" / "code"
    shutil.rmtree(code_dir, ignore_errors=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    edit_run = run_leetcode_command(home, ["edit", str(problem_id), "--lang", language], timeout=1800)
    if edit_run.returncode != 0:
        raise RuntimeError(clip_text(edit_run.stderr or edit_run.stdout, 4000))

    result.problem_name = parse_problem_name(edit_run.stdout)
    code_file = find_single_code_file(home, language)
    result.starter_file = str(code_file)
    starter_code = code_file.read_text(encoding="utf-8")
    prompt = build_prompt(problem_id, starter_code)

    job.log(f"题目 {problem_id} 开始请求模型。")
    model_output = call_model(api_type, api_url, api_key, model_name, prompt)
    result.model_output_preview = clip_text(model_output, 2000)
    final_code = strip_code_fences(model_output)
    code_file.write_text(final_code, encoding="utf-8")

    job.log(f"题目 {problem_id} 开始运行 leetcode test。")
    test_run = run_leetcode_command(home, ["test", str(problem_id)], timeout=1800)
    result.sample_output = clip_text((test_run.stdout + "\n" + test_run.stderr).strip(), 4000)
    result.sample_passed = test_run.returncode == 0 and "Wrong Answer" not in test_run.stdout

    job.log(f"题目 {problem_id} 开始运行 leetcode exec。")
    exec_run = run_leetcode_command(home, ["exec", str(problem_id)], timeout=1800)
    combined_exec = (exec_run.stdout + "\n" + exec_run.stderr).strip()
    result.submit_output = clip_text(combined_exec, 4000)
    result.submit_passed = exec_run.returncode == 0 and "Success" in exec_run.stdout
    result.runtime, result.memory = parse_runtime_and_memory(exec_run.stdout)
    result.status = "accepted" if result.submit_passed else "failed"
    return result


def run_benchmark_job(job: JobState, payload: dict[str, Any]) -> None:
    job.status = "running"
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
        ensure_leetcode_cli_installed(job)

        home = BASE_DIR / "runtime" / "jobs" / job.job_id / "home"
        if home.exists():
            shutil.rmtree(home)
        home.mkdir(parents=True, exist_ok=True)
        write_leetcode_config(home, csrf_token, session_token, language)
        ensure_problem_cache(job, home)

        passed = 0
        for problem_id in problem_ids:
            job.log(f"开始处理题目 {problem_id}。")
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
        job.log(f"任务完成，通过率 {passed}/{total} = {rate}%。")
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = str(exc)
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
        raise HTTPException(status_code=400, detail="api_type 仅支持 chat_completion 或 responses")
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

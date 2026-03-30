# LLM LeetCode Benchmark

在 `1024` 端口启动一个网页服务，输入模型配置和 LeetCode Cookie 后，服务会：

1. 自动检查并安装 `leetcode-cli`
2. 使用 `leetcode edit` 拉取题目模板
3. 调用配置的大模型生成代码
4. 运行 `leetcode test` 和 `leetcode exec`
5. 统计逐题结果和总通过率

## 启动

```bash
python3 app.py
```

或：

```bash
uvicorn app:app --host 0.0.0.0 --port 1024
```

打开 `http://127.0.0.1:1024/`。

## 工作方式

- 默认测试题目：`3743,3501,3486,3435,3389`
- 支持的模型接口类型：
  - `chat_completion`
  - `responses`
- `responses` 类型默认通过 `codex exec` 调用模型，不直接在服务内手写 `POST /responses`
- `chat_completion` 类型仍然使用普通 HTTP 请求
- 模型请求默认使用 `thinking effort = high`
- 首次运行如果机器没有 Rust / `leetcode-cli`，会自动安装兼容版 `leetcode-cli v0.4.3`

## Responses 模式

当网页里选择 `api_type = responses` 时，服务会：

1. 在当前任务目录下创建一个带时间戳的临时 `HOME`
2. 复制 `/root/.codex` 的基础配置到临时 `HOME/.codex`
3. 用网页填写的 `api_url`、`model_name`、`api_key` 覆盖临时 `config.toml` 和 `auth.json`
4. 用这个临时 `HOME` 执行 `codex exec`
5. 读取 `codex exec --json` 的实时事件流，渲染到网页
6. 读取 `-o` 输出的最终 last message，作为最终代码结果
7. 任务结束后删除整个临时 `HOME`

这样做的目的是：

- 让 `responses` 模式和 Codex CLI 的真实调用方式保持一致
- 能在网页上看到实时的 CLI / reasoning 输出
- 不污染系统默认的 `/root/.codex`

## 网页显示

- 任务页会显示当前阶段、逐题结果、执行日志
- `Reasoning / CLI Events` 会按题目隔离：
  - 当前题单独显示
  - 历史题以折叠面板保留
- 如果 `codex exec` 长时间没有新输出，会按空闲超时中断；只要持续有新输出，就不会因为总耗时短时间内被杀掉

## Cookie

- `csrftoken` 和 `LEETCODE_SESSION` 可以填写纯 token 值
- 也可以直接粘贴完整的 `csrftoken=...` 或 `LEETCODE_SESSION=...; Domain=...`，服务会自动提取

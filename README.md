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

## 说明

- 默认测试题目：`3743,3501,3486,3435,3389`
- 支持的模型接口类型：
  - `chat_completion`
  - `responses`
- 首次运行如果机器没有 Rust / `leetcode-cli`，会自动安装兼容版 `leetcode-cli v0.4.3`，耗时会更长
- `csrftoken` 和 `LEETCODE_SESSION` 需要填写纯 token 值，不要带 `key=` 前缀

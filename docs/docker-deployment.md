# FlowPilot Docker 部署指南

FlowPilot 是基于 FastAPI 和 LangGraph 的 AI Workflow Agent。当前 Docker 部署使用 Docker Compose 管理一个 FastAPI 应用容器，通过 Docker Named Volume 保存 SQLite 数据，并使用 `/health` 进行容器健康检查。

Compose 只将服务暴露到本机 `127.0.0.1:8000`。当前方案面向本地开发、验证和单实例运行，不代表已经完成生产级高可用或公网部署。

## 1. 环境准备

Windows 环境建议准备：

- Docker Desktop，并使用 Linux Containers。
- Docker Compose V2。
- Git。
- Windows PowerShell。
- 可访问 Docker Hub，或已缓存项目所需基础镜像的 Docker 环境。

在 PowerShell 中确认 Docker 可用：

```powershell
docker --version
docker compose version
docker info
```

后续命令均应在 FlowPilot 仓库根目录执行。下面的 `<你的 FlowPilot 仓库目录>` 是路径占位符，使用前必须替换为本机仓库的实际目录，不能原样执行：

```powershell
cd "<你的 FlowPilot 仓库目录>"
```

## 2. 环境变量

应用配置由 `app/core/config.py` 中的 Pydantic Settings 读取，Compose 负责把宿主机变量注入容器。

| 变量 | 必填 | 默认值 | 用途 |
| --- | --- | --- | --- |
| `DEEPSEEK_API_KEY` | 是 | 无 | DeepSeek API 身份凭据；未设置时 Compose 拒绝启动 |
| `DEEPSEEK_BASE_URL` | 否 | `https://api.deepseek.com` | DeepSeek OpenAI-compatible API 地址 |
| `DEEPSEEK_MODEL` | 否 | `deepseek-v4-flash` | 默认模型名称 |
| `MCP_SERVERS` | 否 | `[]` | MCP Server 配置的 JSON 数组；空数组禁用外部 MCP Server |

`MCP_SERVERS` 必须是单行合法 JSON。最安全的禁用配置是：

```text
MCP_SERVERS=[]
```

### 2.1 PowerShell 临时变量

变量只在当前 PowerShell 进程及其子进程中有效：

```powershell
$env:DEEPSEEK_API_KEY = "replace-with-your-key"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
$env:DEEPSEEK_MODEL = "deepseek-v4-flash"
$env:MCP_SERVERS = "[]"
```

### 2.2 本地 `.env` 文件

复制安全模板，然后只在本机编辑：

```powershell
Copy-Item .env.example .env
```

`.env` 已被 Git 和 Docker 构建上下文忽略。仍需注意：

- 不要把真实 API Key 提交到 GitHub。
- 不要把 `.env` 复制进镜像或发送给他人。
- 不要公开包含完整变量值的 `docker compose config` 输出。
- `ci-test-key` 等占位值只适用于离线启动和 `/health` 验证；真实 LLM 功能需要有效的 DeepSeek API Key。
- `.env` 和普通容器环境变量不是生产级 Secret 管理方案。

## 3. 构建和启动

确认 Docker Engine 正常后，执行：

```powershell
docker compose config --quiet
docker compose up --build -d
```

如果没有提供 `DEEPSEEK_API_KEY`，第一条命令会返回非零退出码并提示该变量必须设置。

查看服务状态：

```powershell
docker compose ps
```

容器刚启动时 Healthcheck 可能暂时显示 `starting`。应用启动并通过探针后会变为 `healthy`。

检查 API：

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

正常响应为 HTTP 200，并包含：

```json
{"status":"ok","service":"FlowPilot"}
```

查看日志：

```powershell
docker compose logs --tail=100
docker compose logs -f flowpilot-api
```

停止并移除容器与 Compose 网络：

```powershell
docker compose down
```

普通 `docker compose down` 不删除 Named Volume。不要把 `docker compose down -v` 当作日常停止命令，因为 `-v` 会删除 Compose 管理的数据卷。

## 4. Healthcheck

Docker 在容器内部请求：

```text
http://127.0.0.1:8000/health
```

探针使用镜像已有的 Python 标准库，检查 HTTP 200 和预期 JSON 内容，不调用 DeepSeek、MCP 或其他外部服务。

当前参数：

| 参数 | 值 |
| --- | --- |
| `interval` | `30s` |
| `timeout` | `5s` |
| `retries` | `3` |
| `start_period` | `20s` |

- `healthy` 表示最近的 Docker 探针成功。
- `starting` 表示仍在初始等待或尚未完成首次成功检查。
- `unhealthy` 表示连续失败达到重试阈值。

`unhealthy` 不等于容器会自动重启；当前 Compose 没有配置自动恢复策略。

查看探针状态和最近结果：

```powershell
docker compose ps
$containerId = docker compose ps -q flowpilot-api
docker inspect --format '{{json .State.Health}}' $containerId
```

## 5. 日志管理

服务使用 Docker 原生日志配置：

| 配置 | 值 |
| --- | --- |
| driver | `json-file` |
| max-size | `10m` |
| max-file | `3` |

该配置限制 Docker 容器标准输出日志文件的大小和保留数量，避免单个日志无限增长。它不会限制 SQLite 数据库大小，也不代表已部署 ELK、Loki、Prometheus 或其他外部监控系统。

检查实际配置：

```powershell
$containerId = docker compose ps -q flowpilot-api
docker inspect --format '{{json .HostConfig.LogConfig}}' $containerId
```

当前自动化验收验证了日志参数已应用，但没有通过制造大量日志实际触发轮转。

## 6. SQLite 持久化

Compose 配置：

```yaml
volumes:
  - flowpilot_data:/app/data
```

容器内实际路径：

| 数据 | 路径 |
| --- | --- |
| Agent Run、Side-effect Ledger、Execution Trace | `/app/data/flowpilot.db` |
| LangGraph Checkpoint | `/app/data/checkpoints.sqlite` |

整个 `/app/data` 目录被挂载，因此 SQLite 可能创建的 `-wal` 和 `-shm` 文件也位于同一 Volume 中。

Named Volume 使数据独立于某一个容器层：删除并重建容器后，数据仍可保留。普通 `docker compose down` 会保留 Volume；使用 `down -v`、`docker volume prune` 或 `docker system prune --volumes` 可能造成数据丢失，不应作为常规维护方式。

只读列出 Volume：

```powershell
docker volume ls
```

Docker Volume 不是数据库备份。可靠备份需要考虑 SQLite 事务和 WAL 一致性；不要把复制正在写入的数据库文件描述为可靠备份。当前项目尚未提供生产级数据库迁移和自动备份流程。

当前部署只支持单实例应用容器。不要让多个应用副本共享同一个 SQLite Volume。需要多实例扩展时，应重新评估数据库架构。

## 7. Docker 自动化集成测试

运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test_docker.ps1
```

脚本依次验证：

1. Docker CLI、Engine、Compose、仓库文件和端口预检查。
2. Compose 配置解析。
3. Docker 镜像构建。
4. 容器启动。
5. Docker 状态达到 `healthy`。
6. FastAPI `/health` 响应内容。
7. 容器用户不是 root。
8. `/app/data` 中的 SQLite 数据库可写。
9. 实际 Docker 日志驱动和轮转参数。
10. 删除并重建容器后 SQLite 记录仍存在且可继续写入。
11. 本次测试容器和网络清理。

成功时退出码为 `0`，任何关键检查失败或清理失败时返回非零退出码。脚本使用 `flowpilot13ftest<随机标识>` 形式的独立 Compose Project，不复用正常部署或既有测试 Volume。

测试注意事项：

- 脚本使用安全占位 Key、不可路由的模型地址和空 MCP 配置，不执行真实模型或外部 MCP 调用。
- 测试会临时占用宿主机 8000 端口；如果端口已占用，脚本会安全失败，不会停止其他服务。
- 测试结束会删除本次容器和网络，但保留测试 Named Volume，因此 Docker Volume 数量可能增加。
- 不要在不确认归属和内容的情况下删除旧测试 Volume。
- 测试没有验证真实 DeepSeek 调用，也没有验证 LangGraph Checkpoint 的业务级跨容器恢复。

## 8. 故障排查

### 8.1 Docker Desktop 未启动或无法连接 Engine

检查 Docker Desktop 状态以及：

```powershell
docker info
docker context show
```

等待 Docker Desktop 完成启动后重试。不要通过删除数据卷来处理 Engine 连接问题。

### 8.2 `docker compose up` 失败

先检查配置和日志：

```powershell
docker compose config --quiet
docker compose ps -a
docker compose logs --tail=100
```

保留原始失败退出码和安全日志进行分析，不要追加 `|| true` 或忽略错误。

### 8.3 `DEEPSEEK_API_KEY` 未配置

Compose 会提示 `DEEPSEEK_API_KEY must be set`。在当前 PowerShell 会话安全设置变量，或创建本地 `.env`：

```powershell
$env:DEEPSEEK_API_KEY = "replace-with-your-key"
```

不要把真实值写进 Compose、测试脚本或提交记录。

### 8.4 本地 8000 端口被占用

检查监听者：

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
```

确认进程归属后自行决定如何处理。不要让自动化测试擅自终止未知进程，也不要把正式端口改为公网绑定。

### 8.5 容器长时间处于 `starting`

Healthcheck 最初包含启动缓冲时间。检查：

```powershell
docker compose ps
docker compose logs --tail=100 flowpilot-api
```

如果应用仍在初始化，可以等待下一次探针；若持续不变，再检查容器健康详情。

### 8.6 容器持续 `unhealthy`

```powershell
$containerId = docker compose ps -q flowpilot-api
docker inspect --format '{{json .State.Health}}' $containerId
docker compose logs --tail=100 flowpilot-api
```

重点检查 Uvicorn 是否启动、端口是否监听以及 `/health` 是否返回预期 JSON。`unhealthy` 本身不会自动重启容器。

### 8.7 `/health` 无法访问

```powershell
docker compose ps
curl.exe -i http://127.0.0.1:8000/health
docker compose logs --tail=100 flowpilot-api
```

确认访问的是本机回环地址、容器处于运行状态且端口映射仍为 `127.0.0.1:8000:8000`。

### 8.8 SQLite 数据目录权限不足

检查容器身份、目录和挂载：

```powershell
docker compose exec -T flowpilot-api id
docker compose exec -T flowpilot-api ls -ld /app/data
$containerId = docker compose ps -q flowpilot-api
docker inspect --format '{{json .Mounts}}' $containerId
```

容器应以非 root 用户运行。不要使用 `chmod 777`、privileged 模式或删除数据库作为默认修复方案。

### 8.9 Docker Hub 镜像拉取失败

```powershell
docker pull python:3.11-slim
```

检查网络、DNS、Docker Desktop 代理配置和镜像仓库可用性。不要把私人代理地址写入项目文档或提交到仓库。

### 8.10 Docker 自动化测试返回 FAIL

脚本会标记具体失败项并尽可能输出最近的安全容器日志。检查最终报告、失败项和退出码：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test_docker.ps1
$LASTEXITCODE
```

不要只依据打印文本判断成功；成功必须同时满足总体 PASS 和退出码 `0`。

### 8.11 pytest 报告历史 `.pytest_tmp_*` 权限错误

将 `--basetemp` 指向当前用户明确可写、且不包含重要数据的独立目录：

```powershell
pytest -q --basetemp="$env:TEMP\pytest_flowpilot_manual"
```

不要递归删除来源不明或不确认归属的目录。权限警告与测试断言失败需要分别判断。

### 8.12 Git 提示 `LF will be replaced by CRLF`

这是 Git 行尾转换提示，不等同于代码失败。检查仓库和本机配置：

```powershell
git status --short
git config --get core.autocrlf
```

不要为了消除提示而批量改写整个仓库的行尾。

## 9. 当前部署能力边界

Stage 13 已实现：

- Python 3.11 Dockerfile。
- Docker Compose 单实例服务编排。
- 基础环境变量配置和必填 Key 校验。
- 非 root 容器运行。
- SQLite Named Volume。
- Docker Healthcheck。
- Docker `json-file` 日志轮转配置。
- 可重复执行的 Docker 自动化集成测试。

尚未实现或尚未完成验收：

- 生产环境 HTTPS 和反向代理。
- 公网部署安全加固。
- 多实例高可用和容器编排集群。
- 独立生产监控平台。
- 云端 Secret 管理。
- 生产级数据库迁移和自动备份。
- LangGraph Checkpoint 的业务级跨容器恢复验收。
- 日志轮转的真实大文件触发测试。

本地容器能够成功启动，不等同于已经完成生产部署。

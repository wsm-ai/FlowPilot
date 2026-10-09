# FlowPilot 安全指南

本文档描述 FlowPilot Stage 14A–14F 已实现并经过自动化测试的安全边界。它面向维护者和部署人员，不代表项目已经通过第三方安全认证，也不等同于完整的生产公网部署方案。

## 1. 安全架构

FlowPilot 使用分层安全边界：

```text
HTTP 请求
  → 请求体大小限制与安全响应头
  → API Key 身份认证
  → RBAC 路由权限检查
  → FastAPI Schema 校验
  → Agent / Planner / HITL 服务
  → ToolRegistry 风险与参数授权
  → Side-effect Ledger
  → 本地或 MCP 工具
```

- `app/security/api_key.py` 验证 Bearer API Key，并把服务端解析出的可信角色放入请求上下文。
- `app/security/rbac.py` 集中定义角色、权限和真实 HTTP 路由映射；未分类 HTTP 路径默认仅管理员可访问。
- `app/security/http.py` 在业务解析前限制请求体，并为响应添加安全头。
- `app/security/redaction.py` 在日志 Handler 输出前脱敏已配置密钥和常见敏感字段。
- `app/security/tool_authorization.py` 对工具风险、调用身份和可信审批 Grant 进行授权。
- `app/graph/execution_nodes.py` 在审批执行前核对工作流身份；`app/tools/registry.py` 在最终工具调用边界再次检查授权和参数摘要。
- Side-effect Ledger 在副作用 dispatch 前持久化 STARTED 状态，阻止危险的重复执行。

安全日志、Execution Trace、Run Repository、LangGraph Checkpoint 和 Side-effect Ledger 是不同的数据边界。日志脱敏不会自动加密数据库中的业务数据。

## 2. API Key 配置

| 环境变量 | 作用 | 默认值 |
| --- | --- | --- |
| `FLOWPILOT_AUTH_ENABLED` | 是否启用 FlowPilot 入站 HTTP 认证 | `true` |
| `FLOWPILOT_API_KEY` | 兼容旧版的单一管理员 Key | 无 |
| `FLOWPILOT_API_KEYS` | 多 Key 与角色绑定的 JSON 数组 | `[]` |
| `DEEPSEEK_API_KEY` | FlowPilot 调用 DeepSeek 的出站凭据 | 无 |

`FLOWPILOT_API_KEY`/`FLOWPILOT_API_KEYS` 用于客户端访问 FlowPilot；`DEEPSEEK_API_KEY` 仅用于 FlowPilot 调用模型服务。两类凭据不能互换。

认证启用时，如果单 Key 和多角色 Key 均未配置，Settings 校验会使应用在启动阶段失败。单 Key 保持向后兼容并固定映射为 `admin`。多角色配置示例：

```powershell
$env:FLOWPILOT_AUTH_ENABLED = "true"
$env:FLOWPILOT_API_KEY = "replace-with-a-strong-admin-key"
$env:FLOWPILOT_API_KEYS = '[{"key":"replace-with-operator-key","role":"operator"},{"key":"replace-with-viewer-key","role":"viewer"}]'
$env:DEEPSEEK_API_KEY = "replace-with-your-deepseek-key"
```

示例值必须替换，不可作为生产凭据。调用受保护 API 时使用：

```http
Authorization: Bearer <API_KEY>
```

本地离线测试可以显式提供虚构 Key，但生产环境不得关闭认证。API Key 不应放入 URL、日志、镜像、Git 提交或公开的 Compose 输出。当前系统尚未实现 API Key 自动轮换、吊销和请求限流。

## 3. RBAC 角色与权限

权限以服务端验证后的 API Key 为依据。`X-Role`、请求 JSON 或 Query 参数不能指定或提升角色。

| 路由或入口 | admin | operator | viewer | 匿名 |
| --- | --- | --- | --- | --- |
| `GET /health` | 允许 | 允许 | 允许 | 允许 |
| `POST /api/v1/chat` | 允许 | 允许 | 拒绝 | 401 |
| `POST /api/v1/agent/run` | 允许 | 允许 | 拒绝 | 401 |
| `POST /api/v1/agent/plan-run` | 允许 | 允许 | 拒绝 | 401 |
| `POST /api/v1/agent/answer/retry` | 允许 | 允许 | 拒绝 | 401 |
| `POST /api/v1/agent/approval/resume` | 允许 | 拒绝 | 拒绝 | 401 |
| `/docs`、`/redoc`、`/openapi.json` | 允许 | 拒绝 | 拒绝 | 401 |
| `/mcp` 及其子路径 | 允许 | 拒绝 | 拒绝 | 401 |
| 未明确分类的其他 HTTP 路径 | 允许 | 拒绝 | 拒绝 | 401 |

当前没有面向 viewer 的只读 Run 或 Trace HTTP API，因此 viewer 没有受保护业务权限。RBAC 只实现角色级控制，尚未实现“只能查看自己的资源”、多租户隔离或对象级授权。

认证失败返回 `401`；身份已验证但权限不足返回 `403`。权限检查发生在业务依赖和服务调用之前。

## 4. API 输入安全

| 输入或限制 | 当前值 |
| --- | --- |
| `message` 最大长度 | 10,000 字符 |
| `goal` 最大长度 | 10,000 字符 |
| `run_id` 最大长度 | 200 字符 |
| `thread_id` 最大长度 | 200 字符 |
| 默认 HTTP 请求体上限 | 1,048,576 字节（1 MiB） |
| 可配置的请求体上限最大值 | 16,777,216 字节（16 MiB） |

请求模型拒绝空白必填文本、错误数据类型和额外 JSON 字段。正常中文、代码和 Markdown 文本仍可作为合法业务输入。

HTTP 状态含义：

- `400`：非法或含糊的 `Content-Length`。
- `401`：API Key 缺失或无效。
- `403`：身份有效但角色权限不足。
- `413`：声明大小、实际流式读取大小或分块消息数量超过安全边界。
- `422`：请求 JSON 不符合 Pydantic Schema。

请求体中间件不只依赖 `Content-Length`：缺少或伪造该 Header 时，仍会累计实际收到的字节数。分块消息使用有界队列，异常大量的微小或空分块也会被拒绝。合法请求体会按原始 ASGI 消息顺序回放给 FastAPI。

## 5. 日志与敏感信息保护

日志保护覆盖：

- `FLOWPILOT_API_KEY` 和 `FLOWPILOT_API_KEYS` 中的 Key；
- `DEEPSEEK_API_KEY`；
- 已配置的 MCP Bearer Token；
- Authorization/Bearer、password、secret、access token、refresh token 等结构化字段；
- 嵌套字典、列表、元组和 Pydantic `SecretStr`；
- `logger.exception()` 的异常文本和 traceback；
- FlowPilot 子 Logger 传播到目标 Handler 的记录。

统一替换标记为 `[REDACTED]`。Uvicorn access logger 保留其参数结构和可读格式。应用仍记录安全的事件类型、阶段、结果、状态码和失败分类等排障信息，不记录完整 Authorization Header、原始 Prompt、完整模型响应或任意工具参数。

自动脱敏只能识别已配置秘密和已知字段/文本模式，不能保证识别用户自由文本中的所有个人信息或未知秘密。因此首要策略仍是避免记录不必要的原始数据。

Execution Trace 只持久化白名单事件字段。Run Repository、Checkpoint 和 Ledger 保存的是业务恢复或审计数据；它们当前不具备静态加密、自动保留周期或数据删除策略，不能因日志已脱敏而视为已加密。

## 6. MCP 与工具执行安全

工具注册时使用以下风险类型：

| 风险类型 | 规则 |
| --- | --- |
| `READ_ONLY` | 允许可信内部调用以及 admin/operator；viewer 拒绝 |
| `SIDE_EFFECT` | 必须具有可信审批 Grant；HTTP 身份必须为 admin |
| `HIGH_RISK` | 与 SIDE_EFFECT 相同的严格审批要求 |
| `UNCLASSIFIED` | 始终拒绝，包括已有审批上下文时 |

动态发现的 MCP 工具默认按 `HIGH_RISK` 注册。Reactive Agent 使用受限 Registry，不能看到需要审批的 MCP 工具；Planned/HITL 路径使用完整 Registry，并强制高风险动作经过审批。

可信审批由服务端审批流程在验证持久化 Run 和 LangGraph Checkpoint 后创建。Grant 绑定：

- `run_id`
- `thread_id`
- `step_id`
- `action`
- 使用稳定 JSON 序列化计算的 `arguments_digest`

LLM 输出、Prompt、工具名称或客户端伪造的 AgentState 都不能创建可信 Grant。Approved Executor 验证完整工作流身份，ToolRegistry 在最终调用边界再次对实际 action、风险和参数摘要授权。任何授权失败均发生在 Side-effect Ledger reservation 和真实工具 dispatch 之前。

Side-effect Ledger 使用运行身份、步骤、动作和参数摘要防止重复副作用：已完成操作返回持久化结果；STARTED 或 AMBIGUOUS 操作禁止自动重放；参数冲突也会拒绝。它不等同于分布式事务或远程系统的 exactly-once 保证。

FlowPilot 无法自动证明所有远程 MCP 工具都是安全的。启用远程 MCP Server 前仍需审查服务身份、传输安全、工具语义和权限范围。

## 7. Docker 安全部署

完整操作步骤参见 [Docker 部署指南](docker-deployment.md)。当前 Compose 安全特性包括：

- API Key 通过运行环境注入，不写入镜像；
- 主机端口仅绑定 `127.0.0.1:8000`；
- 容器使用非 root 用户 `flowpilot`；
- `/health` Docker Healthcheck 不调用外部服务；
- Docker `json-file` 日志限制为 `10m`、保留 `3` 个文件；
- `/app/data` 使用 Named Volume 保存 SQLite 数据；
- `.env`、数据库、日志和测试文件不进入镜像构建上下文。

本地 `.env` 不应提交 Git。Docker 集成测试使用虚构凭据，并且不会调用真实 DeepSeek 或远程 MCP 工具。

当前部署仍是单实例本地方案，尚未实现生产 HTTPS、反向代理、公网边界防护、云端 Secret 管理、自动备份或高可用部署。

## 8. 安全回归测试

Stage 14 安全测试覆盖：

- API Key、重复/畸形 Authorization Header 和启动配置；
- RBAC、未知路由默认拒绝以及未授权零业务调用；
- Schema、Content-Length、实际流式请求体和分块资源限制；
- 密钥、结构化参数、子 Logger、异常和 Uvicorn 日志脱敏；
- 工具风险分类、角色、可信 Grant、参数摘要和并发 ContextVar 隔离；
- HITL resume、MCP、Checkpoint 和 Side-effect Ledger 组合行为；
- Docker 认证、非 root、日志、健康检查和 SQLite 持久化。

测试使用 Fake、Mock、临时 SQLite 和虚构凭据，不访问真实 DeepSeek、GitHub 或远程 MCP 服务。

## 9. 已知限制与后续方向

| 限制 | 影响 | 建议方向 |
| --- | --- | --- |
| `requirements.txt` 未精确锁定版本 | 构建可复现性和供应链审计能力有限 | 建立经审核的 lock/constraints 和更新流程 |
| 尚未执行漏洞数据库 CVE 扫描 | 不能证明依赖不存在已知漏洞 | 在 CI 中接入受控的依赖审计 |
| 仅有角色级 RBAC | 不能提供资源所有权和租户隔离 | 增加主体、资源和租户级授权模型 |
| API Key 无自动轮换、吊销和限流 | 长期凭据泄露和滥用风险需由部署侧控制 | 引入凭据生命周期与请求限流 |
| SQLite 业务数据无静态加密和保留策略 | 磁盘或备份暴露可能泄露业务数据 | 设计加密、备份、保留和删除流程 |
| 自动日志脱敏能力有限 | 未知格式的秘密可能无法识别 | 坚持数据最小化并持续扩展安全字段策略 |
| 远程 MCP 信任依赖部署审查 | 恶意或被攻陷的 MCP 服务仍构成风险 | 建立服务允许列表、身份验证和工具审核 |
| 尚无生产 HTTPS/公网防护 | 当前配置不适合直接暴露公网 | 部署 TLS 终止、反向代理和网络边界保护 |

这些限制必须结合实际部署环境评估，不能把当前测试结果解释为绝对安全或任何合规认证。

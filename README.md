\# FlowPilot



FlowPilot is an enterprise AI workflow agent designed for multi-step task planning, tool calling, knowledge retrieval, human approval and workflow automation.



> 🚧 This project is currently under active development.



\## Project Goal



FlowPilot aims to build a production-oriented AI Agent capable of:



\- Understanding complex user tasks

\- Planning multi-step workflows

\- Calling external tools

\- Retrieving enterprise knowledge

\- Managing workflow state

\- Requesting human approval for risky operations

\- Recovering from tool failures

\- Evaluating Agent execution quality



\## Planned Tech Stack



\- Python

\- FastAPI

\- LangGraph

\- DeepSeek / Qwen

\- MCP

\- PostgreSQL

\- Redis

\- Qdrant

\- Docker

\- GitHub Actions



\## Development Roadmap



\- \[ ] FastAPI backend

\- \[ ] LLM provider

\- \[ ] Tool calling

\- \[ ] LangGraph workflow

\- \[ ] Planner and Router

\- \[ ] Persistence and Memory

\- \[ ] Human-in-the-loop

\- \[ ] Agentic RAG

\- \[ ] MCP integration

\- \[ ] Reliability engineering

\- \[ ] Agent evaluation

\- \[ ] Observability

\- \[ ] Docker deployment

\- \[ ] CI/CD



\## Status



Current version: `v0.0.1`



Project initialization.

## Docker Deployment

FlowPilot supports local single-container deployment with Docker Compose, including a non-root runtime, SQLite named-volume persistence, a Docker healthcheck, and bounded container logs.

See the [Docker deployment guide](docs/docker-deployment.md) for environment setup, startup, maintenance, troubleshooting, and current deployment limitations.

Quick start after securely providing `DEEPSEEK_API_KEY`:

```powershell
docker compose up --build -d
```

Run the isolated Docker integration test with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test_docker.ps1
```

Never commit a real API key or local `.env` file.

## Author

Independent AI Agent engineering project.


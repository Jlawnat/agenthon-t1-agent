FROM finance-bench-sandbox:latest

LABEL qfbench2.interface_version="2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AGENT_WORK_ROOT=/tmp/agenthon-t1

WORKDIR /app

COPY agent /app/agent

ENTRYPOINT ["python", "-m", "agent.main"]

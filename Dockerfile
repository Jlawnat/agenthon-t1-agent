FROM finance-bench-sandbox:latest

RUN pip install --no-cache-dir \
    numba==0.61.0 \
    pandas==2.2.3 \
    scipy==1.15.2 \
    arch==7.2.0

LABEL qfbench2.interface_version="2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    AGENT_WORK_ROOT=/tmp/agenthon-t1

WORKDIR /app

COPY agent /app/agent

ENTRYPOINT ["python", "-m", "agent.main"]

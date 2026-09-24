FROM python:3.11-slim AS cta_legacy

RUN python -m pip install --no-cache-dir \
    "numpy==1.26.4" \
    "scipy==1.12.0" \
    "pandas==2.1.4" \
    "arch==7.2.0"


FROM finance-bench-sandbox:latest

RUN pip install --no-cache-dir \
    numba==0.61.0 \
    pandas==2.2.3 \
    scipy==1.15.2 \
    arch==7.2.0

COPY --from=cta_legacy /usr/local /opt/cta-py311

LABEL qfbench2.interface_version="2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    AGENT_WORK_ROOT=/tmp/agenthon-t1 \
    CTA_LEGACY_HOME=/opt/cta-py311 \
    CTA_LEGACY_PYTHON=/opt/cta-py311/bin/python3.11

WORKDIR /app

COPY agent /app/agent

ENTRYPOINT ["python", "-m", "agent.main"]

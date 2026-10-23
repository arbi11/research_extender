# syntax=docker/dockerfile:1
FROM ubuntu:24.04 AS femm-build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git qt6-base-dev qt6-svg-dev libgc-dev \
    python3 ca-certificates pkg-config && rm -rf /var/lib/apt/lists/*
COPY deploy/fetch_femm.py /tmp/fetch_femm.py
RUN --mount=type=secret,id=proxy_ca python3 /tmp/fetch_femm.py
WORKDIR /opt/femm-source
COPY deploy/patch_femm_mesh.py /tmp/patch_femm_mesh.py
RUN python3 /tmp/patch_femm_mesh.py /opt/femm-source
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export GIT_SSL_CAINFO=/run/secrets/proxy_ca SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    chmod +x linux_build.sh && CMAKE_BUILD_PARALLEL_LEVEL=2 bash ./linux_build.sh && \
    cmake --install build --prefix /opt/femm && \
    install -m 0755 build/bin/femm-lua.sh /opt/femm/bin/femm-lua && \
    install -m 0644 build/bin/femm.lua /opt/femm/bin/femm.lua && \
    test -x /opt/femm/bin/femm_headless

FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:/opt/femm/bin:$PATH FEMM_DIR=/opt/femm/bin \
    QT_QPA_PLATFORM=offscreen DATA_DIR=/data HOME=/data/home
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates libqt6widgets6 libqt6network6 libqt6svg6 libgc1 && \
    rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv
COPY --from=femm-build /opt/femm /opt/femm
# Retain the pinned upstream source archive with the distributed binaries.
COPY --from=femm-build /tmp/femm5-source.zip /opt/femm/upstream-source.zip
# Reproduce the sole upstream change: opt-in export of the actual pre-solve mesh.
COPY --from=femm-build /tmp/patch_femm_mesh.py /opt/femm/patch_femm_mesh.py
WORKDIR /app
COPY requirements.txt ./
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY tests ./tests
RUN useradd --uid 10001 --create-home --home-dir /data/home assistant && \
    mkdir -p /data/home && chown -R assistant:assistant /data && chmod -R a+rX /app /opt/femm
USER assistant
EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "32", "--timeout-keep-alive", "5"]

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    NO_PROXY=localhost,127.0.0.1,0.0.0.0 \
    no_proxy=localhost,127.0.0.1,0.0.0.0 \
    DST_DATA_DIR=/app/data \
    DST_LOG_DIR=/app/logs \
    DST_TEMP_DIR=/app/data/temp \
    DST_DESIGNER_STYLES_DIR=/app/data/designer_styles

WORKDIR /app

RUN groupadd --gid 10001 metadesigner \
    && useradd --uid 10001 --gid metadesigner --create-home metadesigner

COPY requirements-container.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-container.txt

COPY config ./config
COPY core ./core
COPY train ./train
COPY docker/entrypoint.sh /usr/local/bin/metadesigner-entrypoint

RUN chmod +x /usr/local/bin/metadesigner-entrypoint \
    && mkdir -p /app/cache /app/data /app/logs \
    && chown -R metadesigner:metadesigner /app

USER metadesigner

EXPOSE 12002

VOLUME ["/app/cache", "/app/data", "/app/logs"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:12002/', timeout=3)" || exit 1

ENTRYPOINT ["metadesigner-entrypoint"]
CMD ["training"]

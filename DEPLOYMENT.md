# Container deployment

The current image packages the Training Assistant. It does not contain model
weights or the untracked FashionMatrix/IP-Adapter repositories.

## Prerequisites

- Docker Engine 24+ with Docker Compose v2
- A newly generated API key (the previously committed key must be revoked)

## Start

1. Copy `.env.example` to `.env`.
2. Put new credentials in `.env`; never commit this file.
3. Build and run:

   ```bash
   docker compose up --build -d
   ```

4. Open `http://localhost:12002`.
5. Check status with `docker compose ps` and logs with
   `docker compose logs -f training-assistant`.

Runtime data is stored under `runtime/` and survives container replacement.

## Portable image export

```bash
docker build -t metadesigner-training:1.0.0 .
docker save -o metadesigner-training-1.0.0.tar metadesigner-training:1.0.0
```

On the target host:

```bash
docker load -i metadesigner-training-1.0.0.tar
docker compose up -d
```

## Design service boundary

The design service is not included in this CPU image. Its current code requires
external FashionMatrix and IP-Adapter sources, multiple untracked checkpoints,
and an NVIDIA CUDA runtime. Those assets must be pinned, licensed, and mounted
under `/models` before a reproducible GPU image can be built. Copying model
weights into this image would make it extremely large and would not resolve the
missing-source and licensing issues.

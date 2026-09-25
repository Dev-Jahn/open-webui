# Running this fork in Docker

Open WebUI runs as a Docker container built from this fork. mlx-vlm runs natively on the host and
must listen on port **8199** on all interfaces (the container reaches it as
`host.docker.internal:8199`), for example:

```bash
mlx_vlm.server --model ~/models/<model> --host 0.0.0.0 --port 8199
```

All commands below run from the repo root.

## Build

```bash
scripts/fork/docker-build.sh
```

Tags `open-webui-fork:local` and `open-webui-fork:<commit>`. It refuses to build while the tree has
uncommitted changes (other than `run_video.sh`), so the image always matches a commit. The
upstream `Dockerfile` is not edited: the script builds from a temporary copy with a larger Node
heap, as upstream CI does.

## Start, stop, update

```bash
mkdir -p ~/.local/share/open-webui-fork              # first time only
docker compose -f scripts/fork/compose.yaml up -d    # start (restarts by itself unless stopped)
docker compose -f scripts/fork/compose.yaml down     # stop and remove the container (data stays)
docker compose -f scripts/fork/compose.yaml logs -f  # follow the logs

# update: pull / merge, commit, rebuild, re-create the container
scripts/fork/docker-build.sh && docker compose -f scripts/fork/compose.yaml up -d
```

Open <http://localhost:8080>. The port is bound to 127.0.0.1 only.

## Data

Everything lives in `~/.local/share/open-webui-fork` on the host: the database, uploads, video
frame bundles (`uploads/video_frames/`) and the session secret (`.webui_secret_key`; keeps you
logged in across container re-creation). Back up or delete that directory; the container holds no
state.

## First start

The first account to sign up becomes the admin. Sign up yourself before sharing the URL.

## Settings that only seed the database

`OPENAI_API_BASE_URL`, `OPENAI_API_KEY` and `TASK_MODEL_PARAMS` in `compose.yaml` are written to
the database on the first start only; editing them later has no effect. Change them in the UI:
the connection in **Admin Panel > Settings > Connections**, the task parameters in
**Admin Panel > Settings > Interface**.

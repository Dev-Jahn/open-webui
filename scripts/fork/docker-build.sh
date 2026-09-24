#!/usr/bin/env bash
# Build the fork's Docker image from the current commit, without editing upstream files.
#
# Like upstream CI (.github/workflows/docker.yaml, "Prepare CI Dockerfile"), this builds from a
# temporary copy of the Dockerfile that raises the Node heap for the frontend build. Next to that
# copy goes a Dockerfile-specific ignore file (upstream .dockerignore + local-only directories),
# which BuildKit uses instead of the repo's .dockerignore.
#
# Usage: scripts/fork/docker-build.sh   ->  open-webui-fork:local and open-webui-fork:<short hash>
set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# run_video.sh (the owner's local launcher) and scripts/fork (this tooling) are kept out of the
# build context below, so they may differ from HEAD; anything else would make the image differ
# from the commit it is tagged with.
dirty=$(git status --porcelain -- . ':(exclude)run_video.sh' ':(exclude)scripts/fork')
if [[ -n "$dirty" ]]; then
  echo "Refusing to build: uncommitted changes would not match the image tag. Commit or stash them first:" >&2
  echo "$dirty" >&2
  exit 1
fi

hash=$(git rev-parse --short HEAD)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

awk '
  { print }
  /^FROM .* AS build$/ { print "ENV NODE_OPTIONS=\"--max-old-space-size=8192\""; n++ }
  END { if (n != 1) { print "expected one frontend \"FROM ... AS build\" line, found " n > "/dev/stderr"; exit 1 } }
' Dockerfile > "$tmp/Dockerfile"

{
  cat .dockerignore
  printf '%s\n' .venv .hippo .claude .playwright-mcp build '**/__pycache__' run_video.sh scripts/fork
} > "$tmp/Dockerfile.dockerignore"

docker build -f "$tmp/Dockerfile" \
  --build-arg BUILD_HASH="$hash" \
  -t open-webui-fork:local -t "open-webui-fork:$hash" \
  .

echo "Built open-webui-fork:local = open-webui-fork:$hash"

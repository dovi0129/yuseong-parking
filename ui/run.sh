#!/usr/bin/env bash
# Start the tile relay and the screen from ui/ (so ui/.streamlit/config.toml applies); stopping run.sh stops both.
{
cd "$(dirname "$0")" || exit 1
export TILE_PROXY_PORT="${TILE_PORT:-8503}"
../.venv/bin/python tile_proxy.py --port "$TILE_PROXY_PORT" &
PROXY=$!
../.venv/bin/streamlit run main.py --server.port "${PORT:-8502}" "$@" &
APP=$!
trap 'kill "$APP" "$PROXY" 2>/dev/null; wait' EXIT INT TERM
wait "$APP"
exit
}

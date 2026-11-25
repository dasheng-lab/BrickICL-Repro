#!/usr/bin/env bash
# Download an authorized BrickNet release into $BRICKNET_DATA (baseline/bricknet/data).
# Set BRICKNET_SHARE_URL locally after the authors grant access. Never commit it.
#
# Usage: bash data/fetch_bricknet.sh            dataset/ only: paths, captions, graphs (~13 GB, kept as .xz)
#        bash data/fetch_bricknet.sh --meshes   also ldraw/, inset/, watertight/ (cvx/ and glb/ come from
#                                   `python -m bricknet fetch-meshes` / `python -m bricknet_render fetch-glbs`)
#        bash data/fetch_bricknet.sh --ldraw    also ldraw/ only (98 MB; the viewer's packers need it)
# Resumable (re-run after an interruption); each file is checked against the share listing's byte size.
# Full integrity check afterwards (slow for paths_pt): xz -t $BRICKNET_DATA/dataset/*.xz
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export BRICKNET_DATA=${BRICKNET_DATA:-$REPO/baseline/bricknet/data}

: "${BRICKNET_SHARE_URL:?Request BrickNet data access from the authors, then set BRICKNET_SHARE_URL locally}"
SHARE=${BRICKNET_SHARE_URL%/}
case "$SHARE" in
  https://keeper.mpdl.mpg.de/d/*) ;;
  *) echo "BRICKNET_SHARE_URL must be the authorized BrickNet keeper share URL" >&2; exit 2 ;;
esac
SHARE_ID=${SHARE##*/}
LISTING="https://keeper.mpdl.mpg.de/api/v2.1/share-links/${SHARE_ID}/dirents/?path=/"
DATASET="captions_val.jsonl.xz captions_sft.jsonl.xz paths_val.jsonl.xz val.npz.xz sft.npz.xz
         paths_sft.jsonl.xz pt.npz.xz paths_pt.jsonl.xz"
LIST=$(curl -fsS "$LISTING")

size_of() {
  printf '%s' "$LIST" | python3 -I -c \
    'import json, sys; print({e["file_name"]: e["size"] for e in json.load(sys.stdin)["dirent_list"]}[sys.argv[1]])' "$1"
}
bytes() { wc -c < "$1" | tr -d ' '; }

fetch() {  # fetch <file> <dest dir>
  local f=$1 out=$2/$1 want
  want=$(size_of "$f")
  mkdir -p "$2"
  if [ -f "$out" ] && [ "$(bytes "$out")" = "$want" ]; then echo "ok    $f (already complete)"; return; fi
  echo "get   $f ($((want / 1048576)) MiB)"
  for attempt in 1 2 3 4 5; do
    curl -fsSL -C - --retry 3 --retry-delay 5 -o "$out.part" "$SHARE/files/?p=%2F$f&dl=1" || true
    [ -f "$out.part" ] && [ "$(bytes "$out.part")" = "$want" ] && break
    echo "retry $f (attempt $attempt)"; sleep 5
  done
  [ "$(bytes "$out.part")" = "$want" ] || { echo "FAIL  $f: size mismatch" >&2; exit 1; }
  mv "$out.part" "$out"
  echo "ok    $f"
}

for f in $DATASET; do fetch "$f" "$BRICKNET_DATA/dataset"; done

case "${1:-}" in  # archive -> extraction dir; ldraw.tar.xz already has ldraw/ at its root
  --meshes) ARCHIVES="ldraw:. inset:inset watertight:watertight" ;;
  --ldraw) ARCHIVES="ldraw:." ;;
  *) ARCHIVES="" ;;
esac
if [ -n "$ARCHIVES" ]; then
  TMP=$BRICKNET_DATA/.download
  for spec in $ARCHIVES; do
    name=${spec%%:*}; dir=$BRICKNET_DATA/${spec#*:}
    if [ -n "$(ls -A "$BRICKNET_DATA/$name" 2>/dev/null)" ]; then echo "ok    $name/ (already extracted)"; continue; fi
    fetch "$name.tar.xz" "$TMP"
    mkdir -p "$dir" && tar -xJf "$TMP/$name.tar.xz" -C "$dir" && rm "$TMP/$name.tar.xz"
    echo "ok    $name/ extracted"
  done
  rmdir "$TMP" 2>/dev/null || true
fi
echo "done: $BRICKNET_DATA"

#!/bin/sh
set -eu

case "${1:-training}" in
  training)
    exec python -u train/train.py
    ;;
  shell)
    shift
    exec /bin/sh "$@"
    ;;
  *)
    exec "$@"
    ;;
esac

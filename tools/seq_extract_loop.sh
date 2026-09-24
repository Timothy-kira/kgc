#!/bin/bash
# Keep converting new replay-DB shards into decoder-only training sequences.
# usage: tools/seq_extract_loop.sh <db_dir> <out_dir> <first_shard> [procs]
DB=$1; OUT=$2; FIRST=${3:-0}; PROCS=${4:-3}
while true; do
  python -m data.seq_extract "$DB" "$OUT" 1800 "$PROCS" "$FIRST:100000"
  sleep 300
done

#!/bin/bash
# Keep extracting new replay-DB shards as the crawler writes them.
while true; do
  python -m data.extract /tmp/claude-0/-home-user-kgc/98286b48-2e4c-5438-8398-09a385f01653/scratchpad/replay_db /tmp/claude-0/-home-user-kgc/98286b48-2e4c-5438-8398-09a385f01653/scratchpad/ext 1800 1000000 3
  sleep 300
done

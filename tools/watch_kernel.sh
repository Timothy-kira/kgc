#!/bin/bash
# Live-follow a Kaggle training kernel: only step metrics, validation, errors and exit status.
# usage: KAGGLE_API_TOKEN=... tools/watch_kernel.sh <owner/slug>
kaggle kernels logs -f "$1" 2>&1 | grep --line-buffered -E '"step"|GPU_MON|LATENCY|VAL|Error|error|Traceback|exit code|saved|device=|amp dtype|Killed'

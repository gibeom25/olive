#!/usr/bin/env bash
set -u
cd /home/gibeom_pilab/olive
for s in spatial goal object long; do ./models/olive-model.sh fetch $s & done
hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
  --include "libero_spatial/*" "libero_object/*" "libero_goal/*" "libero_10/*" \
  --local-dir ~/LIBERO/LIBERO-plus/libero/datasets &
wait
echo ALL_DOWNLOADS_DONE

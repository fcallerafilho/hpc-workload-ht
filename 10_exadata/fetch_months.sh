#!/usr/bin/env bash
# Fetch the M100 ExaData months that cover PM100's span (2020-05-05 .. 2020-10-13).
#
#   bash 10_exadata/fetch_months.sh            # all remaining months
#   bash 10_exadata/fetch_months.sh 20-09      # just one
#
# Record:  https://zenodo.org/records/7588815   (CC-BY-4.0, no application, no login)
# Files:   20-06.tar 5.22 GB | 20-07.tar 4.58 | 20-08.tar 3.83 | 20-09.tar 4.20 | 20-10.tar 5.39
#          (20-05.tar 6.60 GB is already extracted under dataset/m100/)
#
# Each tar is Parquet partitioned  year_month=<M>/plugin=<P>/metric=<X>/a_0.parquet
# with three columns: timestamp (ms, UTC) | value | node (0..979).
#
# We keep ~1 GB of the ~5 GB per month by extracting only the metrics the pipeline uses.
# tar is sequential, so the whole file is still *streamed* — it is just never stored.
#
# If you download the .tar files by hand instead (faster link, download manager, etc.),
# put them anywhere and run:   bash 10_exadata/fetch_months.sh --local /path/to/tars
# and it will extract from disk instead of the network.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/dataset/m100"
BASE="https://zenodo.org/records/7588815/files"
mkdir -p "$OUT"

# ---- the metrics the pipeline reads -----------------------------------------
# ganglia_pub: 40 s - 3 min per node   |   ipmi_pub: 20 s per node
GANGLIA=(mem_free mem_cached mem_buffers mem_total swap_free proc_run cpu_wio load_one)
IPMI=(gpu0_core_temp gpu1_core_temp gpu3_core_temp gpu4_core_temp
      gpu0_mem_temp gpu4_mem_temp gv100card0 gv100card3
      ambient total_power fan0_0)

patterns_for() {           # $1 = month
  local M="$1"
  printf '%s\n' "year_month=$M/plugin=job_table/*" "year_month=$M/plugin=nagios_pub/*"
  for g in "${GANGLIA[@]}"; do printf '%s\n' "year_month=$M/plugin=ganglia_pub/metric=$g/*"; done
  for i in "${IPMI[@]}";    do printf '%s\n' "year_month=$M/plugin=ipmi_pub/metric=$i/*";    done
}

LOCAL=""
if [ "${1:-}" = "--local" ]; then LOCAL="$2"; shift 2; fi
MONTHS=("$@"); [ ${#MONTHS[@]} -eq 0 ] && MONTHS=(20-06 20-07 20-08 20-09 20-10)

for M in "${MONTHS[@]}"; do
  if [ -d "$OUT/year_month=$M" ]; then echo "== $M already extracted, skipping"; continue; fi
  echo "== $M"
  mapfile -t PAT < <(patterns_for "$M")
  if [ -n "$LOCAL" ]; then
    tar -xf "$LOCAL/$M.tar" -C "$OUT" --wildcards "${PAT[@]}"
  else
    curl -fL --retry 5 --retry-delay 10 -C - "$BASE/$M.tar?download=1" \
      | tar -x -C "$OUT" --wildcards "${PAT[@]}"
  fi
  du -sh "$OUT/year_month=$M"
done

echo
echo "done. extracted months:"
ls -d "$OUT"/year_month=* 2>/dev/null | sed 's|.*/||'

#!/usr/bin/env bash
set -u
PY="/c/Users/lenono/.workbuddy/binaries/python/envs/gcn312/Scripts/python.exe"
ROOT="/f/zhuomian/graph_beginner-main"
R="$ROOT/results"
echo "===== 任务一 stage1(采样重跑) START $(date '+%F %T') ====="
cd "$ROOT/任务一/code" && "$PY" -u run_all.py --stages 1 >> "$R/task1_log.txt" 2>&1
echo "===== rc=$? DONE $(date '+%F %T') ====="
cd "$ROOT" && "$PY" results/make_figures.py >> "$R/figures_log.txt" 2>&1
echo "===== 图表 rc=$? $(date '+%F %T') ====="
echo "===== ALL DONE $(date '+%F %T') ====="

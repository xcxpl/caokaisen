#!/usr/bin/env bash
# 续跑：任务二剩余 / 任务三剩余 / 任务一 stage5 / 生成图表
set -u
PY="/c/Users/lenono/.workbuddy/binaries/python/envs/gcn312/Scripts/python.exe"
ROOT="/f/zhuomian/graph_beginner-main"
R="$ROOT/results"

echo "===== 任务二 续跑 START $(date '+%F %T') ====="
cd "$ROOT/任务二/code" && "$PY" -u run_all.py --stages 1 2 3 4 >> "$R/task2_log.txt" 2>&1
echo "===== 任务二 rc=$? DONE $(date '+%F %T') ====="

echo "===== 任务三 续跑 START $(date '+%F %T') ====="
cd "$ROOT/任务三/code" && "$PY" -u run_all.py --stages 1 2 3 >> "$R/task3_log.txt" 2>&1
echo "===== 任务三 rc=$? DONE $(date '+%F %T') ====="

echo "===== 任务一 stage5 START $(date '+%F %T') ====="
cd "$ROOT/任务一/code" && "$PY" -u run_all.py --stages 5 >> "$R/task1_log.txt" 2>&1
echo "===== 任务一 stage5 rc=$? DONE $(date '+%F %T') ====="

echo "===== 生成图表 START $(date '+%F %T') ====="
cd "$ROOT" && "$PY" results/make_figures.py >> "$R/figures_log.txt" 2>&1
echo "===== 图表 rc=$? DONE $(date '+%F %T') ====="

echo "===== ALL DONE $(date '+%F %T') ====="

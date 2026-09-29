#!/usr/bin/env bash
# 收尾：任务一 stage4 采样参数消融重跑（batch_size 改小）+ 重新生成全部图表
set -u
PY="/c/Users/lenono/.workbuddy/binaries/python/envs/gcn312/Scripts/python.exe"
ROOT="/f/zhuomian/graph_beginner-main"
R="$ROOT/results"

echo "===== 任务一 stage4 START $(date '+%F %T') ====="
cd "$ROOT/任务一/code" && "$PY" -u run_all.py --stages 4 >> "$R/task1_log.txt" 2>&1
echo "===== 任务一 stage4 rc=$? DONE $(date '+%F %T') ====="

echo "===== 生成图表 START $(date '+%F %T') ====="
cd "$ROOT" && "$PY" results/make_figures.py >> "$R/figures_log.txt" 2>&1
echo "===== 图表 rc=$? DONE $(date '+%F %T') ====="

echo "===== ALL DONE $(date '+%F %T') ====="

#!/usr/bin/env bash
# 补齐：任务三 molhiv（12 组合，修复 int64 特征后重跑）/ 任务四 ConvE（修复 1-N 损失后重跑）/ 图表
set -u
PY="/c/Users/lenono/.workbuddy/binaries/python/envs/gcn312/Scripts/python.exe"
ROOT="/f/zhuomian/graph_beginner-main"
R="$ROOT/results"

echo "===== 任务三 molhiv START $(date '+%F %T') ====="
cd "$ROOT/任务三/code" && "$PY" -u run_all.py --stages 1 >> "$R/task3_log.txt" 2>&1
echo "===== 任务三 rc=$? DONE $(date '+%F %T') ====="

echo "===== 任务四 ConvE 重跑 START $(date '+%F %T') ====="
cd "$ROOT/任务四/code" && "$PY" -u run_all.py --stages 1 >> "$R/task4_log.txt" 2>&1
echo "===== 任务四 rc=$? DONE $(date '+%F %T') ====="

echo "===== 生成图表 START $(date '+%F %T') ====="
cd "$ROOT" && "$PY" results/make_figures.py >> "$R/figures_log.txt" 2>&1
echo "===== 图表 rc=$? DONE $(date '+%F %T') ====="

echo "===== ALL DONE $(date '+%F %T') ====="

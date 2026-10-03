#!/usr/bin/env bash
# Запуск бенчмарков одной работы, результат - JSON в каталоге results/
# Пример: ./scripts/run_benchmarks.sh 1
set -euo pipefail

work=${1:?"укажите номер работы: ./scripts/run_benchmarks.sh <1..5>"}
mkdir -p results

"./build/work${work}/benchmarks/work${work}_benchmarks" \
  --benchmark_out="results/work${work}.json" \
  --benchmark_out_format=json

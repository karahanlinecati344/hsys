#!/usr/bin/env bash
# Построение графиков реальной сложности и ускорения по results/workN.json
# Пример: ./scripts/make_charts.sh 1
set -euo pipefail

work=${1:?"укажите номер работы: ./scripts/make_charts.sh <1..5>"}
json="results/work${work}.json"
plot="tools/benchmark-plot"
mkdir -p charts

case "${work}" in
  1)
    ref="Eigen Vector Addition (CPU)"
    target="CUDA Vector Addition (GPU)"
    ;;
  2)
    ref="Eigen Matrix Multiplication (CPU, float)"
    target="CUDA Matrix Multiplication (GPU, Naive, float)"
    ;;
  3)
    ref="CUDA Matrix Multiplication (GPU, Naive, float)"
    target="CUDA Matrix Multiplication (GPU, Shared, float)"
    ;;
  4)
    ref="CUDA Matrix Multiplication (GPU, Shared, float)"
    target="CUDA Matrix Multiplication (GPU, WMMA, half)"
    ;;
  5)
    ref="CUDA Vector Reduction (GPU, nobr)"
    target="CUDA Vector Reduction (GPU, br)"
    ;;
  *)
    echo "нет такой работы: ${work}" >&2
    exit 1
    ;;
esac

python3 "${plot}/complexity_chart.py" -j "${json}" --xlog --ylog \
  -c "charts/work${work}_complexity.html"

python3 "${plot}/speedup_chart.py" -j "${json}" --xlog --ylog --baseline \
  -r "${ref}" -t "${target}" -c "charts/work${work}_speedup.html"

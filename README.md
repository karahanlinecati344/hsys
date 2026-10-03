# hsys — практические работы по гибридным вычислительным системам (CUDA)

| Работа | Содержание |
|--------|------------|
| work1 | `Data`, `VectorView`, `Vector` (паттерн data + view), `kernel_vecadd`, `operator+` |
| work2 | `MatrixView`, `Matrix`, `kernel_matmul_naive`, `operator*` (стратегия `NaiveMatmul`) |
| work3 | `kernel_matmul_shmem` — блочное умножение через разделяемую память (`ShmemMatmul`) |
| work4 | `kernel_matmul_wmma` — умножение на тензорных ядрах, half (`WmmaMatmul`) |
| work5 | `kernel_vecred_nobr`, `kernel_vecred_br` — редукция вектора (`VecredShared`, `VecredShuffle`) |

Каждая работа устроена одинаково: `core` (библиотека), `tests` (Google Test), `benchmarks` (Google Benchmark).
Работы 2–4 используют `Data` из работы 1, работа 3 подключает работу 2 и т. д.

## Окружение

Dev Container курса: <https://github.com/jzcurious/hsys-dev-container> (CUDA 13, CMake, Ninja, clang 18).
Код собирается и на машине без видеокарты NVIDIA (генерируется код для `sm_75`),
но тесты и бенчмарки запускаются только на GPU (для работы 4 нужны тензорные ядра).

## Сборка

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Тесты

```bash
ctest --test-dir build --output-on-failure      # все работы
./build/work1/tests/work1_tests                 # одна работа
```

## Бенчмарки и графики

```bash
pip install -r tools/benchmark-plot/requirements.txt

./scripts/run_benchmarks.sh 1    # -> results/work1.json
./scripts/make_charts.sh 1       # -> charts/work1_complexity.html, charts/work1_speedup.html
```

Время выделения, копирования и освобождения памяти в замер не входит,
время на GPU измеряется через CUDA Events (`hsys::EventTimer`).

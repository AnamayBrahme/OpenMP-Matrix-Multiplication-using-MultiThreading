# OpenMP Matrix Multiplication using MultiThreading

This project demonstrates matrix multiplication in C++ using both single-threaded and OpenMP-based multi-threaded approaches. It compares performance between the two methods using large (1000×1000) and small (3×3) matrices.

## 🚀 Features

-  **Single-threaded matrix multiplication**
-  **OpenMP-parallelized matrix multiplication**
-  Performance benchmarking using `omp_get_wtime()`


## 🧠 Concepts Used

- Dynamic memory allocation with `new` and `delete`
- Parallel loops using `#pragma omp parallel for`
- Measuring execution time with OpenMP’s `omp_get_wtime()

## 🔧 Compilation

```bash
g++ -fopenmp -o matrix_mult main.cpp
./matrix_mult
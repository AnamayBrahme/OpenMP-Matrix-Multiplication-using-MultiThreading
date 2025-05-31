#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <iomanip>

#define N 1000 // Used only for large matrix test

// Single-threaded matrix multiplication
void singleThread_multiply(int** A, int** B, int** C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
        }
}

// OpenMP-parallelized matrix multiplication
void OMP_multiply(int** A, int** B, int** C, int n) {
    // Parallelize the outer loop
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
        }
}

// Print matrix
// Note: This function is not used in the code, but it's here for my understanding
void print_matrix(const char* name, int** M, int n) {
    std::cout << "\nMatrix " << name << ":\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << M[i][j] << " ";
        std::cout << "\n";
    }
}
// Using a 2D array
// Using Pointers for matrix allocation
// Allocate a 2D matrix dynamically
int** allocate_matrix(int n) {
    int** mat = new int*[n];
    for (int i = 0; i < n; ++i)
        mat[i] = new int[n];
    return mat;
}

// Free the dynamically allocated matrix
void free_matrix(int** mat, int n) {
    for (int i = 0; i < n; ++i)
        delete[] mat[i];
    delete[] mat;
}

int main() {
    // --- Part 1: Large Matrix Performance Test (Random) ---
    std::cout << "Time for " << N << "x" << N << " matrices\n";
    std::cout << "Using " << omp_get_max_threads() << " threads\n";

    int** A = allocate_matrix(N);
    int** B = allocate_matrix(N);
    int** single_C = allocate_matrix(N);
    int** OMP_C = allocate_matrix(N);

    srand(0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }

    double start = omp_get_wtime();
    singleThread_multiply(A, B, single_C, N);
    double end = omp_get_wtime();
    std::cout << "Time (Single-threaded): " << (end - start) << " seconds\n";

    start = omp_get_wtime();
    OMP_multiply(A, B, OMP_C, N);
    end = omp_get_wtime();
    std::cout << "Time (OpenMP): " << (end - start) << " seconds\n";
    // --- End of Part 1 ---
    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(single_C, N);
    free_matrix(OMP_C, N);

    // --- Part 2: Small Matrix (3x3) Demonstration ---
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nTime for 3x3 Predefined Matrices\n";
    const int small_N = 3;
    // Making the matrices small to avoid memory issues
    int** A3 = allocate_matrix(small_N);
    int** B3 = allocate_matrix(small_N);
    int** C_single3 = allocate_matrix(small_N);
    int** C_omp3 = allocate_matrix(small_N);

    int tempA[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    int tempB[3][3] = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };

    for (int i = 0; i < small_N; ++i)
        for (int j = 0; j < small_N; ++j) {
            A3[i][j] = tempA[i][j];
            B3[i][j] = tempB[i][j];
        }

    start = omp_get_wtime();
    singleThread_multiply(A3, B3, C_single3, small_N);
    end = omp_get_wtime();
    std::cout << "Time (Single-threaded): " << (end - start) << " seconds\n";
    start = omp_get_wtime();
    OMP_multiply(A3, B3, C_omp3, small_N);
    end = omp_get_wtime();
    std::cout << "Time (OpenMP): " << (end - start) << " seconds\n";
    // Avoiding Printing matrices that plays a role for slowing down processes.
    //print_matrix("A3", A3, small_N);
    //print_matrix("B3", B3, small_N);
    //print_matrix("C_single (3x3)", C_single3, small_N);
    //print_matrix("C_omp (3x3)", C_omp3, small_N);

    free_matrix(A3, small_N);
    free_matrix(B3, small_N);
    free_matrix(C_single3, small_N);
    free_matrix(C_omp3, small_N);

    return 0;
}

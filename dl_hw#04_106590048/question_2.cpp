#include <omp.h> // linux add compiler flag -fopenmp
#include <stdio.h>
#include <chrono>

#define PARALLEL // flag for executing parallel program
#define N 3

int main()
{
    int a[N][N];
    int b[N][N];
    int value = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            a[i][j] = ++value;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#ifdef PARALLEL
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i)
#ifdef PARALLEL
        #pragma omp parallel for
#endif
        for (int j = 0; j < N; ++j)
        {
            b[i][j] = a[i][j];
            if (i > 0)   b[i][j] += a[i-1][j]; // up
            if (i < N-1) b[i][j] += a[i+1][j]; // down
            if (j > 0)   b[i][j] += a[i][j-1]; // left
            if (j < N-1) b[i][j] += a[i][j+1]; // right
        }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    printf("Main for loop took me %ld nanoseconds.\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j) printf("%d ", b[i][j]);
        printf("\n");
    }
    return 0;
}

#include <omp.h> // linux add compiler flag -fopenmp
#include <stdio.h>
#include <chrono>

#define PARALLEL // flag for executing parallel program
#define N 9

int main()
{
    int a[N];
    int b[N];
    for (int i = 0; i < N; ++i) a[i] = i+1;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#ifdef PARALLEL
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i)
    {
        b[i] = a[i];
        if (i > 0)   b[i] += a[i-1]; // left
        if (i < N-1) b[i] += a[i+1]; // right
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    printf("Main for loop took me %ld nanoseconds.\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    for (int i = 0; i < N; ++i) printf("%d ", b[i]);
    printf("\n");
    return 0;
}

#include <omp.h> // linux add compiler flag -fopenmp
#include <stdio.h>

#define MAX_LOOP (int)1E8

void test(int n)
{
    for (int i = 0; i < MAX_LOOP; ++i)
    {
        // do nothing
    }
    printf("%d, ", n);
}

int main()
{
    #pragma omp parallel for
    for (int i = 0; i < 10; ++i) test(i);
    return 0;
}

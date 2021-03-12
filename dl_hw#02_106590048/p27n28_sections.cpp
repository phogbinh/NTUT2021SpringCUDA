#include <omp.h> // linux add compiler flag -fopenmp
#include <stdio.h>

#define MAX_LOOP (int)1E8

void test(int n)
{
    printf("<T:%d> - %d\n", omp_get_thread_num(), n);
}

int main()
{
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int i = 0; i < MAX_LOOP; ++i)
            {
                // do nothing
            }
            test(0);
        }

        #pragma omp section
        {
            test(1);
        }

        #pragma omp section
        {
            test(2);
        }

        #pragma omp section
        {
            test(3);
        }

    }
    return 0;
}

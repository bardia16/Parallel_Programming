
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <cmath>
#include "omp.h"

#define VECTOR_SIZE 1048576

double serial_avg_std(float data [VECTOR_SIZE])
{
    clock_t start_time = clock();

    double serial_execution_time;
    float sum_avg = 0, sum_sqr = 0;
    float average, standard_deviation;

    for (int i = 0; i < VECTOR_SIZE; i+=4)
    {
        sum_avg += data[i];
        sum_sqr += data[i] * data[i];
    }

    average = sum_avg / VECTOR_SIZE;
    standard_deviation = sqrtf((sum_sqr/VECTOR_SIZE) - (average * average));

    clock_t end_time = clock();
    serial_execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("Serial Mean: %f, Standard Deviation: %f\n", average, standard_deviation);

    return serial_execution_time;
}

double parallel_avg_std(float data [VECTOR_SIZE])
{
    clock_t start_time = clock();

    double parallel_execution_time;
    float sum_avg = 0, sum_sqr = 0, local_sum_sqr, local_sum_avg;
    float average, standard_deviation;
    #pragma omp parallel shared(data, sum_avg, sum_sqr) private(local_sum_avg, local_sum_sqr)
    {
        local_sum_avg = 0;
        local_sum_sqr = 0;
        #pragma omp for nowait
        for (int i = 0; i < VECTOR_SIZE; i+=4)
        {
            local_sum_avg += data[i];
            local_sum_sqr += data[i] * data[i];
        }
        #pragma omp critical 
        {
            sum_avg += local_sum_avg;
            sum_sqr += local_sum_sqr;
        }
    }

    average = sum_avg / VECTOR_SIZE;
    standard_deviation = sqrtf((sum_sqr/VECTOR_SIZE) - (average * average));

    clock_t end_time = clock();
    parallel_execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Open MP Mean: %f, Standard Deviation: %f\n", average, standard_deviation);

    return parallel_execution_time;
}

int main() {
    float data[VECTOR_SIZE];
    double serial_execution_time, simd_execution_time, speedup;

    srand(time(NULL));
    for (int i = 0; i < VECTOR_SIZE; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
    
    serial_execution_time = serial_avg_std(data);
    simd_execution_time = parallel_avg_std(data);
    speedup = serial_execution_time / simd_execution_time;

    printf("Speedup: %.2f\n", speedup);

    return 0;
}

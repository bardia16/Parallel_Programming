#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h> 
#include <cfloat>
#include <time.h>
#include "omp.h"

#define VECTOR_SIZE 1048576

double serial_min(float data [VECTOR_SIZE])
{
    clock_t start_time = clock();

    double serial_execution_time;
    float min = data[0];
    int index = 0;

    for(int i = 1; i < VECTOR_SIZE; i++){
        if(data[i] < min){
            min = data[i];
            index = i;
        }
    }

    clock_t end_time = clock();
    serial_execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Serial Min: %f, Index: %d\n", min, index);

    return serial_execution_time;
}

double parallel_min(float data [VECTOR_SIZE])
{
    clock_t start_time = clock();

    float min = data[0], local_min;
    int index = 0, local_index;

    #pragma omp parallel shared(data, min, index) private(local_min, local_index)
    {
        local_min = data[0];
        local_index = 0;
        #pragma omp for nowait
            for(int i = 1; i < VECTOR_SIZE; i++){
                if(data[i] < local_min){
                    local_min = data[i];
                    local_index = i;
                }
            }
        #pragma omp critical 
        {
            if(local_min < min){
                min = local_min;
                index = local_index;
            }
        }
        
    }

    clock_t end_time = clock();
    double parallel_execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Open MP Min: %f, Index: %d\n", min, index);

    return parallel_execution_time;
}

int main()
{
    double serial_execution_time, simd_execution_time, speedup;
    float data [VECTOR_SIZE];     

    srand(time(NULL));
    for (int i = 0; i < VECTOR_SIZE; i++) {
        data[i] = (float)rand()*1000000000 / RAND_MAX;  //to see the min better since the random numbers are too small
    }

    serial_execution_time = serial_min(data);
    simd_execution_time = parallel_min(data);
    speedup = serial_execution_time / simd_execution_time;

    printf("Speedup: %.2f\n", speedup);

	return 0;
}

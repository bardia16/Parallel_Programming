#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include "omp.h"

// Function to calculate absolute difference using SSE3

int sum_of_frame(uint8_t ** frame, int width, int height){
    int sum = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            sum += frame[i][j];
        }
    }
    return sum;
}

// Function to calculate absolute difference using SSE3
void abs_diff_sse3(uint8_t **frame1, uint8_t **frame2, uint8_t **result, int width, int height) {
    #pragma omp parallel for num_threads(4)
    {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j += 16) {
                __m128i xmm1 = _mm_load_si128((__m128i*)&frame1[i][j]);
                __m128i xmm2 = _mm_load_si128((__m128i*)&frame2[i][j]);
                __m128i xmm_result = _mm_max_epu8(_mm_subs_epu8(xmm1, xmm2), _mm_subs_epu8(xmm2, xmm1)); // no abs in SSE3
                _mm_store_si128((__m128i*)&result[i][j], xmm_result);
            }
        }
    }
}

// Function to calculate absolute difference serially
void abs_diff_serial(uint8_t **frame1, uint8_t **frame2, uint8_t **result, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            result[i][j] = abs(frame1[i][j] - frame2[i][j]);
        }
    }
}

int main() {
    int width = 640; // Width of frames
    int height = 480; // Height of frames

    // Allocate memory for the frames and result matrices with alignment
    uint8_t **frame1 = (uint8_t**)aligned_alloc(16, height * sizeof(uint8_t*));
    uint8_t **frame2 = (uint8_t**)aligned_alloc(16, height * sizeof(uint8_t*));
    uint8_t **result_sse3 = (uint8_t**)aligned_alloc(16, height * sizeof(uint8_t*));
    uint8_t **result_serial = (uint8_t**)aligned_alloc(16, height * sizeof(uint8_t*));

    for (int i = 0; i < height; i++) {
        frame1[i] = (uint8_t*)aligned_alloc(16, width);
        frame2[i] = (uint8_t*)aligned_alloc(16, width);
        result_sse3[i] = (uint8_t*)aligned_alloc(16, width);
        result_serial[i] = (uint8_t*)aligned_alloc(16, width);
    }

    // Initialize frame1 and frame2 with random pixel values
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            frame1[i][j] = rand() % 256; // Random value between 0 and 255
            frame2[i][j] = rand() % 256; // Random value between 0 and 255
        }
    }

    // Measure execution time for SSE3
    timeval t1, t2;
    gettimeofday(&t1, NULL);
    abs_diff_sse3(frame1, frame2, result_sse3, width, height);
    gettimeofday(&t2, NULL);
    
    double parallelTime = (t2.tv_sec - t1.tv_sec) * 1000.0; 
    parallelTime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    // Measure execution time for serial
    gettimeofday(&t1, NULL);
    abs_diff_serial(frame1, frame2, result_serial, width, height);
    gettimeofday(&t2, NULL);
     double serialTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
    serialTime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    // Calculate speedup
    double speedUp = serialTime / parallelTime;
    
    
    int checksum_parallel = sum_of_frame(result_sse3, width, height);
    int checksum_serial = sum_of_frame(result_serial, width, height);
    
    printf("Parallel checksum: %d\n", checksum_parallel);
    printf("Serial checksum: %d\n", checksum_serial);
    printf("SSE3 execution time: %f seconds\n", parallelTime);
    printf("Serial execution time: %f seconds\n", serialTime);
    printf("Speedup: %f\n", speedUp);

    // Free allocated memory
    for (int i = 0; i < height; i++) {
        free(frame1[i]);
        free(frame2[i]);
        free(result_sse3[i]);
        free(result_serial[i]);
    }
    free(frame1);
    free(frame2);
    free(result_sse3);
    free(result_serial);

    return 0;
}

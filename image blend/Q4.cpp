#include <immintrin.h>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "omp.h"

void fillImage(cv::Mat& image, float** imageData) {
    int imageWidth = image.cols;
    int imageHeight = image.rows;

    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            imageData[i][j] = image.at<unsigned char>(i, j);
        }
    }
}


void saveGrayscaleImage(float** imageData, int imageWidth, int imageHeight, const std::string& filename) {
    // Create an OpenCV Mat object from the 2D array
    cv::Mat imageMat(imageHeight, imageWidth, CV_8U);

    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            imageMat.at<unsigned char>(i, j) = imageData[i][j];
        }
    }

    // Save the image as a grayscale image
    cv::imwrite(filename, imageMat);
}


// Function to perform pixel-wise addition with alpha blending using SIMD (SSE3)
void alphaBlendSIMD(float** image1, float** image2, float** result, float alpha, int width, int height) {
    __m128 alphaVec = _mm_set1_ps(alpha);
    __m128 oneMinusAlphaVec = _mm_set1_ps(1.0f - alpha);

	#pragma omp parallel for num_threads(8)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j += 4) {
            // Load 4 pixels from image1 and image2
            __m128 pixel1 = _mm_load_ps(&image1[i][j]);
            __m128 pixel2 = _mm_load_ps(&image2[i][j]);

            // Blend using alpha
            __m128 blended = _mm_add_ps(_mm_mul_ps(pixel2, alphaVec), _mm_mul_ps(pixel1, oneMinusAlphaVec));

            // Store the result
            _mm_store_ps(&result[i][j], blended);
        }
    }
}

void alphaBlendSerial(float** image1, float** image2, float** result, float alpha, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float pixel1 = image1[i][j];
            float pixel2 = image2[i][j];

            // Blend using alpha
            result[i][j] = alpha * pixel2 + (1.0f - alpha) * pixel1;
        }
    }
}

int main() {
    float alpha = 0.4;
    
            // Load image1 and image2 from input files using OpenCV
    cv::Mat img1 = cv::imread("image1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("image2.png", cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Failed to load images." << std::endl;
        return 1;
    }


    
    int width = std::max(img1.cols, img2.cols);
    int height = std::max(img1.rows, img2.rows);
    
    

    // Allocate memory for image1, image2, and result as 2D arrays
    float** image1 = (float**)aligned_alloc(16, height * sizeof(float*));
    float** image2 = (float**)aligned_alloc(16, height * sizeof(float*));
    float** result_sse3 = (float**)aligned_alloc(16, height * sizeof(float*));
    float** result_serial = (float**)aligned_alloc(16, height * sizeof(float*));

    for (int i = 0; i < height; i++) {
        image1[i] = (float*)aligned_alloc(16, width * sizeof(float));
        image2[i] = (float*)aligned_alloc(16, width * sizeof(float));
        result_sse3[i] = (float*)aligned_alloc(16, width * sizeof(float));
        result_serial[i] = (float*)aligned_alloc(16, width * sizeof(float));
    }
    


    fillImage(img1, image1);
    fillImage(img2, image2);


    // Calculate execution time for SIMD alpha blending
    clock_t startSIMD = clock();
    alphaBlendSIMD(image1, image2, result_sse3, alpha, width, height);
    clock_t endSIMD = clock();
    double timeSIMD = double(endSIMD - startSIMD) / CLOCKS_PER_SEC;
    
    clock_t startSerial = clock();
    alphaBlendSerial(image1, image2, result_serial, alpha, width, height);
    clock_t endSerial = clock();
    double timeSerial = double(endSerial - startSerial) / CLOCKS_PER_SEC;
    
    double speedup = timeSerial / timeSIMD;
    
    printf("SSE3 execution time: %f seconds\n", timeSIMD);
    printf("Serial execution time: %f seconds\n", timeSerial);
    printf("Speedup: %f\n", speedup);


    //save results
    saveGrayscaleImage(result_serial, width, height, "serial_result.png");
    saveGrayscaleImage(result_sse3, width, height, "parallel_result.png");

    // Clean up memory
    for (int i = 0; i < height; i++) {
        free(image1[i]);
        free(image2[i]);
        free(result_sse3[i]);
        free(result_serial[i]);
    }
    free(image1);
    free(image2);
    free(result_sse3);
    free(result_serial);

    return 0;
}

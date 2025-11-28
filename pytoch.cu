#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

// Simple GFLOPS calculation functions
double calculate_gemm_gflops(double milliseconds, int M, int N, int K) {
    double total_flops = 2.0 * M * N * K;
    double seconds = milliseconds / 1000.0;
    double gflops = total_flops / (seconds * 1e9);
    return gflops;
}

double calculate_conv_gflops(double milliseconds, int batch_size, int in_channels, 
                            int out_channels, int height, int width, int kernel_size) {
    // Simplified convolution FLOPs calculation
    double total_flops = 2.0 * batch_size * out_channels * height * width * 
                        in_channels * kernel_size * kernel_size;
    double seconds = milliseconds / 1000.0;
    double gflops = total_flops / (seconds * 1e9);
    return gflops;
}

// Function to measure performance of any operation
template<typename Func>
std::pair<double, double> measure_performance(Func&& operation, int M, int N, int K, int warmup = 10, int iterations = 100) {
    std::vector<double> times;
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        operation();
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    for (int i = 0; i < iterations; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        operation();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        times.push_back(milliseconds);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double avg_gflops = calculate_gemm_gflops(avg_time, M, N, K);
    double best_time = *std::min_element(times.begin(), times.end());
    double peak_gflops = calculate_gemm_gflops(best_time, M, N, K);
    
    return {avg_gflops, peak_gflops};
}

// Example usage with your GEMM kernel
extern "C" void benchmark_custom_gemm(int M, int N, int K, float* d_A, float* d_B, float* d_C) {
    // This would call your custom GEMM kernel
    // For now, just demonstrate the timing framework
    
    auto operation = [&]() {
        // Call your GEMM kernel here
        // your_gemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    };
    
    auto [avg_gflops, peak_gflops] = measure_performance(operation, M, N, K);
    
    std::cout << "Custom GEMM Performance:" << std::endl;
    std::cout << "Matrix: " << M << "x" << K << " * " << K << "x" << N << std::endl;
    std::cout << "Average GFLOPS: " << avg_gflops << std::endl;
    std::cout << "Peak GFLOPS: " << peak_gflops << std::endl;
}

// Main function for testing
int main() {
    int M = 4096, N = 4096, K = 4096;
    
    // Example: Measure performance of some operation
    auto dummy_operation = []() {
        // Simulate some work
        cudaDeviceSynchronize();
    };
    
    auto [avg_gflops, peak_gflops] = measure_performance(dummy_operation, M, N, K, 5, 10);
    
    std::cout << "Performance Results:" << std::endl;
    std::cout << "Average GFLOPS: " << avg_gflops << std::endl;
    std::cout << "Peak GFLOPS: " << peak_gflops << std::endl;
    
    return 0;
}
// #include <algorithm>
// #include <cstdio>
// #include <cstdlib>
// #include <cublas_v2.h>
// #include <cuda_runtime.h>

// template<const int BM, const int BN, const int BK, const int TM, const int TN>
// __global__ void sgemm_6(int M, int N, int K, float *A, float *B, float *C) {
//     __shared__ float As[BM * BK];
//     __shared__ float Bs[BK * BN];
//     const int crow = blockIdx.x;
//     const int ccol = blockIdx.y;
    
//     const int threadcol=threadIdx.x%(BN/TN);
//     const int threadrow=threadIdx.x/(BN/TN);

//     A+=crow*BM*K;
//     B+=ccol*BN;
//     C+=crow*BM*N+ccol*BN;

//     const int innerrowA=threadIdx.x/BK;
//     const int innercolA=threadIdx.x%BK;
//     const int innerrowB=threadIdx.x/BN;
//     const int innercolB=threadIdx.x%BN;
//     float threadResults[TM * TN] = {0.0f};
//     float regM[TM] = {0.0f};
//     float regN[TN] = {0.0f};
//     for(int bkIdx=0;bkIdx<K;bkIdx+=BK){
//         float tmp=reinterpret_cast<float&>(0);
//         As[(innerColA*4+0)*BM+innerrowA]=tmp.x;
//         As[(innerColA*4+1)*BM+innerrowA]=tmp.y;
//         As[(innerColA*4+2)*BM+innerrowA]=tmp.z;
//         As[(innerColA*4+3)*BM+innerrowA]=tmp.w
//         reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
//         reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
//         __syncthreads();
//         A += BK;
//         B+=BK*N;
//         for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
//             for (int i = 0; i < TM; ++i) {
//                 regM[i] = As[(threadrow * TM + i) * BK + dotIdx];
//             }
//             for (int i = 0; i < TN; ++i) {
//                 regN[i] = Bs[dotIdx * BN + threadcol * TN + i];
//             }
//             for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
//                 for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
//                     threadResults[resIdxM * TN + resIdxN] +=
//                     regM[resIdxM] * regN[resIdxN];
//                 }
//             }
//         }
//         __syncthreads();

//     }
//     for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
//         for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
//       // load C vector into registers
//             float4 tmp = reinterpret_cast<float4 *>(
//             &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
//       // perform GEMM update in reg
//             tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
//             tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
//             tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
//             tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
//       // write back
//             reinterpret_cast<float4 *>(
//                 &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
//                     tmp;
//         }
//     }
// }


// int main(){
//     int M=128,N=128,K=128;
//     size_t bytesA=M*K*sizeof(float);
//     size_t bytesB=K*N*sizeof(float);
//     size_t bytesC=M*N*sizeof(float);

//     float *h_A=(float*)malloc(bytesA);
//     float *h_B=(float*)malloc(bytesB);
//     float *h_C=(float*)malloc(bytesC);

//     for (int i = 0; i < M*K; ++i) h_A[i] = 1.0f; 
//     for (int i = 0; i < K*N; ++i) h_B[i] = 1.0f;
//     for (int i = 0; i < M*N; ++i) h_C[i] = 0.0f;  // Initialize to 0

//     float *d_A,*d_B,*d_C;
//     cudaMalloc(&d_A, bytesA);
//     cudaMalloc(&d_B, bytesB); 
//     cudaMalloc(&d_C, bytesC);

//     cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice);  // Initialize device C
    
//     constexpr int BLOCK = 32;
    
//     // CORRECT for 1D blocks: grid uses 2D, block uses 1D with BLOCK*BLOCK threads
//     dim3 gridDim(CEIL_DIV(M, BLOCK), CEIL_DIV(N, BLOCK));
//     int blockSize = BLOCK * BLOCK;  // 1D block with 1024 threads

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start);
//     sgemm_3<BLOCK><<<gridDim, blockSize>>>(M, N, K, d_A, d_B, d_C);
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
    
//     float ms = 0;
//     cudaEventElapsedTime(&ms, start, stop);

//     double gflops =(2.0 * M * N * K) / (ms / 1e3) / 1e6;
//     printf("Time = %.3f ms\n", ms);
//     printf("GFLOPS = %.3f\n", gflops);

//     cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
    
//     printf("\n--- Checking first 2 rows (first 10 columns) ---\n");
//     for (int r = 0; r < 2; r++) {
//         printf("Row %d: ", r);
//         for (int c = 0; c < 10; c++) {
//             printf("%.1f ", h_C[r * N + c]);
//         }
//         printf("\n");
//     }

//     printf("\nExpected value everywhere = %d\n", K);
    
//     // Verify results
//     bool correct = true;
//     for (int i = 0; i < M * N; i++) {
//         if (fabs(h_C[i] - K) > 1e-5) {
//             printf("Error at index %d: got %.1f, expected %d\n", i, h_C[i], K);
//             correct = false;
//             break;
//         }
//     }
//     printf("Results: %s\n", correct ? "CORRECT" : "INCORRECT");
    
//     cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
//     free(h_A); free(h_B); free(h_C);
//     return 0;
// }






#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_6(int M, int N, int K, float *A, float *B, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    const int cRow = blockIdx.y;  // Fixed: x for rows (conventional)
    const int cCol = blockIdx.x;  // Fixed: y for columns (conventional)
    
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // Move pointers to current block position
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Vectorized loading indices - load 4 elements at once
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    
    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};
    
    for(int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load and transpose A using float4
        float4 tmpA = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmpA.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmpA.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmpA.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmpA.w;

        // Load B without transposition using float4
        float4 tmpB = reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] = tmpB;
        
        __syncthreads();
        
        // Advance to next block in K dimension
        A += BK;
        B += BK * N;

        // Compute partial results
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load TM elements from A into registers
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            // Load TN elements from B into registers  
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            // Compute TM x TN results
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // Write results back to global memory (C = A * B, no alpha/beta)
    for (int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            // Write 4 elements at once
            float4 tmpC;
            tmpC.x = threadResults[resIdxM * TN + resIdxN + 0];
            tmpC.y = threadResults[resIdxM * TN + resIdxN + 1];
            tmpC.z = threadResults[resIdxM * TN + resIdxN + 2];
            tmpC.w = threadResults[resIdxM * TN + resIdxN + 3];
            
            reinterpret_cast<float4*>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmpC;
        }
    }
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f; 
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB); 
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice);
    
    // Kernel parameters - adjust based on your BM, BN, BK, TM, TN choices
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8, TN = 8;
    
    // Grid dimensions: M/BM blocks in x, N/BN blocks in y
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    
    // Block threads: (BN/TN) * (BM/TM) threads
    int blockSize = (BN / TN) * (BM / TM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_6<BM, BN, BK, TM, TN><<<gridDim, blockSize>>>(M, N, K, d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double gflops = (2.0 * M * N * K) / (ms / 1e3) / 1e9;  // Fixed to Giga FLOPS
    printf("Time = %.3f ms\n", ms);
    printf("GFLOPS = %.3f\n", gflops);

    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
    
    printf("\n--- Checking first 2 rows (first 10 columns) ---\n");
    for (int r = 0; r < 2; r++) {
        printf("Row %d: ", r);
        for (int c = 0; c < 10; c++) {
            printf("%.1f ", h_C[r * N + c]);
        }
        printf("\n");
    }

    printf("\nExpected value everywhere = %d\n", K);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - K) > 1e-5) {
            printf("Error at index %d: got %.1f, expected %d\n", i, h_C[i], K);
            correct = false;
            break;
        }
    }
    printf("Results: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    std::cout << "Number of CUDA devices: " << deviceCount << "\n\n";

for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    std::cout << "Device " << i << ": " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total global memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
    std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
    std::cout << "  Shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes\n";
    std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
    std::cout << "  Warp size: " << prop.warpSize << "\n";
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "  Number of SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "  Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << "\n";
    std::cout << "  Max grid dimensions: ["
                << prop.maxGridSize[0] << ", "
                << prop.maxGridSize[1] << ", "
                << prop.maxGridSize[2] << "]\n";
    std::cout << "  Max threads dim (block): ["
                << prop.maxThreadsDim[0] << ", "
                << prop.maxThreadsDim[1] << ", "
                << prop.maxThreadsDim[2] << "]\n";
    std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz\n";
    std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz\n";
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "  L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
    std::cout << "  Registers per SM: " << prop.regsPerMultiprocessor << " registers\n";
    std::cout << "  Registers per Block: " << prop.regsPerBlock;
    std::cout << "  Registers per Block: " << prop.warpSize;

    std::cout << std::endl;
}

    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    free(h_A); 
    free(h_B); 
    free(h_C);
    
    return 0;
}
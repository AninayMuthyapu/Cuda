
// #include <algorithm>
// #include <cstdio>
// #include <cstdlib>
// #include <cublas_v2.h>
// #include <cuda_runtime.h>
// #define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

// template<const int blocksize>
// __global__ void sgemm_3(int M,int N,int K,float *A,float *B,float *C){
//     __shared__ float A_s[blocksize*blocksize];
//     __shared__ float B_s[blocksize*blocksize];
//     const int threadrow=threadIdx.x/blocksize;
//     const int threadcol=threadIdx.x%blocksize;
//     const  int crow=blockIdx.x;
//     const int ccol=blockIdx.y; //(0,0)---based on blockindex
//     //pointers to the starting position.
//     A += crow * blocksize * K;                    // row=cRow, col=0
//     B += ccol * blocksize;                        // row=0, col=cCol
//     C += crow * blocksize * N + ccol * blocksize; // row=cRow, col=cCol

    
    
//     float temp=0;
//     for(int bkIdx=0;bkIdx<K;bkIdx+=blocksize){
//         A_s[threadrow*blocksize+threadcol]=A[threadrow*K+threadcol];
//         B_s[threadrow*blocksize+threadcol]=B[threadrow*N+threadcol];
//         __syncthreads();
//         A+=blocksize;
//         B+=blocksize*N;
//         //execute the dotproduct on the currently cached block;
        
//         for(int dotIdx=0;dotIdx<blocksize;++dotIdx){
//             temp+=A_s[threadrow*blocksize+dotIdx]*B_s[dotIdx*blocksize+threadcol];
//         }
//         __syncthreads();
        
//     }
//     C[threadrow*N+threadcol]+=temp;
// }





// int main(){
// int M=128,N=128,K=128;
// size_t bytesA=M*K*sizeof(float);
// size_t bytesB=K*N*sizeof(float);
// size_t bytesC=M*N*sizeof(float);

// float *h_A=(float*)malloc(bytesA);
// float *h_B=(float*)malloc(bytesB);
// float *h_C=(float*)malloc(bytesC);

// for (int i = 0; i < M*K; ++i) h_A[i] = 1.0f; 
// for (int i = 0; i < K*N; ++i) h_B[i] = 1.0f;
// float *d_A,*d_B,*d_C;
// cudaMalloc(&d_A, bytesA);
// cudaMalloc(&d_B, bytesB); 
// cudaMalloc(&d_C, bytesC);

// cudaMemcpy(d_A,h_A,bytesA,cudaMemcpyHostToDevice);
// cudaMemcpy(d_B,h_B,bytesB,cudaMemcpyHostToDevice);
// constexpr int BLOCK = 32;
// dim3 blockDim(BLOCK, BLOCK);
// dim3 gridDim(CEIL_DIV(N, BLOCK), CEIL_DIV(M, BLOCK));

// cudaEvent_t start, stop;
// cudaEventCreate(&start);
// cudaEventCreate(&stop);

// cudaEventRecord(start);
// sgemm_3<BLOCK><<<gridDim,blockDim>>>(M,N,K,d_A,d_B,d_C);
// cudaEventRecord(stop);
// cudaDeviceSynchronize();
// float ms = 0;
// cudaEventElapsedTime(&ms, start, stop);

// double gflops =(2.0 * M * N * K) / (ms / 1e3) / 1e9;
// printf("Time = %.3f ms\n", ms);
// printf("GFLOPS = %.3f\n", gflops);

// cudaMemcpy(h_C,d_C,bytesC,cudaMemcpyDeviceToHost);
// printf("\n--- Checking first 2 rows (first 10 columns) ---\n");

// for (int r = 0; r < 2; r++) {
//     printf("Row %d: ", r);
//     for (int c = 0; c < 10; c++) {
//         printf("%.1f ", h_C[r * N + c]);
//     }
//     printf("\n");
// }

// printf("\nExpected value everywhere = %d\n", K);
// cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
// free(h_A); free(h_B); free(h_C);
// return 0;



// }




#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

template<const int blocksize>
__global__ void sgemm_3(int M,int N,int K,float *A,float *B,float *C){
    __shared__ float A_s[blocksize*blocksize];
    __shared__ float B_s[blocksize*blocksize];
    
    // CORRECT for 1D blocks: compute threadrow and threadcol from threadIdx.x
    const int threadrow = threadIdx.x / blocksize;
    const int threadcol = threadIdx.x % blocksize;
    
    const int crow = blockIdx.x;
    const int ccol = blockIdx.y;
    
    A += crow * blocksize * K;
    B += ccol * blocksize;
    C += crow * blocksize * N + ccol * blocksize;

    float temp=0;
    for(int bkIdx=0;bkIdx<K;bkIdx+=blocksize){
        // Load A tile - FIXED: use K for A indexing
        A_s[threadrow*blocksize+threadcol] = A[threadrow*K+threadcol];
        
        // Load B tile - FIXED: use N for B indexing  
        B_s[threadrow*blocksize+threadcol] = B[threadrow*N+threadcol];
        
        __syncthreads();
        
        // CORRECT pointer advancement
        A += blocksize;
        B += blocksize * N;
        
        // Execute dot product on cached blocks
        for(int dotIdx=0;dotIdx<blocksize;++dotIdx){
            temp += A_s[threadrow*blocksize+dotIdx] * B_s[dotIdx*blocksize+threadcol];
        }
        __syncthreads();
    }
    
    // Write result - FIXED: use = instead of +=
    C[threadrow*N+threadcol] = temp;
}

int main(){
    int M=128,N=128,K=128;
    size_t bytesA=M*K*sizeof(float);
    size_t bytesB=K*N*sizeof(float);
    size_t bytesC=M*N*sizeof(float);

    float *h_A=(float*)malloc(bytesA);
    float *h_B=(float*)malloc(bytesB);
    float *h_C=(float*)malloc(bytesC);

    for (int i = 0; i < M*K; ++i) h_A[i] = 1.0f; 
    for (int i = 0; i < K*N; ++i) h_B[i] = 1.0f;
    for (int i = 0; i < M*N; ++i) h_C[i] = 0.0f;  // Initialize to 0

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB); 
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice);  // Initialize device C
    
    constexpr int BLOCK = 32;
    
    // CORRECT for 1D blocks: grid uses 2D, block uses 1D with BLOCK*BLOCK threads
    dim3 gridDim(CEIL_DIV(M, BLOCK), CEIL_DIV(N, BLOCK));
    int blockSize = BLOCK * BLOCK;  // 1D block with 1024 threads

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_3<BLOCK><<<gridDim, blockSize>>>(M, N, K, d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double gflops =(2.0 * M * N * K) / (ms / 1e3) / 1e6;
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
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
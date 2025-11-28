// #include <stdio.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <stdlib.h>

// constexpr int WARPSIZE=32;
// int BM=128;
// int BN=128;
// int BK=8;
// template <const int BM, const int BN, const int BK, const int row_stride_a, const int row_stride_b>
// __device__ void load_from_gmem(int num_cols_b, int num_cols_a,const float *matrix_a, const float *matrix_b,float *tile_a, float *tile_b,int inner_row_a, int inner_col_a,int inner_row_b, int inner_col_b) {
//     // Load A tile: BM x BK from global to shared memory
//     // Each thread loads multiple rows using float4 for coalescing
//     for(int offset=0;offset+row_stride_a<BM;offset+=row_stride_a){
//         onst float4 tmp_a = reinterpret_cast<const float4 *>(
//             &matrix_a[(inner_row_a + offset) * num_cols_a + inner_col_a * 4])[0];
        
//         // Store in shared memory with transposed layout for better access pattern
//         tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a + offset] = tmp_a.x;
//         tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a + offset] = tmp_a.y;
//         tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a + offset] = tmp_a.z;
//         tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a + offset] = tmp_a.w;
//     }
//     for (uint offset = 0; offset + row_stride_b <= BK; offset += row_stride_b) {
//         // Load 4 contiguous elements from B as float4 (already well-aligned)
//         reinterpret_cast<float4 *>(
//             &tile_b[(inner_row_b + offset) * BN + inner_col_b * 4])[0] =
//             reinterpret_cast<const float4 *>(
//                 &matrix_b[(inner_row_b + offset) * num_cols_b + inner_col_b * 4])[0];
//     }
//     /*
//         A tile: Loaded with transposed layout in shared memory for coalesced access during computation

//     B tile: Loaded with same layout as global memory

//     float4: 4 elements per load → better memory bandwidth utilization*/
// }

// template <const int BM, const int BN, const int BK, const int WM, const int WN,
//           const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
//           const int TM, const int TN>
// __device__ void process_warp_tile(float *register_m, float *register_n, float *thread_results,
//                                   const float *tile_a, const float *tile_b,
//                                   const uint warp_row, const uint warp_col,
//                                   const uint thread_row_in_warp, const uint thread_col_in_warp) {
//     for(int dot_idx=0;dot_idx<BK;++dot_idx){
//         //load A  fragments from shared memory to registers;
//         for(int wsub_row_idx=0;wsub_row_idx<WMITER;++wsub_row_idx){
//             for(int i=0;i<TM;++i){
//                 int row_in_block=warp_row*WM+wsub_row_idx*TM+thread_row_in_warp*TM+i;
//                 int col_in_block=dot_idx;
//                 register_m[wsub_row_idx*TM+i]=tile_a[col_in_block*BM+row_in_block];
//             }
//         }
//         for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
//             for (uint i = 0; i < TN; ++i) {
//                 // Calculate position in shared memory B tile
//                 uint row_in_block = dot_idx;  // Current K dimension
//                 uint col_in_block = warp_col * WN + wsub_col_idx * WSUBN + thread_col_in_warp * TN + i;
                
//                 // Load from shared memory B[dot_idx, col]
//                 register_n[wsub_col_idx * TN + i] = 
//                     tile_b[row_in_block * BN + col_in_block];
//             }
//         }
//         for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
//             for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
//                 // Multiply A fragment with B fragment and accumulate
//                 for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
//                     for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
//                         // Calculate index in thread_results
//                         uint res_idx = (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
//                                      (wsub_col_idx * TN) + res_idx_n;
                        
//                         // C += A × B accumulation
//                         thread_results[res_idx] += 
//                             register_m[wsub_row_idx * TM + res_idx_m] *
//                             register_n[wsub_col_idx * TN + res_idx_n];
//                     }
//                 }
//             }
//         }

//     }

    
// }



// __global__ void sgemm_warp_tiling_kerenel(const int M,const int N,const int K, float *A,float*B,float* C){

//     const crow=blockIdx.x;
//     const ccol=blockIdx.y;//which output block we are in
//     //Each block computes BMxBN tile of C -Block (i,j) computes C[i*BM : (i+1)*BM][j*BN:(j+1)*BN]
//     A+=crow*BM*K;//point to the first element of the block row of A
//     B+=ccol*BN;//point to the first element of the block column of B
//     C+=crow*BM*N+ccol*BN;//point to the first element of

//     constexpr int WM=64;//divide blocks into warps of size WMxWN
//     constexpr int WN=64;

//     const int warp_idx=threadIdx.x/WARPSIZE; //for eaxmple warp 0-3 for 128 threads.
//     const int warp_col=warp_idx% (BN/WN);//which warp column in the block
//     const int warp_row=warp_idx/ (BN/WN);//which warp row in the block

//     //each warp computes WMxWN tile of C -Warp (i,j) computes C[i*WM:(i+1)*WM][j*WN:(j+1)*WN]
//     C+=warp_row*WM*N+warp_col*WN;//point to the first element of the warp tile of C
//     const int thread_idx_in_warp=threadIdx.x%WARPSIZE;//0-31 didnt we already claculate threadidx?- yes but this is within the warp but we need it for loading  

//     const int TM=4;//thread rows
//     const int TN=4;
//     const int WINTER=2;//warp colmn iterations

//     const int WMITER=WM/(TM*WARPSIZE);//number of thread row iterations per warp

//     const int WSUBM=WM/WINTER;//warp sub matrix rows
//     const int WSUBN=WN/WINTER;//warp sub matrix columns

//     //thread position within warp subtile
//     const int thread_row=thread_idx_in_warp%(WSUBN/TN);//which row within the thread tile --//0-7
//     const int thread_col=thread_idx_in_warp/(WSUBN/TN);//which column within the thread tile--//0-3
//     // Each thread computes TM x TN elements
//     // Thread (i,j) in warp computes:
//     // Rows: warp_row*WM + [thread_row_in_warp*TM : (thread_row_in_warp+1)*TM] across iterations
//     // Cols: warp_col*WN + [thread_col_in_warp*TN : (thread_col_in_warp+1)*TN] across iterations


//     __shared__ float tile_a[BM*BN] ////BMxBK=128 x8;
//     __shared__ float tile_b[BK*BN] ////BKxBN=8x128;

//     const int inner_row_a=threadIdx.x/(BK/4); //// Which row to load in A tile
//     const int inner_col_a=(threadIdx.x%(BK/4))*4; // Which column to load in A tile

//     const int row_stride_a = (blockDim.x * 4) / BK; // How many rows each thread loads
//     const int inner_row_b=threadIdx.x/(BN/4); //// Which row to load in B tile
//     const int inner_col_b=(threadIdx.x%(BN/4))*4; // Which column to load in B tile
//     const int row_stride_b = (blockDim.x * 4) / BN; // How many rows each thread loads

//     float thread_results[WMITER*TM*WNITER*TN]={0};//accumulator for each thread

//     float register_m[WMITER*TM]={0.0f};
//     float register_n[WNITER*TN]={0.0f};

//     /*
//         128 threads collaboratively load 128×8 A tile and 8×128 B tile

//     Each thread loads multiple elements using float4 for coalesced access

//     inner_row_a, inner_col_a determine which part of the tile each thread loads

// */
//     for(int block_k_idx=0;nlock_k_idx<K;block_k_idx+=BK){
//         // Step 5.1: Load current BK slice from global to shared memory
//         load_from_gmem<BM,BN,BK,row_stride_a,row_stride_b>(N,K,A,B,tile_a,tile_b,inner_row_a,inner_col_a,inner_row_b,inner_col_b );
//         __syncthreads();
//         //Process the loaded tiles 
//         process_warp_tile<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
//             register_m, register_n, thread_results, tile_a, tile_b,
//             warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);
//         A+=BK;
//         B+=BK*N;
//         __syncthreads();
//     }
//     for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
//         for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            
//             // Calculate base pointer for this warp subtile in output matrix C
//             float *matrix_c_interim = C + (wsub_row_idx * WSUBM) * N + wsub_col_idx * WSUBN;
            
//             // Write TM × TN thread tile using vectorized stores (float4)
//             for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m += 1) {
//                 for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n += 4) {
                    
//                     // Step 10.1: Calculate global memory position
//                     uint global_row = (wsub_row_idx * WSUBM) + (thread_row_in_warp * TM + res_idx_m);
//                     uint global_col = (wsub_col_idx * WSUBN) + (thread_col_in_warp * TN + res_idx_n);
                    
//                     // Step 10.2: Load existing C values (if beta != 0)
//                     float4 tmp_c = reinterpret_cast<float4 *>(
//                         &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * N +
//                                         thread_col_in_warp * TN + res_idx_n])[0];
                    
//                     // Step 10.3: Calculate result index in thread_results
//                     const int res_idx = (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
//                                       wsub_col_idx * TN + res_idx_n;
                    
//                     // Step 10.4: Apply alpha and beta: C = alpha*A*B + beta*C
//                     tmp_c.x = alpha * thread_results[res_idx + 0] + beta * tmp_c.x;
//                     tmp_c.y = alpha * thread_results[res_idx + 1] + beta * tmp_c.y;
//                     tmp_c.z = alpha * thread_results[res_idx + 2] + beta * tmp_c.z;
//                     tmp_c.w = alpha * thread_results[res_idx + 3] + beta * tmp_c.w;
                    
//                     // Step 10.5: Store back to global memory
//                     reinterpret_cast<float4 *>(
//                         &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * N +
//                                         thread_col_in_warp * TN + res_idx_n])[0] = tmp_c;
//                 }
//             }
//         }
//     }


// }
















/*
## Step 7: Implement the Core Computation - `process_warp_tile`

```cpp
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void process_warp_tile(float *register_m, float *register_n, float *thread_results,
                                  const float *tile_a, const float *tile_b,
                                  const uint warp_row, const uint warp_col,
                                  const uint thread_row_in_warp, const uint thread_col_in_warp) {
    
    // Loop over inner dimension BK (reduction dimension)
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
        
        // Step 7.1: Load A fragments from shared memory to registers
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (uint i = 0; i < TM; ++i) {
                // Calculate position in shared memory A tile
                uint row_in_block = warp_row * WM + wsub_row_idx * WSUBM + thread_row_in_warp * TM + i;
                uint col_in_block = dot_idx;  // Current K dimension
                
                // Load from shared memory A[dot_idx, row] (transposed layout)
                register_m[wsub_row_idx * TM + i] = 
                    tile_a[col_in_block * BM + row_in_block];
            }
        }
        
        // Step 7.2: Load B fragments from shared memory to registers  
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            for (uint i = 0; i < TN; ++i) {
                // Calculate position in shared memory B tile
                uint row_in_block = dot_idx;  // Current K dimension
                uint col_in_block = warp_col * WN + wsub_col_idx * WSUBN + thread_col_in_warp * TN + i;
                
                // Load from shared memory B[dot_idx, col]
                register_n[wsub_col_idx * TN + i] = 
                    tile_b[row_in_block * BN + col_in_block];
            }
        }
        
        // Step 7.3: Matrix multiplication accumulation
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
                // Multiply A fragment with B fragment and accumulate
                for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                    for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                        // Calculate index in thread_results
                        uint res_idx = (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                     (wsub_col_idx * TN) + res_idx_n;
                        
                        // C += A × B accumulation
                        thread_results[res_idx] += 
                            register_m[wsub_row_idx * TM + res_idx_m] *
                            register_n[wsub_col_idx * TN + res_idx_n];
                    }
                }
            }
        }
    }
}
```

## Step 8: Dry Run of `process_warp_tile` with Concrete Example

Let's trace through with actual numbers:

### Initial Setup:
```cpp
WM = 64, WN = 64, WMITER = 4, WNITER = 2
WSUBM = 16, WSUBN = 32, TM = 4, TN = 4
warp_row = 0, warp_col = 0
thread_row_in_warp = 0, thread_col_in_warp = 0
BK = 8
```

### Iteration 1: `dot_idx = 0`

**Loading A fragments:**
```cpp
// Thread (0,0) loads from A tile:
wsub_row_idx=0: rows 0-3   (0*16 + 0*4 + [0,1,2,3])
wsub_row_idx=1: rows 16-19 (0*16 + 1*16 + 0*4 + [0,1,2,3])  
wsub_row_idx=2: rows 32-35
wsub_row_idx=3: rows 48-51
// All from column dot_idx=0 in A tile
```

**Loading B fragments:**
```cpp
// Thread (0,0) loads from B tile:
wsub_col_idx=0: cols 0-3   (0*32 + 0*4 + [0,1,2,3])
wsub_col_idx=1: cols 32-35 (0*32 + 1*32 + 0*4 + [0,1,2,3])
// All from row dot_idx=0 in B tile
```

**Computation:**
```cpp
// Multiply all combinations:
// [4×1] from A × [1×4] from B → [4×4] result tile
// Do this for all wsub_row_idx × wsub_col_idx combinations
// Total: 4 × 2 × 4 × 4 = 128 multiply-accumulate operations per dot_idx
```

### Memory Access Pattern Visualization:

```
Shared Memory A Tile (128×8) - Transposed Layout:
dot_idx=0: [all rows from warp tile] 
dot_idx=1: [all rows from warp tile]
...
dot_idx=7: [all rows from warp tile]

Shared Memory B Tile (8×128):
dot_idx=0: [all cols from warp tile]
dot_idx=1: [all cols from warp tile]  
...
dot_idx=7: [all cols from warp tile]
```

## Step 9: Indexing Scheme Deep Dive

### For Thread (thread_row_in_warp=1, thread_col_in_warp=2):

**Global position in output matrix C:**
```cpp
// Base position from block and warp
base_row = block_row * BM + warp_row * WM  // e.g., 0*128 + 0*64 = 0
base_col = block_col * BN + warp_col * WN  // e.g., 0*128 + 0*64 = 0

// Thread's specific elements:
for wsub_row_idx = 0 to 3:
    for wsub_col_idx = 0 to 1:
        row_start = base_row + wsub_row_idx * WSUBM + thread_row_in_warp * TM
        col_start = base_col + wsub_col_idx * WSUBN + thread_col_in_warp * TN
        
        // Thread (1,2) computes:
        // Rows: [1*16 + 1*4 : 1*16 + 2*4-1] = [20:23] across iterations
        // Cols: [2*32 + 2*4 : 2*32 + 3*4-1] = [72:75] across iterations
```

### Shared Memory Access Pattern:

**A tile access** (transposed layout - column major):
```cpp
tile_a[col_in_block * BM + row_in_block]
// dot_idx * 128 + (warp_row*64 + wsub_row_idx*16 + thread_row*4 + i)
```

**B tile access** (row major):  
```cpp
tile_b[row_in_block * BN + col_in_block]  
// dot_idx * 128 + (warp_col*64 + wsub_col_idx*32 + thread_col*4 + i)
```

This completes the core computation! Should I continue with the final step - writing results back to global memory?*/

































#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <math.h>

constexpr int WARPSIZE = 32;

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// FIXED: Make row_stride_a and row_stride_b template parameters instead of constexpr
template <const int BM, const int BN, const int BK, const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int num_cols_b, int num_cols_a, 
                               const float *matrix_a, const float *matrix_b,
                               float *tile_a, float *tile_b,
                               int inner_row_a, int inner_col_a,
                               int inner_row_b, int inner_col_b) {
    // Load A tile: BM x BK from global to shared memory
    for(int offset = 0; offset < BM; offset += row_stride_a) {
        const float4 tmp_a = reinterpret_cast<const float4 *>(
            &matrix_a[(inner_row_a + offset) * num_cols_a + inner_col_a * 4])[0];
        
        // Store in shared memory with transposed layout
        tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a + offset] = tmp_a.x;
        tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a + offset] = tmp_a.y;
        tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a + offset] = tmp_a.z;
        tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a + offset] = tmp_a.w;
    }
    
    // Load B tile: BK x BN from global to shared memory  
    for (int offset = 0; offset < BK; offset += row_stride_b) {
        reinterpret_cast<float4 *>(
            &tile_b[(inner_row_b + offset) * BN + inner_col_b * 4])[0] =
            reinterpret_cast<const float4 *>(
                &matrix_b[(inner_row_b + offset) * num_cols_b + inner_col_b * 4])[0];
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void process_warp_tile(float *register_m, float *register_n, float *thread_results,
                                  const float *tile_a, const float *tile_b,
                                  const int warp_row, const int warp_col,
                                  const int thread_row_in_warp, const int thread_col_in_warp) {
    for(int dot_idx = 0; dot_idx < BK; ++dot_idx) {
        // Load A fragments from shared memory to registers
        for(int wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for(int i = 0; i < TM; ++i) {
                int row_in_block = warp_row * WM + wsub_row_idx * WSUBM + thread_row_in_warp * TM + i;
                int col_in_block = dot_idx;
                register_m[wsub_row_idx * TM + i] = tile_a[col_in_block * BM + row_in_block];
            }
        }
        
        // Load B fragments from shared memory to registers
        for (int wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            for (int i = 0; i < TN; ++i) {
                int row_in_block = dot_idx;
                int col_in_block = warp_col * WN + wsub_col_idx * WSUBN + thread_col_in_warp * TN + i;
                register_n[wsub_col_idx * TN + i] = tile_b[row_in_block * BN + col_in_block];
            }
        }
        
        // Matrix multiplication accumulation
        for (int wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (int wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
                for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                    for (int res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                        int res_idx = (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                    (wsub_col_idx * TN) + res_idx_n;
                        thread_results[res_idx] += 
                            register_m[wsub_row_idx * TM + res_idx_m] *
                            register_n[wsub_col_idx * TN + res_idx_n];
                    }
                }
            }
        }
    }
}

// FIXED: Make the kernel template with NUM_THREADS as a parameter
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void sgemm_warp_tiling_kernel(const int M, const int N, const int K, 
                                        float alpha, const float *A, const float *B,
                                        float beta, float *C) {
    const int crow = blockIdx.y;  // block row
    const int ccol = blockIdx.x;  // block column
    
    // Each block computes BMxBN tile of C
    A += crow * BM * K;
    B += ccol * BN;  
    C += crow * BM * N + ccol * BN;

    const int warp_idx = threadIdx.x / WARPSIZE;
    const int warp_col = warp_idx % (BN / WN);
    const int warp_row = warp_idx / (BN / WN);

    // Each warp computes WMxWN tile of C
    C += warp_row * WM * N + warp_col * WN;
    const int thread_idx_in_warp = threadIdx.x % WARPSIZE;

    // FIXED: Correct WMITER calculation
    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    // FIXED: Corrected thread_row/thread_col assignment
    const int thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);  // 0-3
    const int thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);  // 0-7

    // FIXED: Correct shared memory sizes
    __shared__ float tile_a[BM * BK];  // BM x BK = 128 x 8
    __shared__ float tile_b[BK * BN];  // BK x BN = 8 x 128

    // FIXED: Don't multiply by 4 in inner_col calculation
    const int inner_row_a = threadIdx.x / (BK / 4);
    const int inner_col_a = threadIdx.x % (BK / 4);

    // FIXED: Calculate row_strides at compile time using NUM_THREADS
    constexpr int row_stride_a = (NUM_THREADS * 4) / BK;
    
    const int inner_row_b = threadIdx.x / (BN / 4);
    const int inner_col_b = threadIdx.x % (BN / 4);
    
    constexpr int row_stride_b = (NUM_THREADS * 4) / BN;

    float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
    float register_m[WMITER * TM] = {0.0f};
    float register_n[WNITER * TN] = {0.0f};

    // Main K loop
    for(int block_k_idx = 0; block_k_idx < K; block_k_idx += BK) {
        // FIXED: Use the template parameters correctly
        load_from_gmem<BM, BN, BK, row_stride_a, row_stride_b>(
            N, K, A, B, tile_a, tile_b, 
            inner_row_a, inner_col_a, inner_row_b, inner_col_b);
        
        __syncthreads();

        process_warp_tile<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            register_m, register_n, thread_results, tile_a, tile_b,
            warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);

        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // Write results back
    for (int wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
        for (int wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            float *matrix_c_interim = C + (wsub_row_idx * WSUBM) * N + wsub_col_idx * WSUBN;
            
            for (int res_idx_m = 0; res_idx_m < TM; res_idx_m += 1) {
                for (int res_idx_n = 0; res_idx_n < TN; res_idx_n += 4) {
                    float4 tmp_c = reinterpret_cast<float4 *>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * N +
                                        thread_col_in_warp * TN + res_idx_n])[0];
                    
                    const int res_idx = (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                      wsub_col_idx * TN + res_idx_n;
                    
                    // Apply: C = alpha*A*B + beta*C
                    tmp_c.x = alpha * thread_results[res_idx + 0] + beta * tmp_c.x;
                    tmp_c.y = alpha * thread_results[res_idx + 1] + beta * tmp_c.y;
                    tmp_c.z = alpha * thread_results[res_idx + 2] + beta * tmp_c.z;
                    tmp_c.w = alpha * thread_results[res_idx + 3] + beta * tmp_c.w;
                    
                    reinterpret_cast<float4 *>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * N +
                                        thread_col_in_warp * TN + res_idx_n])[0] = tmp_c;
                }
            }
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
    
    // Kernel parameters - use the same as in kernel
    constexpr int BM = 128, BN = 128, BK = 8;
    constexpr int WM = 64, WN = 64, TM = 4, TN = 4, WNITER = 2;
    
    // Calculate thread block size
    constexpr int warps_per_block = (BM / WM) * (BN / WN);  // 2 * 2 = 4
    constexpr int NUM_THREADS = warps_per_block * WARPSIZE;  // 4 * 32 = 128
    
    // Grid dimensions
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockSize(NUM_THREADS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    // FIXED: Launch with all template parameters
    sgemm_warp_tiling_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockSize>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double gflops = (2.0 * M * N * K) / (ms / 1e3) / 1e9;
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
        if (fabs(h_C[i] - K) > 1e-2) {
            printf("Error at index %d: got %.1f, expected %d\n", i, h_C[i], K);
            correct = false;
            if (i > 10) break;
        }
    }
    printf("Results: %s\n", correct ? "CORRECT" : "INCORRECT");

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
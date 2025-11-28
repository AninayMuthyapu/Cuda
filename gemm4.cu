#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))


template< const int BM, const int BN, const int BK ,const int TM>
__global__ void sgemm4(int M,int N,int K,float *A,float *B,float *C){
    __shared__ float A_s[BM*BK];
    __shared__ float B_s[BK*BN];

    const int crow=blockIdx.x;
    const int ccol=blockIdx.y;

    const int threadrow=threadIdx.x/BN;
    const int threadcol=threadIdx.x%BN;

    A+=cRow*BM*K;
    B+=cCol*BN;
    C+=cRow*BM*N+cCol*BN;


    const int innercolA=threadIdx.x%BK;
    const int innerrowB=threadIdx.x/BN;
    const int innercolB=threadIdx.x%BN;
    const int innerrowA=threadIdx.x/BK;

    float temp=0;
    for(int bkIdx=0;bkIdx<K;bkIdx+=BK){
        //load A and B to shared memory
        A_s[threadrow*BK+innercolA]=A[threadrow*K+innercolA];
        B_s[innerrowB*BN+innercolB]=B[innerrowB*N+innercolB];
        __syncthreads();
        A+=BK;
        B+=BK*N;
        //compute
        for(int dotIdx=0;dotIdx<BK;++dotIdx){
            // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }
    for(int resIdx=0;resIdx<TM;++resIdx){
        C[(threadrow*TM+resIdx)*N+threadcol]=temp;
    }
}
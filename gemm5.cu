#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


template<const int BM.const int BN, const int BK, cosnt int TM,const int TN>
__global__ void sgemm_5(int M,int N,int K,float *A,float *B,float *C){
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];
    const int crow=blockIdx.x;
    const int ccol=blockIdx.y;
    const int threadResultsBlock=BM*BN;
    const int numtbt=threadResultsBlock/(TM*TN);//number of threads in blocktile,A thread will calculate TM*TN elements in the block tile.

    const int threadrow=threadIdx.x/(BN/TN);
    const int threadcol=threadIdx.x%(BN/TN);
    A+=crow*BM*K;
    B+=ccol*BN;
    C+=crow*BM*N+ccol*BN;
    //caculating no.of rows of A 
    const int strideA= numtbt/BK;
    const int strideB= numtbt/BN;
    const int innerrowA=threadIdx.x/BK;
    const int innercolA=threadIdx.x%BK;
    
    const int innerrowB=threadIdx.x/BN;
    const int innercolB=threadIdx.x%BN;

    float threadResults[TM*TN]={0.0}
    float reg[TM]={0.0};
    float reg[TN]={0.0};
    for(int bkIdx=0;bkIdx<K;bkIdx+=BK){
        for(int loadoffset=0;loadoffset<BM;loadoffset+=strideA){
            As[(innerrowA+loadoffset)*BK+innercolA]=A[(innerrowA+loadoffset)*K+bkIdx+innercolA];
        }
        for(int loadoffset=0;loadoffset<BK;loadoffset+=strideB){
            Bs[(innerrowB+loadoffset)*BN+innercolB]=B[(innerrowB+loadoffset)*N+bkIdx+innercolB];
        }
        __syncthread();
        A+=BK;
        B+=BK*N;
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      
        for (uint i = 0; i < TM; ++i) {
            regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
        }
        for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
        }
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[resIdxM * TN + resIdxN] +=
                regM[resIdxM] * regN[resIdxN];
            }
        }
    }
    __syncthreads();
    }
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =threadResults[resIdxM * TN + resIdxN] + C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}
    
#include <cstdio>
#include <cstdlib>


__global__ void sgemm_naive(int M,int N,int K ,const float *A ,const float *B,  float *C){

// compute position in C that this thread is responsible for
const  int row=blockDim.y*blockIdx.y+threadIdx.y;
const int col=blockDim.x*blockIdx.x+threadIdx.x;


if(row<M && col<N ){
    float tmp=0.0f;
    for(int i=0;i<K;++i){
        tmp+=A[row*K+i]*B[i*N+col];
    }
    C[row*N+col] =tmp;
}
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
float *d_A,*d_B,*d_C;
cudaMalloc(&d_A, bytesA);
cudaMalloc(&d_B, bytesB); 
cudaMalloc(&d_C, bytesC);

cudaMemcpy(d_A,h_A,bytesA,cudaMemcpyHostToDevice);
cudaMemcpy(d_B,h_B,bytesB,cudaMemcpyHostToDevice);
dim3 block(16,16);//256 threads per block
dim3 grid((N+block.x-1)/block.x,(M+block.y-1)/block.y);
sgemm_naive<<<grid,block>>>(M,N,K,d_A,d_B,d_C);
cudaDeviceSynchronize();
cudaMemcpy(h_C,d_C,bytesC,cudaMemcpyDeviceToHost);
printf("C[0,0] = %f (expected %d)\n", h_C[0], K);
cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
free(h_A); free(h_B); free(h_C);
return 0;



}






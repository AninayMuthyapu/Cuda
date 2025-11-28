#include <cstdio>
#include <cuda_runtime.h>

__global__  void hello_kernel(){
    printf("hello from block %d ,thread %d\n",blockIdx.x,threadIdx.x);
}

int main(){
    hello_kernel<<<2,4>>>(); //2 blocks of 4 threads each 
    cudaDeviceSynchronize();
    int dev;// this creates an integer variable that stores teh curremnt GPU device ID.
    cudaGetDevice(&dev);
    cudaDeviceProperties(&prop,dev);
    printf("Device: %s, SMs: %d, maxThreadsPerBlock: %d\n",
           prop.name, prop.multiProcessorCount, prop.maxThreadsPerBlock);
    return 0;
}
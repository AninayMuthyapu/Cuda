#include <stdio.h>
#include "cuda.h"
#include "book.h"        // For HANDLE_ERROR and CPUBitmap
#include "cpu_bitmap.h"  // For bitmap display

#define DIM 1000

// Complex number struct
struct cuComplex {
    float r;
    float i;

    __host__ __device__
    cuComplex(float a, float b) : r(a), i(b) {}

    __host__ __device__
    float magnitude2(void) const {
        return r * r + i * i;
    }

    __host__ __device__
    cuComplex operator*(const cuComplex& a) const {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    __host__ __device__
    cuComplex operator+(const cuComplex& a) const {
        return cuComplex(r + a.r, i + a.i);
    }
};

// Julia set function
__device__ int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x) / (DIM/2);
    float jy = scale * (float)(DIM/2 - y) / (DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

// CUDA kernel
__global__ void kernel(unsigned char *ptr) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    int juliaValue = julia(x, y);

    ptr[offset * 4 + 0] = 255 * juliaValue;  // R
    ptr[offset * 4 + 1] = 0;                // G
    ptr[offset * 4 + 2] = 0;                // B
    ptr[offset * 4 + 3] = 255;              // A
}

int main(void) {
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    dim3 grid(DIM, DIM);

    kernel<<<grid, 1>>>(dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
                            bitmap.image_size(),
                            cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    HANDLE_ERROR(cudaFree(dev_bitmap));
    return 0;
}

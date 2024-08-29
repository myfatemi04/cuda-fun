#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

struct mat2d_t {
    float *data; // data[i, j] = data[i * w + j]
    int h, w;
};

mat2d_t init_mat2d_cu(int h, int w) {
    float *data;
    cudaMalloc(&data, h * w * sizeof(float));
    return {data, h, w};
}
void free_mat2d_cu(mat2d_t mat) { cudaFree(mat.data); }

mat2d_t init_mat2d_cpu(int h, int w) {
    float *data = (float *) malloc(h * w * sizeof(float));
    // printf("data size: %d\n", h * w);
    return {data, h, w};
}
void free_mat2d_cpu(mat2d_t mat) { free(mat.data); }

__global__ void matmul_cu_kernel(const mat2d_t mat1, const mat2d_t mat2, mat2d_t out) {
    // Initialize shared memory.
    __shared__ float smem[256];
    int tile_i = threadIdx.y * 16 + threadIdx.x;
    smem[tile_i] = 0;
    // __syncthreads(); // not necessary actually.

    int i = blockIdx.y * 16 + threadIdx.y;
    int k = blockIdx.x * 16 + threadIdx.x;

    // possible to go out of bounds because of blockidx
    if (k < out.w && i < out.h) {
        // sum_j (a[i, j] * b[j, k]) -> c[i, k]
        for (int j = 0; j < mat1.w; j++) {
            smem[tile_i] += __ldg(&mat1.data[i * mat1.w + j]) * __ldg(&mat2.data[j * mat2.w + k]);
        }
    }
    __syncthreads();

    // copy to global memory
    if (k < out.w && i < out.h) {
        out.data[i * out.w + k] = smem[tile_i];
    }
}

__global__ void matmul_cu_kernel_global(const mat2d_t mat1, const mat2d_t mat2, mat2d_t out) {
    int i = blockIdx.y * 16 + threadIdx.y;
    int k = blockIdx.x * 16 + threadIdx.x;

    // possible to go out of bounds because of blockidx
    if (k < out.w && i < out.h) {
        // sum_j (a[i, j] * b[j, k]) -> c[i, k]
        out.data[i * out.w + k] = 0;
        for (int j = 0; j < mat1.w; j++) {
            out.data[i * out.w + k] += mat1.data[i * mat1.w + j] * mat2.data[j * mat2.w + k];
        }
    }
    __syncthreads();
}

mat2d_t matmul_cu(mat2d_t mat1, mat2d_t mat2) {
    // creates a tiled kernel. the tiling is in the output space.
    mat2d_t out = init_mat2d_cu(mat1.h, mat2.w);

    // can have at most {2^31 - 1, 65535, 65535} blocks in the {x, y, z} dimensions of a grid.
    // in theory if we have a massive matrix we will need to manually run multiple cuda kernels and then synchronize them at the end.
    int grid_w = (15 + out.w) / 16;
    int grid_h = (15 + out.h) / 16;
    dim3 gridSize(grid_w, grid_h);
    dim3 blockSize(16, 16);

    matmul_cu_kernel<<<gridSize, blockSize>>>(mat1, mat2, out);

    return out;
}

mat2d_t matmul_cu_global(mat2d_t mat1, mat2d_t mat2) {
    // creates a tiled kernel. the tiling is in the output space.
    mat2d_t out = init_mat2d_cu(mat1.h, mat2.w);

    // can have at most {2^31 - 1, 65535, 65535} blocks in the {x, y, z} dimensions of a grid.
    // in theory if we have a massive matrix we will need to manually run multiple cuda kernels and then synchronize them at the end.
    int grid_w = (15 + out.w) / 16;
    int grid_h = (15 + out.h) / 16;
    dim3 gridSize(grid_w, grid_h);
    dim3 blockSize(16, 16);

    matmul_cu_kernel_global<<<gridSize, blockSize>>>(mat1, mat2, out);

    return out;
}

mat2d_t mat_to_cpu(mat2d_t mat_cu) {
    mat2d_t mat_cpu = init_mat2d_cpu(mat_cu.h, mat_cu.w);
    cudaMemcpy(mat_cpu.data, mat_cu.data, mat_cu.w * mat_cu.h * sizeof(float), cudaMemcpyDeviceToHost);
    free_mat2d_cu(mat_cu);
    return mat_cpu;
}

mat2d_t mat_to_cu(mat2d_t mat_cpu) {
    mat2d_t mat_cu = init_mat2d_cu(mat_cpu.h, mat_cpu.w);
    cudaMemcpy(mat_cu.data, mat_cpu.data, mat_cpu.w * mat_cpu.h * sizeof(float), cudaMemcpyHostToDevice);
    free_mat2d_cpu(mat_cpu);
    return mat_cu;
}

mat2d_t matmul_cpu(mat2d_t mat1, mat2d_t mat2) {
    mat2d_t out = init_mat2d_cpu(mat1.h, mat2.w);

    for (int i = 0; i < out.h; i++) {
        for (int j = 0; j < out.w; j++) {
            out.data[i * out.w + j] = 0;
            for (int k = 0; k < mat1.w; k++) {
                // if (i * out.w + j >= out.w * out.h) {
                //     printf("Out of bounds??? [1]\n");
                // }
                // if (i * mat1.w + k >= mat1.w * mat1.h) {
                //     printf("Out of bounds??? [2]\n");
                // }
                // if (k * mat2.w + j >= mat2.w * mat2.h) {
                //     printf("Out of bounds??? [3]\n");
                // }
                // printf("%f\n", mat2.data[0]);
                // printf("%d %d %d\n", i * out.w + j, i * mat1.w + k, k * mat2.w + j);
                out.data[i * out.w + j] += mat1.data[i * mat1.w + k] * mat2.data[k * mat2.w + j];
            }
        }
    }

    return out;
}

mat2d_t random_mat_cpu(int h, int w) {
    mat2d_t out = init_mat2d_cpu(h, w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            out.data[i * out.w + j] = rand() / RAND_MAX;
        }
    }
    return out;
}

bool almost_equal(mat2d_t a, mat2d_t b, float eps) {
    for (int i = 0; i < a.h; i++) {
        for (int j = 0; j < a.w; j++) {
            double diff = a.data[i * a.w + j] - b.data[i * a.w + j];
            if (!(diff < eps && diff > -eps)) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    mat2d_t mat1 = random_mat_cpu(1000, 2000);
    mat2d_t mat2 = random_mat_cpu(2000, 1000);
    
    clock_t custart, cuend, cpustart, cpuend;
    cpustart = clock();
    mat2d_t result = matmul_cpu(mat1, mat2);
    cpuend = clock();
    printf("CPU matmul speed: %.3f\n", (float) (cpuend - cpustart) / CLOCKS_PER_SEC);
    
    custart = clock();
    mat2d_t mat1_cu = mat_to_cu(mat1), mat2_cu = mat_to_cu(mat2);
    mat2d_t result_cu = matmul_cu(mat1_cu, mat2_cu);
    cuend = clock();

    printf("CUDA matmul speed: %.3f\n", (float) (cuend - custart) / CLOCKS_PER_SEC);

    custart = clock();
    mat2d_t result_cu_global = matmul_cu_global(mat1_cu, mat2_cu);
    cuend = clock();

    printf("CUDA matmul speed [global memory]: %.3f\n", (float) (cuend - custart) / CLOCKS_PER_SEC);

    mat2d_t result_cu_cpu = mat_to_cpu(result_cu);
    mat2d_t result_cu_global_cpu = mat_to_cpu(result_cu_global);

    printf("Equal: %d\n", almost_equal(result, result_cu_cpu, 1e-8));
    printf("Equal [global memory]: %d\n", almost_equal(result, result_cu_global_cpu, 1e-8));
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CALL(call)                                                   \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",     \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);    \
            exit(1);                                                      \
        }                                                                 \
    }

__global__ void countBins(const int *input, int *count, int N, int K) {
    extern __shared__ int count_shared[];
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Clear the shared memory.
    if (threadIdx.x < K) { count_shared[threadIdx.x] = 0; }
    __syncthreads();

    if (i < N) {
        // Count the values. __ldg is a "read-only data cache load" function for faster access.
        atomicAdd(&count_shared[__ldg(&input[i])], 1);
    }
    __syncthreads();
    
    // Parallel reduction sum
    if (threadIdx.x < K) {
        int bucket = threadIdx.x;
        atomicAdd(&count[bucket], count_shared[bucket]);
    }
    __syncthreads();

    return;
}

__global__ void runPsum(const int *count, int *psum, int K) {
    // Prefix-sum.
    psum[0] = count[0] - 1;
    for (int i = 1; i < K; i++) {
        psum[i] = psum[i - 1] + count[i];
    }

    // for (int i = 0; i < K; i++) {
    //     printf("count[%d]: %d\n", i, count[i]);
    // }
}

__global__ void expandCounts(const int *psum, int *output, int N, int K) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Sort.
    if (i < N) {
        int low = 0;
        int high = K;
        while (low < high) {
            int mid = (low + high) / 2;
            if (i <= psum[mid]) {
                high = mid;
            } else if (i > psum[mid]) {
                low = mid + 1;
            } else {
                break;
            }
        }
        // printf("i = %d, s[mid] = %d\n", i, s[low+K]);
        // Now, low == high.
        output[i] = low;
    }
}

// from stackoverflow
int compare( const void* a, const void* b)
{
    int int_a = * ( (int*) a );
    int int_b = * ( (int*) b );

    // an easy expression for comparing
   return (int_a > int_b) - (int_a < int_b);
}

int main() {
    // int N = 9;
    // int K = 10;
    // int values[N] = {3, 3, 2, 1, 0, 4, 4, 2, 3};

    srand(time(NULL));

    int K = 100;
    // int N = 1024; // 1024 * 1024;
    int N = 1024 * 1024;
    int grid_dim = (N + 1023) / 1024;
    int block_dim = 1024;
    int *values = (int *) malloc(sizeof(int) * N);
    int counts[100] = {0};
    for (int i = 0; i < N; i++) {
        // Choose a random integer 0 ... K - 1.
        values[i] = rand() % K;
        counts[values[i]]++;
    }
    // for (int i = 0; i < K; i++) {
    //     printf("true_count[%d] = %d\n", i, counts[i]);
    // }

    clock_t custart, cuend, cpustart, cpuend;
    
    // Mallocs take a while but sort of get amortized
    int* values_gpu;
    int* result_gpu;
    int* count_gpu;
    int* psum_gpu;
    cudaMalloc(&values_gpu, sizeof(int) * N);
    cudaMalloc(&result_gpu, sizeof(int) * N);
    cudaMalloc(&count_gpu, sizeof(int) * K);
    cudaMalloc(&psum_gpu, sizeof(int) * K);
    cudaMemset(count_gpu, 0, sizeof(int) * K);
    cudaMemset(psum_gpu, 0, sizeof(int) * K);

    custart = clock();

    cudaMemcpy(values_gpu, values, sizeof(int) * N, cudaMemcpyHostToDevice);

    countBins<<<grid_dim, block_dim, K * sizeof(int)>>>(values_gpu, count_gpu, N, K);
    cudaDeviceSynchronize();
    runPsum<<<1, 1>>>(count_gpu, psum_gpu, K);
    cudaDeviceSynchronize();
    expandCounts<<<grid_dim, block_dim, K * sizeof(int)>>>(psum_gpu, result_gpu, N, K);
    cuend = clock();

    CUDA_CALL(cudaGetLastError());

    cpustart = clock();
    qsort(values, N, sizeof(int), compare);
    cpuend = clock();

    printf("CUDA: %.3f\n", (double) (cuend - custart) / CLOCKS_PER_SEC);
    printf("CPU: %.3f\n", (double) (cpuend - cpustart) / CLOCKS_PER_SEC);

    int* result = (int*) malloc(sizeof(int) * N);
    cudaMemcpy(result, result_gpu, sizeof(int) * N, cudaMemcpyDeviceToHost);

    bool match = true;
    for (int i = 0; i < N; i++) {
        // printf("sorted_cuda[i] = %d\n", result[i]);
        if (result[i] != values[i]) {
            printf("value at i = %d does not match. true[i] = %d, got[i] = %d\n", i, values[i], result[i]);
            match = false;
            break;
        }
    }
    if (match) {
        printf("match :)\n");
    } else {
        printf("mismatch ...\n");
    }
}

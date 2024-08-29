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

__global__ void countSort(const int *input, int *result, int *count, int *psum, int N, int K) {
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
    if (threadIdx.x == 0) {
        printf("final result: count[0] : %d\n", count[0]);
    }

    // Prefix-sum.
    if (i == 0) {
        psum[0] = count[0];
        for (int j = 1; j < K; j++) {
            psum[j] = psum[j - 1] + count[j];
        }
    }
    __syncthreads();

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
    result[i] = low;
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
    for (int i = 0; i < K; i++) {
        printf("true_count[%d] = %d\n", i, counts[i]);
    }

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
    cudaMemcpy(values_gpu, values, sizeof(int) * N, cudaMemcpyHostToDevice);

    countSort<<<grid_dim, block_dim, K * sizeof(int)>>>(values_gpu, result_gpu, count_gpu, psum_gpu, N, K);
    CUDA_CALL(cudaGetLastError());

    qsort(values, N, sizeof(int), compare);

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

# -PCA-Implement-Matrix-Multiplication-using-CUDA-C.-Find-the-elapsed-time.
Implement Matrix Multiplication using GPU.

## Aim:
To implement Matrix Multiplication using GPU.

## Procedure:
### Step 1 : 
Include the required files and library.
### Step 2 :
Declare the block size and the size of elements .
### Step 3 : 
Introduce Kernel function to perform matrix multiplication.In the kernal function,decalre the row column size and initialize the sum to be 0,then using for loop calculate the sum.
### Step 4 :
Intoduce a Main function, in the main method declare the required variables and Initialize the matrices 'a' and 'b'.Allocate memory on the device and then copy the input matrices from host to device memory and set the grid and block sizes . Launch the kernel,Copy the result matrix from device to host memory ,Print the result matrix and the elapsed time followed by freeing the device memory.
### Step 5 :
Save the program and execute it . 
## Program :
```
DEVELOPED BY : PradeepRaj.P
REGISTER NO : 212222240073

#include <stdio.h>
#include <sys/time.h>

#define SIZE 4
#define BLOCK_SIZE 2

// Kernel function to perform matrix multiplication
__global__ void matrixMultiply(int *a, int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < size; ++k)
    {
        sum += a[row * size + k] * b[k * size + col];
    }

    c[row * size + col] = sum;
}

int main()
{
    int a[SIZE][SIZE], b[SIZE][SIZE], c[SIZE][SIZE];
    int *dev_a, *dev_b, *dev_c;
    int size = SIZE * SIZE * sizeof(int);

    // Initialize matrices 'a' and 'b'
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    // Copy input matrices from host to device memory
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 dimGrid(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Launch kernel
    matrixMultiply<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, SIZE);

    // Copy result matrix from device to host memory
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    // Stop timer
    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    // Print the result matrix
    printf("Result Matrix:\n");
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    // Print the elapsed time
    printf("Elapsed Time: %.6f seconds\n", elapsed_time);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
```
## Output:
```
root@MidPC:/home/student/Desktop# nvcc first.cu
root@MidPC:/home/student/Desktop# ./a.out
Result Matrix:
14 8 2 -4 
20 10 0 -10 
26 12 -2 -16 
32 14 -4 -22 
Elapsed Time: 0.000023 seconds
root@MidPC:/home/student/Desktop# nvprof ./a.out
==18221== NVPROF is profiling process 18221, command: ./a.out
Result Matrix:
14 8 2 -4 
20 10 0 -10 
26 12 -2 -16 
32 14 -4 -22 
Elapsed Time: 0.000037 seconds
==18221== Profiling application: ./a.out
==18221== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   39.90%  2.5280us         1  2.5280us  2.5280us  2.5280us  matrixMultiply(int*, int*, int*, int)
                   38.89%  2.4640us         2  1.2320us     928ns  1.5360us  [CUDA memcpy HtoD]
                   21.21%  1.3440us         1  1.3440us  1.3440us  1.3440us  [CUDA memcpy DtoH]
      API calls:   99.38%  126.78ms         3  42.262ms  2.2600us  126.78ms  cudaMalloc
                    0.28%  356.84us         1  356.84us  356.84us  356.84us  cuDeviceTotalMem
                    0.20%  252.08us        97  2.5980us     210ns  107.52us  cuDeviceGetAttribute
                    0.07%  87.360us         3  29.120us  2.5700us  79.360us  cudaFree
                    0.03%  36.470us         1  36.470us  36.470us  36.470us  cuDeviceGetName
                    0.02%  29.180us         3  9.7260us  5.9900us  12.080us  cudaMemcpy
                    0.02%  23.180us         1  23.180us  23.180us  23.180us  cudaLaunchKernel
                    0.00%  4.5900us         1  4.5900us  4.5900us  4.5900us  cuDeviceGetPCIBusId
                    0.00%  2.4000us         3     800ns     250ns  1.8100us  cuDeviceGetCount
                    0.00%     930ns         2     465ns     210ns     720ns  cuDeviceGet
                    0.00%     310ns         1     310ns     310ns     310ns  cuDeviceGetUuid
root@MidPC:/home/student/Desktop# 106 
```
![WhatsApp Image 2023-05-26 at 9 39 17 AM](https://github.com/SOWMIYA2003/-PCA-Implement-Matrix-Multiplication-using-CUDA-C.-Find-the-elapsed-time./assets/93427443/7f42d036-3357-48c9-90f0-87fe9337823b)
## Result:
The implementation of Matrix Multiplication using GPU is done successfully.

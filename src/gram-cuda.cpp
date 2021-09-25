 
/*  
 *  How to compile (assume cuda is installed at /usr/local/cuda/)
 *  nvcc -c -I/usr/local/cuda/include gesvdp_example.cpp
 *  nvcc -o a.out gesvdp_example.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
 */
#include <algorithm>

#include <iostream>

#include <stdlib.h>
#include <ctype.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

void gram( 
        double * h_A,
        double * h_C,
        const int64_t m,
        const int64_t n
    ) 
{
    cudaError_t cudaStat ;

    cublasHandle_t handle = NULL;

    cublasStatus_t status = cublasCreate(&handle);
    assert( CUBLAS_STATUS_SUCCESS == status );

    double *d_A    = NULL;
    double *d_B    = NULL;
    double *d_C    = NULL;

    cudaStat = cudaMalloc((void **) &d_A, sizeof(double)*m*n);
    assert( cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void **) &d_B, sizeof(double)*m*n);
    assert( cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void **) &d_C, sizeof(double)*m*n );
    assert( cudaSuccess == cudaStat);
    
   	/* copy input matrix from host to device memory */
    cudaStat = cudaMemcpy(d_A, h_A, sizeof(double)*m*n, cudaMemcpyHostToDevice);
    assert( cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(d_B, h_A, sizeof(double)*m*n, cudaMemcpyHostToDevice);
    assert( cudaSuccess == cudaStat);

    cudaStat = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat);

    double alpha = 1.0;
	double beta = 0.0;

    status = cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        m, m, n,
        &alpha,
        d_A, m,
        d_B, n,
        &beta,
        d_C, m ) ;
    assert( CUBLAS_STATUS_SUCCESS == status );

    cudaStat = cudaMemcpy(h_C, d_C, m*n*sizeof(double), cudaMemcpyDeviceToHost);
    assert( cudaSuccess == cudaStat );

    cudaStat = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat );

    if ( d_A ) { cudaFree( d_A ); }
    if ( d_B ) { cudaFree( d_B ); }
    if ( d_C ) { cudaFree( d_C ); }

    if( handle ) cublasDestroy( handle ) ;
}


        

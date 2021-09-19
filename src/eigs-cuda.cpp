 
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
#include <cusolverDn.h>



void eigs( 
        double * h_S,   // output eigenvalues
        double * h_V,   // output eigen vectors
        double * h_A,   // input matrix ( symmetric = upper )
        const int64_t m // m x m matrix
    ) 
{
    cusolverStatus_t status ;    
    cudaError_t cudaStat ;

    cusolverDnHandle_t handle = nullptr;
    cusolverDnParams_t params = nullptr;

    void *d_A    = nullptr ;
    void *d_S    = nullptr;
    void *d_V    = nullptr; 
	int  *d_info = nullptr;
    int   h_info = 0;
    int64_t h_meig ; // Num eigs found

    double *d_work = nullptr;
    double *h_work = nullptr;
    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;


    status = cusolverDnCreate(&handle);
    assert( CUSOLVER_STATUS_SUCCESS == status );

    status = cusolverDnCreateParams(&params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat = cudaMalloc((void **) &d_A, sizeof(double)*m*m);
    assert( cudaSuccess == cudaStat);

    cudaStat = cudaMalloc((void **) &d_V, sizeof(double)*m*m);
    assert( cudaSuccess == cudaStat);

    cudaStat = cudaMalloc((void **) &d_S, sizeof(double)*m);
    assert( cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void **) &d_info, sizeof(int));

	/* copy input matrix from host to device memory */
    cudaStat = cudaMemcpy(d_A, h_A, sizeof(double)*m*m, cudaMemcpyHostToDevice);
    assert( cudaSuccess == cudaStat);
    cudaStat = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat);
 
    status = cusolverDnXsyevdx_bufferSize(
        handle,
        params,
        CUSOLVER_EIG_MODE_VECTOR,
        CUSOLVER_EIG_RANGE_ALL,
        CUBLAS_FILL_MODE_LOWER,
        m,
        CUDA_R_64F,
        d_A,
        m,
        nullptr,
        nullptr,
        0,
        0,
        &h_meig,
        CUDA_R_64F,
        d_S,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost
    ) ;

    assert( status == CUSOLVER_STATUS_SUCCESS );

    h_work = (double*) malloc ( workspaceInBytesOnHost);
    assert(h_work != NULL );

    cudaStat = cudaMalloc((void **) &d_work, workspaceInBytesOnDevice);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat);
 
    uint64_t dummy ;

    status = cusolverDnXsyevdx (
        handle,
        params,
        CUSOLVER_EIG_MODE_VECTOR,
        CUSOLVER_EIG_RANGE_ALL,
        CUBLAS_FILL_MODE_LOWER,
        m,
        CUDA_R_64F,
        d_A,
        m,
        &dummy,
        &dummy,
        0,
        0,
        &h_meig,
        CUDA_R_64F,
        d_S,
        CUDA_R_64F,
        d_work ,
        workspaceInBytesOnDevice,
        h_work ,
        workspaceInBytesOnHost,    
        d_info ) ;
    if( status != CUSOLVER_STATUS_SUCCESS ) {
        cudaStat = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        assert( cudaSuccess == cudaStat );
        assert( 0 == h_info ) ;
    }

    cudaStat = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat );

	/* copy info from device to host and sync */
    cudaStat = cudaMemcpy(h_S, d_S, m * sizeof(double), cudaMemcpyDeviceToHost);
    assert( cudaSuccess == cudaStat );
    cudaStat = cudaMemcpy(h_V, d_A, m * m * sizeof(double), cudaMemcpyDeviceToHost);
    assert( cudaSuccess == cudaStat );
    cudaStat = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat );


    /* Deallocate host and device workspace */
    if ( d_A ) { cudaFree( d_A ); }
    if ( d_V ) { cudaFree( d_V ); }
    if ( d_S ) { cudaFree( d_S ); }
    if ( d_info ) { cudaFree( d_info ); }
    if ( d_work ) { cudaFree( d_work ); }
    if ( h_work ) { free( h_work ); }
    if ( params ) { cusolverDnDestroyParams(params); }
    if( handle ) cusolverDnDestroy( handle ) ;

}


        

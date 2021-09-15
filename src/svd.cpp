 
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



void svd( 
        double * h_U,
        double * h_S,
        double * h_V,
        double * h_A,
        const int64_t m,
        const int64_t n,
        const int64_t lda,
        const int64_t k
    ) 
{
    cusolverStatus_t status ;    
    cudaError_t cudaStat ;

    cusolverDnHandle_t handle = nullptr;
    cusolverDnParams_t params_gesvdr = nullptr;

    const int64_t ldu = m;
    const int64_t ldv = n;
 
	const int64_t min_mn = std::min(m,n); 

    void *d_A    = nullptr ;
    void *d_U    = nullptr;
    void *d_S    = nullptr;
    void *d_V    = nullptr; 
	int  *d_info = nullptr;
    int   h_info = 0;

    double *d_work_gesvdr = nullptr;
    double *h_work_gesvdr = nullptr;
    size_t workspaceInBytesOnDevice_gesvdr = 0;
    size_t workspaceInBytesOnHost_gesvdr = 0;

	/* Compute left/right eigenvectors */
    signed char jobu = h_U==nullptr ? 'N' : 'S' ;
    signed char jobv = h_V==nullptr ? 'N' : 'S' ;

	/* Number of iterations */
    const int64_t iters = 10 ;
    const int64_t p    = 0 ; //std::min(5L, min_mn - rank);
    // printf( "Rank %ld, p %ld, mn %ld\n\n", k, p, min_mn ) ;
    assert( (k + p) <= min_mn );

    status = cusolverDnCreate(&handle);
    assert( CUSOLVER_STATUS_SUCCESS == status );

    status = cusolverDnCreateParams(&params_gesvdr);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat = cudaMalloc((void **) &d_A, sizeof(double)*lda*n);
    assert( cudaSuccess == cudaStat);

    if( h_U != nullptr ) {
        cudaStat = cudaMalloc((void **) &d_U, sizeof(double)*ldu*k);
        assert( cudaSuccess == cudaStat);
    }
    if( h_V != nullptr ) {
        cudaStat = cudaMalloc((void **) &d_V, sizeof(double)*ldv*k);
        assert( cudaSuccess == cudaStat);
    }
    cudaStat = cudaMalloc((void **) &d_S, sizeof(double)*n);
    assert( cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void **) &d_info, sizeof(int));

	/* copy input matrix from host to device memory */
    cudaStat = cudaMemcpy(d_A, h_A, sizeof(double)*lda*n, cudaMemcpyHostToDevice);
    assert( cudaSuccess == cudaStat);
    cudaStat = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat);

 
    status = cusolverDnXgesvdr_bufferSize(
        handle,
        params_gesvdr,
        jobu,
        jobv,
        m,
        n,
        k,
        p,
        iters,
        CUDA_R_64F,
        d_A,
        lda,
        CUDA_R_64F,
        d_S,
        CUDA_R_64F,
        d_U,
        ldu,
        CUDA_R_64F,
        d_V,
        ldv,
        CUDA_R_64F,
        &workspaceInBytesOnDevice_gesvdr,
        &workspaceInBytesOnHost_gesvdr );
    assert( status == CUSOLVER_STATUS_SUCCESS );

    h_work_gesvdr = (double*) malloc ( workspaceInBytesOnHost_gesvdr);
    assert(h_work_gesvdr != NULL );

    cudaStat = cudaMalloc((void **) &d_work_gesvdr, workspaceInBytesOnDevice_gesvdr);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat);
 
    status = cusolverDnXgesvdr (
        handle,
        params_gesvdr,
        jobu,
        jobv,
        m,
        n,
        k,
        p,
       	iters,
        CUDA_R_64F,
        d_A,
        lda,
        CUDA_R_64F,
        d_S,
        CUDA_R_64F,
        d_U,
        ldu,
        CUDA_R_64F,
        d_V,
        ldv,
        CUDA_R_64F,
        d_work_gesvdr,
        workspaceInBytesOnDevice_gesvdr,
        h_work_gesvdr,
        workspaceInBytesOnHost_gesvdr,
        d_info ) ;
    if( status != CUSOLVER_STATUS_SUCCESS ) {
        cudaStat = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        assert( cudaSuccess == cudaStat );
        assert( 0 == h_info ) ;
    }

    cudaStat = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat );

	/* copy info from device to host and sync */
    cudaStat = cudaMemcpy(h_S, d_S, k * sizeof(double), cudaMemcpyDeviceToHost);
    assert( cudaSuccess == cudaStat );
    if( d_U != nullptr ) {
        cudaStat = cudaMemcpy(h_U, d_U, ldu * k * sizeof(double), cudaMemcpyDeviceToHost);
        assert( cudaSuccess == cudaStat );
    }
    if( d_V != nullptr ) {
        cudaStat = cudaMemcpy(h_V, d_V, ldv * k * sizeof(double), cudaMemcpyDeviceToHost);
        assert( cudaSuccess == cudaStat );
    }
    cudaStat = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat );


    /* Deallocate host and device workspace */
    if ( d_A ) { cudaFree( d_A ); }
    if ( d_U ) { cudaFree( d_U ); }
    if ( d_V ) { cudaFree( d_V ); }
    if ( d_S ) { cudaFree( d_S ); }
    if ( d_info ) { cudaFree( d_info ); }
    if ( d_work_gesvdr ) { cudaFree( d_work_gesvdr ); }
    if ( h_work_gesvdr ) { free( h_work_gesvdr ); }
    if ( params_gesvdr ) { cusolverDnDestroyParams(params_gesvdr); }
    if( handle ) cusolverDnDestroy( handle ) ;

}


        

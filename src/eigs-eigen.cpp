 
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

#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 



typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> Matrix ;


void eigs( 
        double * h_S,   // output eigenvalues
        double * h_V,   // output eigen vectors
        double * h_A,   // input matrix ( symmetric = upper )
        const int64_t m // m x m matrix
    ) 
{
    Eigen::Map<Matrix> A = Eigen::Map<Matrix>(h_A,m,m) ;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    memcpy( h_S, es.eigenvalues().data(), sizeof(double)*m ) ;
    memcpy( h_V, es.eigenvectors().data(), sizeof(double)*m*m ) ;

    // std::cout << es.eigenvalues() << std::endl ;
    // std::cout << es.eigenvectors() << std::endl ;
}


        

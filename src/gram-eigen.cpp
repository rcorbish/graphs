 
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

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> Matrix ;

void gram( 
        double * h_A,
        double * h_C,
        const int64_t m,
        const int64_t n
    ) 
{
    Eigen::Map<Matrix> a = Eigen::Map<Matrix>(h_A,m,n) ;
    Eigen::Map<Matrix> b = Eigen::Map<Matrix>(h_A,m,n) ;
    // std::cout << "A=\n" << a << std::endl ;
    // std::cout << "B=\n" << b.transpose() << std::endl ;

    Eigen::MatrixXd c = a*b.transpose() ;
    // std::cout << "C=\n" << c << std::endl ;
    memcpy( h_C, c.data(), m*m*sizeof(double) ) ;

}


        

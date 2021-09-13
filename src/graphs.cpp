#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include <stdlib.h>

int svd( 
        double * h_U,
        double * h_S,
        double * h_V,
        double * h_A,
        const int64_t m,
        const int64_t n,
        const int64_t lda,
        const int64_t rank
    ) ;

template <class T>
std::string toString( const int64_t m, const int64_t n, const T* data ) {
    std::ostringstream rc ;
    for( int r=0 ; r<m ; r++ ) {
        for( int c=0 ; c<n ; c++ ) {
            rc << std::setw(10) << std::fixed << data[m*c + r] << " " ;
        }
        rc << '\n' ;
    }
    return rc.str() ;
}


int main( int argc, char *argv[]) 
{
    constexpr int64_t m = 6 ;
    constexpr int64_t n = 5 ;
    constexpr int64_t k = 5 ;

    constexpr int64_t lda = m ;
    /* Reference matrix in COL order */
	double A[m * n] = {
		0.76420743, 0.61411544, 0.81724151, 0.42040879, 0.03446089, 6,
		0.03697287, 0.85962444, 0.67584086, 0.45594666, 0.02074835, 7, 
		0.42018265, 0.39204509, 0.12657948, 0.90250559, 0.23076218, 8, 
		0.50339844, 0.92974961, 0.21213988, 0.63962457, 0.58124562, 9,
		0.58325673, 0.11589871, 0.39831112, 0.21492685, 0.00540355, 0		
	};

    std::cout << toString<double>( m, n, A ) << std::endl ;

    int64_t min_mn = std::min(m,n) ; 
    double S[ k ] ;
    double U[ m*k ] ;
    double V[ k*n ] ;
    svd( U, S, V, A, m, n, lda, k ) ;

    std::cout << toString<double>( m, k, U ) << std::endl ;

    for( int i=0 ; i<k ; i++ ) {
        std::cout << std::setw(10) << std::fixed << S[i] << " " ;
    }
    std::cout << std::endl << std::endl ;

    std::cout << toString<double>( k, n, V ) << std::endl ;

    return 0 ;
}

#pragma once

// #include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include <stdlib.h>    // pointsOld.clear() ;


#include "graphFactory.hpp"


void gemm( 
        double * h_A,
        double * h_B,
        double * h_C,
        const int64_t m,
        const int64_t n,
        const int64_t k
    ) ;
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

std::vector<double> incidence( Graph &adjacency ) ;
std::vector<std::pair<double,double>> getCoords( Graph &adjacency, int a, int b ) ;
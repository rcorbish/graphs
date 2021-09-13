#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include <stdlib.h>


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


std::vector<double> incidence( Graph &adjacency ) {

    int num_edges = 0 ;
    for( auto node : adjacency ) {
        num_edges += node.size() ;
    }

    const size_t num_nodes = adjacency.size() ;

    std::vector<double> rc ;
    rc.resize( num_nodes*num_edges) ;
    std::fill(rc.begin(), rc.end(), 0.0 );

    int edge = 0 ;
    for( int n=0 ; n<adjacency.size() ; n++ ) {        
        for( auto destination : adjacency[n] ) {
            // std::cout << "Edges # " << edge << " " << n << " -> " << destination << std::endl ;
            rc[edge*num_nodes+n] = -1.0 ;
            rc[edge*num_nodes+destination] = 1.0 ;
            edge++ ;
        }
    } 

    return rc ;
}



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



std::vector<std::pair<double,double>> getCoords( Graph &adjacency, int a, int b ) {
    int num_nodes = adjacency.size() ;
    int num_edges = 0 ;
    for( auto node : adjacency ) {
        num_edges += node.size() ;
    }

    std::vector<double> inc = incidence( adjacency ) ;
    std::vector<double> lap ;
    lap.resize( num_nodes * num_nodes ) ;
    gemm( &inc[0], &inc[0], &lap[0], num_nodes, num_nodes, num_edges ) ;

    int m = num_nodes ;
    int n = num_nodes ;
    int k = num_nodes ;

    double S[ k ] ;
    double U[ m*k ] ;
    double V[ k*n ] ;
    svd( U, S, V, &lap[0], m, n, m, k ) ;


    // int numZeroSingularValue = 0 ;
    // for( int i=0 ; i<k ; i++ ) {
    //     if( S[i] < 1e-8 ) {
    //         numZeroSingularValue ++ ;
    //     }
    // }

    // int xIx = num_nodes - numZeroSingularValue - 1;
    // int yIx = xIx - 1 ;
    // while( std::abs( S[yIx-1]-S[xIx] ) < 1e-4 ) yIx-- ;
    
    // std::cout << xIx << " " << yIx << std::endl ;
    // std::cout << toString<double>( 1, m, S ) << std::endl ;
    // std::cout << toString<double>( m, k, U ) << std::endl ;

    std::vector<std::pair<double,double>> points ;
    // double *x = U + ((num_nodes-numZeroSingularValue)*num_nodes) ;
    // double *y = U + ((num_nodes-numZeroSingularValue-numZeroSingularValue)*num_nodes) ;
    double *x = U + a*num_nodes ;
    double *y = U + b*num_nodes ;

    // std::cout << toString<double>( 1, m, x ) << std::endl ;
    // std::cout << toString<double>( 1, m, y ) << std::endl ;

    for( int i=0 ; i<num_nodes ; i++ ) {
        points.emplace_back( x[i], y[i] ) ;
    }
    return points ;
}

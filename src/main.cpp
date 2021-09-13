#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include <stdlib.h>

#include <vector>


#include "graphFactory.hpp"
#include "functions.hpp"



int main( int argc, char *argv[]) 
{
    std::vector<std::vector<int>> adjacency = stars() ;

    int num_nodes = adjacency.size() ;
    int num_edges = 0 ;
    for( auto node : adjacency ) {
        num_edges += node.size() ;
    }

    std::vector<double> inc = incidence( adjacency ) ;
    std::cout << toString<double>( num_nodes, num_edges, &inc[0] ) << std::endl ;

    std::vector<double> lap ;
    lap.resize( num_nodes * num_nodes ) ;

    gemm( &inc[0], &inc[0], &lap[0], num_nodes, num_nodes, num_edges ) ;
    std::cout << toString<double>( num_nodes, num_nodes, &lap[0] ) << std::endl ;

    int m = num_nodes ;
    int n = num_nodes ;
    int k = num_nodes ;

    double S[ k ] ;
    double U[ m*k ] ;
    double V[ k*n ] ;
    svd( U, S, V, &lap[0], m, n, m, k ) ;

    std::cout << toString<double>( 1, n, S ) << std::endl ;
    std::cout << toString<double>( m, k, U ) << std::endl ;
    return 0 ;
}



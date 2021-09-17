
#include <stdlib.h>
#include <time.h>

#include "graphFactory.hpp"


Graph * GraphFactory::petersen() {
    constexpr int N = 10 ;

    Graph *graph = new Graph( "petersen", N ) ;

    graph->addEdge(0, 1 ) ;
    graph->addEdge(0, 5 ) ;
    graph->addEdge(0, 4 ) ;

    graph->addEdge(1, 2 ) ;
    graph->addEdge(1, 6 ) ;
    graph->addEdge(1, 0 ) ;

    graph->addEdge(2, 3 ) ;
    graph->addEdge(2, 7 ) ;
    graph->addEdge(2, 1 ) ;

    graph->addEdge(3, 4 ) ;
    graph->addEdge(3, 8 ) ;
    graph->addEdge(3, 2 ) ;

    graph->addEdge(4, 0 ) ;
    graph->addEdge(4, 9 ) ;
    graph->addEdge(4, 3 ) ;

    graph->addEdge(5, 0 ) ;
    graph->addEdge(5, 7 ) ;
    graph->addEdge(5, 8 ) ;

    graph->addEdge(6, 1 ) ;
    graph->addEdge(6, 8 ) ;
    graph->addEdge(6, 9 ) ;

    graph->addEdge(7, 2 ) ;
    graph->addEdge(7, 9 ) ;
    graph->addEdge(7, 5 ) ;

    graph->addEdge(8, 3 ) ;
    graph->addEdge(8, 5 ) ;
    graph->addEdge(8, 6 ) ;

    graph->addEdge(9, 4 ) ;
    graph->addEdge(9, 6 ) ;
    graph->addEdge(9, 7 ) ;

    return graph ;
}



Graph * GraphFactory::barbell() {
    constexpr int N = 6 ;

    Graph *graph = new Graph( "barbell", N ) ;

    graph->addEdge(0, 1 ) ;
    graph->addEdge(1, 2 ) ;
    graph->addEdge(2, 0 ) ;

    graph->addEdge(3, 4 ) ;
    graph->addEdge(4, 5 ) ;
    graph->addEdge(5, 3 ) ;

    graph->addEdge(0, 3 ) ;

    return graph ;
}


Graph * GraphFactory::sample() {

    constexpr int N = 5 ;

    Graph *graph = new Graph( "sample", N ) ;

    graph->addEdge(0, 1 ) ;
    graph->addEdge(0, 2 ) ;
    graph->addEdge(0, 4 ) ;

    graph->addEdge(1, 2 ) ;
    graph->addEdge(1, 0 ) ;

    graph->addEdge(2, 0 ) ;
    graph->addEdge(2, 1 ) ;
    graph->addEdge(2, 3 ) ;

    graph->addEdge(3, 2 ) ;

    return graph ;
}


Graph * GraphFactory::grid() {

    constexpr int N = 3 ;

    Graph *graph = new Graph( "grid", N*N ) ;

    for( int r=0 ; r<N ; r++ ) {
        for( int c=0 ; c<N ; c++ ) {
            int n = r*N + c ;
            int m ;

            if( r>0 ) {
                m = (r-1)*N + c ;
                graph->addEdge(n,m) ;
            }

            if( r<(N-1) ) {
                m = (r+1)*N + c ;
                graph->addEdge(n,m) ;
            }

            if( c>0 ) {
                m = r*N + c-1 ;
                graph->addEdge(n,m) ;
            }

            if( c<(N-1) ) {
                m = r*N + c+1 ;
                graph->addEdge(n,m) ;
            }
        }
    }
    return graph ;
}



Graph * GraphFactory::star() {

    constexpr int N = 6 ;

    Graph *graph = new Graph( "star", N ) ;

    for( int n=1 ; n<N ; n++ ) {
        graph->addEdge(0,n) ;
    }
    return graph ;
}

Graph * GraphFactory::stars() {

    constexpr int N = 8 ;

    Graph *graph = new Graph( "stars", N ) ;

    graph->addEdge(0,1) ;
    graph->addEdge(0,2) ;
    graph->addEdge(0,3) ;

    graph->addEdge(4,5) ;
    graph->addEdge(4,6) ;
    graph->addEdge(4,7) ;

    return graph ;
}

Graph * GraphFactory::tree() {

    constexpr int Depth = 4 ;
    constexpr int N = (1 << Depth) - 1 ;

    Graph *graph = new Graph( "tree", N ) ;

    for( int i=1 ; i<Depth ; i++ ) {
        int width = 1<<i ;
        int firstLeaf = width - 1 ;
        for( int i=0 ; i<width ; i++ ) {
            int child = firstLeaf+ i ;
            int parent = (child-1) >> 1 ;
            graph->addEdge(parent, child ) ;
        }
    }

    return graph ;
}


Graph * GraphFactory::ring() {

    constexpr int N = 10 ;

    Graph *graph = new Graph( "ring", N ) ;

    for( auto i=0 ; i<(N-1) ; i++ ) {
        graph->addEdge(i,i+1) ;
    }
    graph->addEdge(N-1,0) ;

    return graph ;
}



Graph * GraphFactory::bipartite() {

    constexpr int N = 10 ;

    Graph *graph = new Graph( "bipartite", N ) ;

    for( auto i=0 ; i<N ; i+=2 ) {
        if( i>1 ) {
            graph->addEdge(i,i-1) ;
        }
        graph->addEdge(i+1,i) ;
        if( (i+3)<N ) {
            graph->addEdge(i,i+3) ;
        }
    }
    // graph->print( std::cout ) ;
    return graph ;
}
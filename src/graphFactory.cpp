
#include <stdlib.h>
#include <time.h>

#include "graphFactory.hpp"


Graph petersen() {
    constexpr int N = 10 ;

    Graph adjacency ;
    for( int i=0 ; i<N ; i++ ) {
        adjacency.emplace_back( ) ;
    }

    adjacency[0].push_back( 1 ) ;
    adjacency[0].push_back( 5 ) ;
    adjacency[0].push_back( 4 ) ;

    adjacency[1].push_back( 2 ) ;
    adjacency[1].push_back( 6 ) ;
    adjacency[1].push_back( 0 ) ;

    adjacency[2].push_back( 3 ) ;
    adjacency[2].push_back( 7 ) ;
    adjacency[2].push_back( 1 ) ;

    adjacency[3].push_back( 4 ) ;
    adjacency[3].push_back( 8 ) ;
    adjacency[3].push_back( 2 ) ;

    adjacency[4].push_back( 0 ) ;
    adjacency[4].push_back( 9 ) ;
    adjacency[4].push_back( 3 ) ;

    adjacency[5].push_back( 0 ) ;
    adjacency[5].push_back( 7 ) ;
    adjacency[5].push_back( 8 ) ;

    adjacency[6].push_back( 1 ) ;
    adjacency[6].push_back( 8 ) ;
    adjacency[6].push_back( 9 ) ;

    adjacency[7].push_back( 2 ) ;
    adjacency[7].push_back( 9 ) ;
    adjacency[7].push_back( 5 ) ;

    adjacency[8].push_back( 3 ) ;
    adjacency[8].push_back( 5 ) ;
    adjacency[8].push_back( 6 ) ;

    adjacency[9].push_back( 4 ) ;
    adjacency[9].push_back( 6 ) ;
    adjacency[9].push_back( 7 ) ;

    return adjacency ;
}



Graph barbell() {
    constexpr int N = 6 ;

    Graph adjacency ;
    for( int i=0 ; i<N ; i++ ) {
        adjacency.emplace_back( ) ;
    }

    adjacency[0].push_back( 1 ) ;
    adjacency[1].push_back( 2 ) ;
    adjacency[2].push_back( 0 ) ;

    adjacency[3].push_back( 4 ) ;
    adjacency[4].push_back( 5 ) ;
    adjacency[5].push_back( 3 ) ;

    adjacency[0].push_back( 3 ) ;

    return adjacency ;
}


Graph sample() {

    constexpr int N = 5 ;

    Graph adjacency ;
    for( int i=0 ; i<N ; i++ ) {
        adjacency.emplace_back( ) ;
    }
    adjacency[0].push_back( 1 ) ;
    adjacency[0].push_back( 2 ) ;
    adjacency[0].push_back( 4 ) ;

    adjacency[1].push_back( 2 ) ;
    adjacency[1].push_back( 0 ) ;

    adjacency[2].push_back( 0 ) ;
    adjacency[2].push_back( 1 ) ;
    adjacency[2].push_back( 3 ) ;

    adjacency[3].push_back( 2 ) ;

    return adjacency ;
}


Graph grid() {

    constexpr int N = 3 ;

    Graph adjacency ;
    for( int i=0 ; i<N*N ; i++ ) {
        adjacency.emplace_back( ) ;
    }
    for( int r=0 ; r<N ; r++ ) {
        for( int c=0 ; c<N ; c++ ) {
            int n = r*N + c ;
            int m ;

            if( r>0 ) {
                m = (r-1)*N + c ;
                adjacency[n].push_back(m) ;
            }

            if( r<(N-1) ) {
                m = (r+1)*N + c ;
                adjacency[n].push_back(m) ;
            }

            if( c>0 ) {
                m = r*N + c-1 ;
                adjacency[n].push_back(m) ;
            }

            if( c<(N-1) ) {
                m = r*N + c+1 ;
                adjacency[n].push_back(m) ;
            }
        }
    }
    return adjacency ;
}



Graph star() {

    constexpr int N = 6 ;

    Graph adjacency ;
    for( int i=0 ; i<N ; i++ ) {
        adjacency.emplace_back( ) ;
    }
    for( int n=1 ; n<N ; n++ ) {
        adjacency[0].push_back(n) ;
    }
    return adjacency ;
}

Graph stars() {

    constexpr int N = 8 ;

    Graph adjacency ;
    for( int i=0 ; i<N ; i++ ) {
        adjacency.emplace_back( ) ;
    }
    adjacency[0].push_back(1) ;
    adjacency[0].push_back(2) ;
    adjacency[0].push_back(3) ;

    adjacency[4].push_back(5) ;
    adjacency[4].push_back(6) ;
    adjacency[4].push_back(7) ;

    return adjacency ;
}

Graph tree() {


    constexpr int Depth = 4 ;
    constexpr int N = (1 << Depth) - 1 ;

    Graph adjacency ;
    for( int i=0 ; i<N ; i++ ) {
        adjacency.emplace_back( ) ;
    }

    for( int i=1 ; i<Depth ; i++ ) {
        int width = 1<<i ;
        int firstLeaf = width - 1 ;
        for( int i=0 ; i<width ; i++ ) {
            int child = firstLeaf+ i ;
            int parent = (child-1) >> 1 ;
            adjacency[parent].push_back( child ) ;
        }
    }

    return adjacency ;
}


Graph ring() {

    constexpr int N = 10 ;

    Graph adjacency ;
    for( int i=0 ; i<N ; i++ ) {
        adjacency.emplace_back( ) ;
    }
    for( auto i=0 ; i<(N-1) ; i++ ) {
        adjacency[i].push_back(i+1) ;
    }
    adjacency[N-1].push_back(0) ;

    return adjacency ;
}

Graph bipartite() {

    constexpr int N = 10 ;

    Graph adjacency ;
    for( int i=0 ; i<N ; i++ ) {
        adjacency.emplace_back( ) ;
    }
    for( auto i=0 ; i<N ; i+=2 ) {
        if( i>1 ) {
            adjacency[i].push_back(i-1) ;
        }
        adjacency[i].push_back(i+1) ;
        adjacency[i+1].push_back(i) ;
        if( (i+3)<N ) {
            adjacency[i].push_back(i+3) ;
            adjacency[i+3].push_back(i+2) ;
        }
    }

    return adjacency ;
}
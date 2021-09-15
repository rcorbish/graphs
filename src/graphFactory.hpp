
#pragma once 

#include "graph.hpp"


class GraphFactory {
        static Graph *petersen() ;
        static Graph *barbell() ;
        static Graph *sample() ;
        static Graph *grid() ;
        static Graph *star() ;
        static Graph *stars() ;
        static Graph *ring() ;
        static Graph *bipartite() ;
        static Graph *tree() ;

    public:
        static constexpr int NumGraphs = 9 ;

        static Graph *get( int ix ) {
            Graph *rc = nullptr ;
            switch( ix ) {
                case 0 : rc = petersen() ; break ;
                case 1 : rc = tree() ; break ;
                case 2 : rc = sample() ; break ;
                case 3 : rc = grid() ; break ;
                case 4 : rc = star() ; break ;
                case 5 : rc = stars() ; break ;
                case 6 : rc = ring() ; break ;
                case 7 : rc = bipartite() ; break ;
                case 8 : rc = barbell() ; break ;
            }
            if( rc == nullptr ) rc = sample() ;

            rc -> calculate() ;

            return rc ;
        }
} ;

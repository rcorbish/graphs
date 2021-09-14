
#pragma once 

#include <vector>

typedef std::vector<std::vector<int>> Graph ;

Graph petersen() ;
Graph barbell() ;
Graph sample() ;
Graph grid() ;
Graph star() ;
Graph stars() ;
Graph ring() ;
Graph bipartite() ;
Graph tree() ;

class GraphFactory {
    public:
        static constexpr int NumGraphs = 9 ;

        static Graph get( int ix ) {

            switch( ix ) {
                case 0 : return bipartite() ;
                case 1 : return tree() ;
                case 2 : return sample() ;
                case 3 : return grid() ;
                case 4 : return star() ;
                case 5 : return stars() ;
                case 6 : return ring() ;
                case 7 : return petersen() ;
                case 8 : return barbell() ;
            }

            return sample() ;
        }
} ;

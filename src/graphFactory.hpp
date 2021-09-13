
#pragma once 

#include <vector>

typedef std::vector<std::vector<int>> Graph ;

Graph petersen() ;
Graph barbell() ;
Graph sample() ;
Graph grid() ;
Graph star() ;
Graph stars() ;
Graph huge() ;

class GraphFactory {
    public:
        static constexpr int NumGraphs = 7 ;

        static Graph get( int ix ) {

            switch( ix ) {
                case 0 : return petersen() ;
                case 1 : return barbell() ;
                case 2 : return sample() ;
                case 3 : return grid() ;
                case 4 : return star() ;
                case 5 : return stars() ;
                case 6 : return huge() ;
            }

            return sample() ;
        }
} ;

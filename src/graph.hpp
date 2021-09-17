
#include <vector>
#include <string>
#include <iostream>

typedef std::vector<std::vector<int>> AdjacencyMatrix ;
typedef std::pair<double,double> Point ;

class LineSegment {
    public :
        const Point a ;
        const Point b ;

        LineSegment( const Point _a, const Point _b ) : a(_a), b(_b) {}

        bool crosses( LineSegment o ) {

            double l = ( a.first - b.first) * ( o.a.second - o.b.second ) - 
                ( a.second - b.second ) * ( o.a.first - o.b.first ) ;

            double tu = ( a.first - o.a.first) * ( o.a.second - o.b.second ) - 
                ( a.second - o.a.second ) * ( o.a.first - o.b.first ) ;
                       
            double t = tu / l ;

            double uu = ( b.first - a.first) * ( a.second - o.a.second ) - 
                ( b.second - a.second ) * ( a.first - o.a.first ) ;

            double u = uu / l ;

            return ( 0 < t && t < 1.0 ) && ( 0 < u && u < 1.0 ) ;
        }
} ;

class Graph {

        AdjacencyMatrix adjacencyMatrix ;
        const std::string _name ;

        std::vector<double> incidenceMatrix ;
        std::vector<double> singularValues ;
        std::vector<double> singularVectors ;

    protected:
        void incidence() ;

    public: 
        Graph( std::string _name, int numNodes ) ;
        virtual ~Graph() ;

        void addEdge( int from, int to ) ;
        std::vector<int> &edges( int from ) ;

        void calculate() ;

        size_t numNodes() ;
        size_t numEdges() ;
        std::string name() ;

        std::vector<Point> getCoords( int a, int b ) ;
        std::vector<LineSegment> getLines( std::vector<Point> points ) ;

        void print( std::ostream &os ) ;
} ;

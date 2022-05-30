#pragma once 

#include <tuple>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

typedef std::vector<std::vector<int>> AdjacencyMatrix ;

class Point {
    const double _x,_y,_z ;
  public:
    Point( double __x, double __y, double __z ) : _x(__x), _y(__y), _z(__z) {}
    Point( const Point &o ) : _x(o._x), _y(o._y), _z(o._z) {}

    double x() const { return _x ; }
    double y() const { return _y ; }
    double z() const { return _z ; }

    double distanceTo( const Point &o ) const {
        double dx = o.x() - x() ;
        double dy = o.y() - y() ;
        double dz = o.z() - z() ;
        return sqrt( dx*dx + dy*dy + dz*dz ) ;
    }
} ;
typedef std::vector<Point> Points ;


class LineSegment {
    public :
        const Point a ;
        const Point b ;

        LineSegment( const Point _a, const Point _b ) : a(_a), b(_b) {}
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

        std::vector<Point> getCoords( int a, int b, int c ) ;
        std::vector<LineSegment> getLines( std::vector<Point> points ) ;
        std::vector<double> getSingularValues() { return singularValues  ; }

        void print( std::ostream &os ) ;
} ;

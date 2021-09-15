
#include <vector>
#include <string>

typedef std::vector<std::vector<int>> AdjacencyMatrix ;

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

        std::vector<std::pair<double,double>> getCoords( int a, int b ) ;
} ;

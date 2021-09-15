
#include "graph.hpp"



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


Graph::Graph( std::string name, int numNodes ) : _name( name ) {
            adjacencyMatrix.reserve( numNodes ) ;
            singularValues.reserve( numNodes ) ;    
            singularVectors.reserve( numNodes*numNodes ) ;
            for( int i=0 ; i<numNodes ; i++ ) {
                adjacencyMatrix.emplace_back() ;
            }
}

Graph::~Graph() {

}

void Graph::addEdge( int from, int to ) {
    adjacencyMatrix[from].push_back( to ) ;
}

std::vector<int> &Graph::edges( int from ) {
    return adjacencyMatrix[ from ] ;
}

void Graph::calculate() {
    incidence() ;
    std::vector<double> lap ;
    lap.resize( numNodes() * numNodes() ) ;
    gemm( &incidenceMatrix[0], &incidenceMatrix[0], &lap[0], numNodes(), numNodes(), numEdges() ) ;

    int m = numNodes() ;
    int n = numNodes() ;
    int k = numNodes() ;

    double *U = &singularVectors[0] ;
    double V[ k*n ] ;
    svd( U, &singularValues[0], V, &lap[0], m, n, m, k ) ;
}



void Graph::incidence() {

    int num_edges = 0 ;
    for( auto node : adjacencyMatrix ) {
        num_edges += node.size() ;
    }

    const size_t num_nodes = adjacencyMatrix.size() ;

    incidenceMatrix.clear() ;
    incidenceMatrix.resize( num_nodes*num_edges) ;
    std::fill(incidenceMatrix.begin(), incidenceMatrix.end(), 0.0 );

    int edge = 0 ;
    for( int n=0 ; n<numNodes() ; n++ ) {        
        for( auto destination : adjacencyMatrix[n] ) {
            incidenceMatrix[edge*num_nodes+n] = -1.0 ;
            incidenceMatrix[edge*num_nodes+destination] = 1.0 ;
            edge++ ;
        }
    } 
}

size_t Graph::numNodes() {
    return adjacencyMatrix.size() ;
}

size_t Graph::numEdges() {
    size_t num_edges = 0 ;
    for( auto node : adjacencyMatrix ) {
        num_edges += node.size() ;
    }
    return num_edges ;
}

std::string Graph::name() { return this->_name ; } 

std::vector<std::pair<double,double>> Graph::getCoords( int a, int b ) {

    double *U = &singularVectors[0] ;

    std::vector<std::pair<double,double>> points ;
    double *x = U + a*numNodes() ;
    double *y = U + b*numNodes() ;

    for( int i=0 ; i<numNodes() ; i++ ) {
        points.emplace_back( x[i], y[i] ) ;
    }

    return points ;
}

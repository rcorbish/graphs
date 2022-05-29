
#include "graph.hpp"
#include "linalg.hpp"

#include <set>

Graph::Graph(std::string name, int numNodes) : _name(name) {
    adjacencyMatrix.reserve(numNodes);
    singularValues.reserve(numNodes);
    singularVectors.reserve(numNodes * numNodes);
    for (int i = 0; i < numNodes; i++) {
        adjacencyMatrix.emplace_back();
    }
}

Graph::~Graph() {
}

void Graph::addEdge(int from, int to) {
    adjacencyMatrix[from].push_back(to);
}

std::vector<int> &Graph::edges(int from) {
    return adjacencyMatrix[from];
}

void Graph::calculate() {
    incidence();
    std::vector<double> lap;
    lap.resize(numNodes() * numNodes());
    gram(&incidenceMatrix[0], &lap[0], numNodes(), numEdges());

    int m = numNodes();
    int n = numNodes();
    int k = numNodes();

    singularValues.resize(m);
    singularVectors.resize(m * m);

    eigs(&singularValues[0], &singularVectors[0], &lap[0], m);
}

void Graph::incidence() {

    int num_edges = 0;
    for (auto node : adjacencyMatrix) {
        num_edges += node.size();
    }

    const size_t num_nodes = adjacencyMatrix.size();

    incidenceMatrix.clear();
    incidenceMatrix.resize(num_nodes * num_edges);
    std::fill(incidenceMatrix.begin(), incidenceMatrix.end(), 0.0);

    int edge = 0;
    for (int n = 0; n < numNodes(); n++) {
        for (auto destination : adjacencyMatrix[n]) {
            incidenceMatrix[edge * num_nodes + n] = -1.0;
            incidenceMatrix[edge * num_nodes + destination] = 1.0;
            edge++;
        }
    }
}

size_t Graph::numNodes() {
    return adjacencyMatrix.size();
}

size_t Graph::numEdges() {
    size_t num_edges = 0;
    for (auto node : adjacencyMatrix) {
        num_edges += node.size();
    }
    return num_edges;
}

std::string Graph::name() { return this->_name; }

std::vector<Point> Graph::getCoords(int a, int b) {

    double *U = &singularVectors[0];

    std::vector<Point> points;
    double *x = U + a * numNodes();
    double *y = U + b * numNodes();

    for (int i = 0; i < numNodes(); i++) {
        points.emplace_back(x[i], y[i]);
    }

    return points;
}

std::vector<LineSegment> Graph::getLines(std::vector<Point> points) {

    std::vector<LineSegment> lines;
    lines.reserve(numEdges());

    std::set<std::pair<int, int>> existingLines;

    int n = 0;
    for (auto point : points) {
        for (auto e : edges(n)) {
            int a, b;
            if (n >= e) {
                a = n;
                b = e;
            } else {
                a = e;
                b = n;
            }
            std::pair<int, int> thisLine(a, b);
            if (existingLines.find(thisLine) == existingLines.end()) {
                lines.emplace_back(point, points[e]);
                existingLines.emplace(a, b);
            }
        }
        n++;
    }
    return lines;
}

void Graph::print(std::ostream &os) {
    int n = 0;
    for (auto node : adjacencyMatrix) {
        os << n << " ";
        n++;
        for (auto tgt : node) {
            os << tgt << " ";
        }
        os << std::endl;
    }
}

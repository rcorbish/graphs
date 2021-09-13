
#include <iostream>
#include <array>
#include <vector>

#include <pthread.h>

#include "graphFactory.hpp"
#include "functions.hpp"

#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>
#include <glm/glm.hpp>  
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define IMG_LIMIT 100.0


const float time_step_size = 0.01f ;
const int world_size = 7000 ;
glm::vec3 hsv_to_rgb(float h, float s, float v ) ;

// ----------------------------------------------------------
// Function Prototypes
// ----------------------------------------------------------
void display();

int main( int argc, char **argv ) {

  setlocale( LC_ALL, "" ) ;

  glutInit(&argc,argv);
 
  //  Request double buffered true color window with Z-buffer
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB );
  // glutInitDisplayMode(GLUT_RGB );
 
  // Create window
  glutCreateWindow("Graphs");
  glutReshapeWindow( 500, 500 );
 
  // Enable Z-buffer depth test
  glEnable(GL_DEPTH_TEST);

  // Callback functions
  glutDisplayFunc(display);
  
  // Pass control to GLUT for events
  glutMainLoop(); 
}


int a = 10000 ; 
int b = 10000 ;
int graph = 10000 ;
float t = 10000.f ;  // force reload of stuff
float dt = 1 ;
int num_edges ;
std::vector<std::pair<double,double>> pointsNew ;
std::vector<std::pair<double,double>> pointsOld ;
constexpr float MaxClock = 60.f ;
Graph adjacency ;

// ----------------------------------------------------------
// display() Callback function
// ----------------------------------------------------------
void display(){
  t += dt ;

  if( t > MaxClock ) {
    b++ ;
    if( (b+1) >= adjacency.size() ) { 
      a++ ; 
      b = a+1 ;
      if( (a+2) >= adjacency.size() ) {
        a = 0 ;
        b = 1 ;
        graph++ ;
        if( graph>=GraphFactory::NumGraphs ) graph = 0 ;
        adjacency = GraphFactory::get(graph) ;
        pointsNew.clear() ;
        num_edges = 0 ;
        for( auto &node : adjacency ) {
          num_edges += node.size() ;
        }
      }
    } 
    // pointsOld.clear() ;
    // pointsOld = std::move( pointsNew ) ;
    pointsOld = pointsNew ;
    pointsNew = getCoords( adjacency, a, b ) ;
    // std::cout << graph << " " << a << "," << b << std::endl ;
    t = 0 ;
  }

  //  Clear screen and Z-buffer
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
 
  // Reset transformations
  glLoadIdentity(); 

  glColor3f( .2, 1, .8 );
  glRasterPos2f( -.9, 0.9 );
  char s[64] ;
  sprintf( s, "Graph: %'d  %'d - %'d  Nodes: %'ld  Edges: %'d", graph, a, b, adjacency.size(), num_edges ) ;
  int len = (int)strlen(s);
  for( int i = 0; i < len; i++ ) {
    glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, s[i] );
  }

  // L R T B
  glOrtho( -IMG_LIMIT,  IMG_LIMIT,  IMG_LIMIT,  -IMG_LIMIT,  1,  -1) ;
  float rate = (4.f * t) / MaxClock ;
  if( rate > 1.f ) rate = 1.f ;

  std::vector<std::pair<double,double>> points ;
  int n = 0 ;
  for( int n=0 ; n<pointsNew.size() ; n++ ) {
    float xnew = pointsNew[n].first ;
    float ynew = pointsNew[n].second ;
    float xold = pointsOld.empty() ? 0 : pointsOld[n].first ;
    float yold = pointsOld.empty() ? 0 : pointsOld[n].second ;

    float x = ( xold + ( rate * ( xnew - xold ) ) ) * IMG_LIMIT ;
    float y = ( yold + ( rate * ( ynew - yold ) ) ) * IMG_LIMIT ;
    points.emplace_back(x,y) ;
  }

  glm::vec3 rgbl = hsv_to_rgb( (graph/(float)GraphFactory::NumGraphs), 1, 1 ) ;
  glColor3f( rgbl.r, rgbl.g, rgbl.b );

  n=0 ;
  for( auto point : points ) {

    float x = point.first ;    
    float y = point.second ;    

    for( auto e : adjacency[n] ) {
      float x2 = points[e].first ;
      float y2 = points[e].second ;

      glBegin(GL_LINES);
        glVertex2f( x, y ) ;
        glVertex2f( x2, y2 ) ;
      glEnd();
    }
    n++ ;
  }

  float hue = 0.f ;
  float dhue = 1.f / points.size() ;
  for( auto point : points ) {

    float x = point.first ;    
    float y = point.second ;    

    hue += dhue ;
    glm::vec3 rgb = hsv_to_rgb( hue, 1, 1 ) ;
    glColor3f( rgb.r, rgb.g, rgb.b );
    glBegin(GL_TRIANGLE_STRIP);
  		glVertex2f( x-2, y+2 ) ;
  		glVertex2f( x-2, y-2 ) ;
  		glVertex2f( x+2, y+2 ) ;
  		glVertex2f( x+2, y-2 ) ;
    glEnd();
  }

  glutSwapBuffers();
  glutPostRedisplay();
}

glm::vec3 hsv_to_rgb(float h, float s, float v)
{
	float c = v * s;
	h = glm::mod((h * 6.f), 6.f);
	float x = c * (1.0 - abs(glm::mod(h, 2.f) - 1.0));
	glm::vec3 color;

	if (0.0 <= h && h < 1.0) {
		color = glm::vec3(c, x, 0.0);
	} else if (1.0 <= h && h < 2.0) {
		color = glm::vec3(x, c, 0.0);
	} else if (2.0 <= h && h < 3.0) {
		color = glm::vec3(0.0, c, x);
	} else if (3.0 <= h && h < 4.0) {
		color = glm::vec3(0.0, x, c);
	} else if (4.0 <= h && h < 5.0) {
		color = glm::vec3(x, 0.0, c);
	} else if (5.0 <= h && h < 6.0) {
		color = glm::vec3(c, 0.0, x);
	} else {
		color = glm::vec3(0.0, 0.0, 0.0);
	}

	color.r += v - c;
	color.g += v - c;
	color.b += v - c;

	return color;
}



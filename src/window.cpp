
#include <iostream>
#include <array>
#include <vector>

#include <pthread.h>

#include "graphFactory.hpp"

#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>
#include <glm/glm.hpp>  
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define IMG_LIMIT 100.0


typedef std::vector<Point> Points ;
const float time_step_size = 0.01f ;
const int world_size = 7000 ;
glm::vec3 hsv_to_rgb(float h, float s, float v ) ;
void drawNodes( Points points ) ;
Points relocateNodes( double rate ) ;
void labelNodes( Points points ) ;

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
Points pointsNew ;
Points pointsOld ;
constexpr float MaxClock = 150.f ;
Graph * theGraph = GraphFactory::get( 0 ) ;

// ----------------------------------------------------------
// display() Callback function
// ----------------------------------------------------------
void display(){
  t += dt ;
  char s[256] ;

  if( t > MaxClock ) {
    b++ ;
    if( (b) >= theGraph->numNodes() ) { 
      a++ ; 
      b = a+1 ;
      if( (a+1) >= theGraph->numNodes() ) {
        a = 1 ;
        b = 2 ;
        graph++ ;
        delete theGraph ;
        if( graph>=GraphFactory::NumGraphs ) graph = 0 ;
        theGraph = GraphFactory::get(graph) ;
        pointsNew.clear() ;
        num_edges = theGraph->numEdges() ;
      }
    } 
    pointsOld = pointsNew ;
    pointsNew = theGraph->getCoords( a, b ) ;
    t = 0 ;

    std::vector<LineSegment> lines = theGraph->getLines( pointsNew ) ;

    int numCrossings = 0 ;
    for( size_t i=0 ; i<lines.size() ; i++ ) {
      LineSegment a = lines[i] ;
      for( size_t j=(i+1) ; j<lines.size() ; j++ ) {
        LineSegment b = lines[j] ;
        numCrossings += a.crosses(b) ? 1 : 0 ;
      }
    }

    float separation = 0.0 ;
    for( auto n=0 ; n<pointsNew.size() ; n++ ) {
      for( auto e=n+1 ; e<pointsNew.size() ; e++ ) {
        separation += 1.f / ( distance( pointsNew[n], pointsNew[e] ) + .1 ) ;
      }
    }
    separation /= pointsNew.size() ;

    std::cout << a << " - " << b << " Num crossings " << numCrossings << " separation "<< separation << std::endl ;

  }

  //  Clear screen and Z-buffer
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
 
  // Reset transformations
  glLoadIdentity(); 

  glColor3f( .2, 1, .8 );

  float rate = (3.f * t) / MaxClock ;
  if( rate > 1.f ) rate = 1.f ;

  Points points = relocateNodes( rate ) ;

  std::vector<LineSegment> lines = theGraph->getLines( points ) ;

  glRasterPos2f( -.9, 0.9 );
  sprintf( s, "Graph: %s  %'d - %'d  Nodes: %'ld  Edges: %'d/%'ld", theGraph->name().c_str(), a, b, theGraph->numNodes(), num_edges, lines.size() ) ;
  int len = (int)strlen(s);
  for( int i = 0; i < len; i++ ) {
    glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, s[i] );
  }

  glRasterPos2f( -.9, -0.9 );
  std::vector<double> eigs = theGraph->getSingularValues() ;

  int l=0 ;
  s[0] = 0 ;
  for( auto i=0 ; i<eigs.size() ; i++ ) {
    l += sprintf( s+l, "%6.3f ", eigs[i] ) ;
    if( l > 240 ) break ;  // limit printing to bufsize
  }
  len = (int)strlen(s);
  for( int i = 0; i < len; i++ ) {
    glutBitmapCharacter( GLUT_BITMAP_HELVETICA_12, s[i] );
  }

  labelNodes( points ) ;

  glm::vec3 rgbl = hsv_to_rgb( (graph/(float)GraphFactory::NumGraphs), 1, 1 ) ;
  glColor3f( rgbl.r, rgbl.g, rgbl.b );

  // L R T B
  glOrtho( -IMG_LIMIT,  IMG_LIMIT,  -IMG_LIMIT,  IMG_LIMIT,  1,  -1) ;

  for( auto line : lines ) {
      glBegin(GL_LINES) ;
        glVertex2f( line.a.first, line.a.second ) ;
        glVertex2f( line.b.first, line.b.second ) ;
      glEnd() ;
  }

  drawNodes( points ) ;
  glutSwapBuffers();
  glutPostRedisplay();
}

Points relocateNodes( double rate ) {
  Points points ;
  int n = 0 ;
  for( int n=0 ; n<pointsNew.size() ; n++ ) {
    float xnew = pointsNew[n].first ;
    float ynew = pointsNew[n].second ;
    float xold = pointsOld.empty() ? 0 : pointsOld[n].first ;
    float yold = pointsOld.empty() ? 0 : pointsOld[n].second ;

    float x = ( xold + ( rate * ( xnew - xold ) ) )  ;
    float y = ( yold + ( rate * ( ynew - yold ) ) )  ;

    points.emplace_back(x*IMG_LIMIT,y*IMG_LIMIT) ;
  }
  return points ;
}

void labelNodes( Points points ) {
  int n = 0 ;
  char s[64] ;
  for( auto point : points ) {
    float x = point.first / IMG_LIMIT ;
    float y = point.second / IMG_LIMIT ;
    n++ ;
    glRasterPos2f( x+.035f, y );

    sprintf( s, "%'d", n ) ;
    int len = (int)strlen(s);
    for( int i = 0; i < len; i++ ) {
      glutBitmapCharacter( GLUT_BITMAP_HELVETICA_10, s[i] );
    }
  }
}

void drawNodes( std::vector<Point> points ) {

  float hue = 0.f ;
  float dhue = 1.f / points.size() ;

  for( auto point : points ) {
    float x = point.first ;    
    float y = point.second ;    

    glm::vec3 rgb = hsv_to_rgb( hue, 1, 1 ) ;
    hue += dhue ;

    glColor3f( rgb.r, rgb.g, rgb.b );
    glBegin(GL_TRIANGLE_STRIP);
  		glVertex2f( x-2, y+2 ) ;
  		glVertex2f( x-2, y-2 ) ;
  		glVertex2f( x+2, y+2 ) ;
  		glVertex2f( x+2, y-2 ) ;
    glEnd();
  }
}

glm::vec3 hsv_to_rgb(float h, float s, float v)
{
	float c = v * s;
	h *= 6 ;//glm::mod((h * 6.f), 6.f);
	float x = c * (1.f - fabs(glm::mod(h, 2.f) - 1.f));
	glm::vec3 color ;

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




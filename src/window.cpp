
#include <iostream>
#include <array>
#include <vector>

#include <pthread.h>

#include "graphFactory.hpp"

#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>
#include <GL/freeglut_ext.h> 
#include <glm/glm.hpp>  
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define IMG_LIMIT 100.0


const float time_step_size = 0.01f ;
const int world_size = 7000 ;
glm::vec3 hsv_to_rgb(float h, float s, float v ) ;
void drawNodes( Points points ) ;
Points relocateNodes() ;
void labelNodes( Points points ) ;
void mousebutton(int button,int state,int x,int y) ;
void mousemove( int x,int y) ;
void keyboard(unsigned char key, int x, int y) ;
void setSpin(double x, double y, double z) ;
void reset() ;
void reshape(int w,int h) ;
void nextGraph() ;
void changeDrawingParams() ;
void setCameraPosition() ;
void drawText( std::vector<LineSegment> lines ) ;


// ----------------------------------------------------------
// Function Prototypes
// ----------------------------------------------------------
void display();

int main( int argc, char **argv ) {

  setlocale( LC_ALL, "" ) ;

  glutInit(&argc,argv);
 
  //  Request double buffered true color window with Z-buffer
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB );
 
  // Create window
  glutCreateWindow( "Graph Layouts" );

  glutReshapeFunc(reshape);
  glutMouseFunc(mousebutton);
  glutMotionFunc(mousemove) ;
	glutKeyboardFunc(keyboard);

  glutReshapeWindow( 500, 500 );
 
  // Enable Z-buffer depth test
  glEnable(GL_DEPTH_TEST);

  // Callback functions
  glutDisplayFunc(display);

  reset() ;
  nextGraph() ;

  // Pass control to GLUT for events
  glutMainLoop(); 
}

//===================================
// GLOBAL STUFF FOR display()
//

// Next 5 items are meant to cause overflow so the
// if (x>nnn) test all fail in each loop of display
// saves us putting the init code in 2 places
int a = 10000 ; 
int b = 10000 ;
int c = 10000 ;
int graph = 10000 ;
float t = 10000.f ;  

constexpr float dt = 1.f ;
int num_edges ;
Points pointsNew ;
Points pointsOld ;
constexpr float MaxClock = 150.f ;
Graph * theGraph = GraphFactory::get( 0 ) ;
float angley ;
float anglex ;
float radius ;

void nextGraph() {
    graph++ ;
    delete theGraph ;
    if( graph>=GraphFactory::NumGraphs ) graph = 0 ;
    theGraph = GraphFactory::get(graph) ;
    pointsNew.clear() ;
    num_edges = theGraph->numEdges() ;
    a = 10000 ;
    b = 10000 ;
    c = 10000 ;
 		changeDrawingParams() ;
}

void changeDrawingParams() {

    c++ ;
    if( (c) >= theGraph->numNodes() ) { 
      b++ ;
      c = b + 1 ;
      if( (b) >= theGraph->numNodes() ) { 
        a++ ; 
        b = a + 1 ;
        if( a >= theGraph->numNodes() ) {
          a = 1 ;
          b = 2 ;
          c = 3 ;
        }
      } 
    }
    //----------------------
    // pointsOld = pointsNew    
    pointsOld.clear() ; 
    pointsOld.reserve( pointsNew.size() ) ;
    for( int i=0 ; i<pointsNew.size() ; i++ ) {
      pointsOld.emplace_back( pointsNew[i] ) ;
    }
    //----------------------
    pointsNew = theGraph->getCoords( a, b, c ) ;
}


// ----------------------------------------------------------
// display() Callback function
// ----------------------------------------------------------
void display() {
  t += dt ;

  if( t > MaxClock ) {
    t = 0 ;
  }

  //  Clear screen and Z-buffer
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  Points points = relocateNodes() ;
  std::vector<LineSegment> lines = theGraph->getLines( points ) ;

  // Reset transformations
  drawText( lines ) ;
  setCameraPosition() ;

  glColor3f( .2, 1, .8 );

  labelNodes( points ) ;

  glm::vec3 rgbl = hsv_to_rgb( (graph/(float)GraphFactory::NumGraphs), 1, 1 ) ;
  glColor3f( rgbl.r, rgbl.g, rgbl.b );

  // L R T B
  glOrtho( -IMG_LIMIT,  IMG_LIMIT,  -IMG_LIMIT,  IMG_LIMIT,  IMG_LIMIT,  -IMG_LIMIT) ;

  for( auto line : lines ) {
      glBegin(GL_LINES) ;
        glVertex3f( line.a.x(), line.a.y(), line.a.z() ) ;
        glVertex3f( line.b.x(), line.b.y(), line.b.z() ) ;
      glEnd() ;
  }

  drawNodes( points ) ;
  glutSwapBuffers();
  glutPostRedisplay() ;
}

void drawText( std::vector<LineSegment> lines ) {
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();             
  glLoadIdentity();   
  int w = glutGet( GLUT_WINDOW_WIDTH );
  int h = glutGet( GLUT_WINDOW_HEIGHT );
  glOrtho( 0, w, 0, h, -1, 1 );

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // 1st draw the text stuff w/o transformations
  glRasterPos2i( 0, h-20) ;
  char s[256] ;
  sprintf( s, "Graph: %s  %'d - %'d - %'d Nodes: %'ld  Edges: %'d/%'ld", theGraph->name().c_str(), a, b, c, theGraph->numNodes(), num_edges, lines.size() ) ;
  int len = (int)strlen(s);
  for( int i = 0; i < len; i++ ) {
    glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, s[i] );
  }

  glRasterPos2i( 0, 5) ;
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

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();  
}


void setCameraPosition() {
  glMatrixMode(GL_MODELVIEW) ;
  glLoadIdentity() ; 

  // Then figure out the camera position based on mose movement
  glm::vec3 c1(cos( angley ), 0.0, -sin( angley ));
	glm::vec3 c2(0.0, 1.0, 0.0);
	glm::vec3 c3(sin( angley ), 0.0, cos( angley ));
  glm::mat3 ry(c1, c2, c3 ) ;

	glm::vec3 c4(1.0, 0.0, 0.0);
  glm::vec3 c5(0.0, cos( anglex ), -sin( anglex ));
	glm::vec3 c6(0.0, sin( anglex ), cos( anglex ));
  glm::mat3 rx(c4, c5, c6 ) ;

  glm::vec3 camera( 0, 0, radius ) ;
  glm::vec3 camera2 = rx * ry * camera ;
  
  gluLookAt( camera2.x, camera2.y, camera2.z, 
             0.0f, 0.0f, 0.0f, /* point at */
             0.0f, 1.0f,  0.0f /* up */ 
           ) ;
}


int prev_x ;
int prev_y ;
void mousebutton(int button,int state,int x,int y)
{
	if(button == GLUT_LEFT_BUTTON ) {
		if(state==GLUT_DOWN) {
        prev_x = x ;
        prev_y = y ;
    } 
	}

	if(button == 3 ) {
    radius += 0.2f ;
  }
	if(button == 4 ) {
    radius -= 0.2f ;
  }
}


void mousemove(int x,int y)
{
    float dx = ( x - prev_x ) / 100.f ;
    float dy = ( y - prev_y ) / 100.f ;
    angley = dx ;
    anglex = dy ;
}



void keyboard(unsigned char key, int x, int y)
{
	//-------- reset -------
	if(key=='r')
	{
		reset();
	}
	//-------- next graph -------
	else if(key=='g')
	{
		nextGraph() ;
	}
	else if(key=='n')
	{
		changeDrawingParams() ;
	}
	//-------- next graph -------
	else if(key=='q')
	{
    exit(0) ;
	}
}


Points relocateNodes() {
  Points points ;
  int n = 0 ;
  for( int n=0 ; n<pointsNew.size() ; n++ ) {
    float x = pointsNew[n].x() ;
    float y = pointsNew[n].y() ;
    float z = pointsNew[n].z() ;
    points.emplace_back(x*IMG_LIMIT,y*IMG_LIMIT,z*IMG_LIMIT) ;
  }
  return points ;
}

void labelNodes( Points points ) {
  int n = 0 ;
  char s[64] ;
  for( auto point : points ) {
    float xloc = point.x() / IMG_LIMIT ;
    float yloc = point.y() / IMG_LIMIT ;
    float zloc = point.z() / IMG_LIMIT ;
    n++ ;
    // std::cout << xloc << ',' << yloc <<',' <<zloc << std::endl ;
    glRasterPos3f( xloc+.035f, yloc, zloc );

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
    float xloc = point.x() ;
    float yloc = point.y() ;
    float zloc = point.z() ;

    glm::vec3 rgb = hsv_to_rgb( hue, 1, 1 ) ;
    hue += dhue ;

    glColor3f( rgb.r, rgb.g, rgb.b );
    glBegin(GL_TRIANGLE_STRIP);
  		glVertex3f( xloc-2, yloc+2, zloc+2 ) ;
  		glVertex3f( xloc-2, yloc-2, zloc+2 ) ;
  		glVertex3f( xloc+2, yloc+2, zloc+2 ) ;
  		glVertex3f( xloc+2, yloc-2, zloc+2 ) ;
  		glVertex3f( xloc-2, yloc+2, zloc-2 ) ;
  		glVertex3f( xloc-2, yloc-2, zloc-2 ) ;
  		glVertex3f( xloc+2, yloc+2, zloc-2 ) ;
  		glVertex3f( xloc+2, yloc-2, zloc-2 ) ;
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



void reset()
{
  anglex = 0.f ;
  angley = 0.f ;
  radius = 3.f ;
}

void reshape(int w,int h)
{
	glViewport(0,0, (GLsizei)w,(GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(100.0f, (GLfloat)w/(GLfloat)h, .10f, 50.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

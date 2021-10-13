# add_library(power SHARED Connection.cpp Device.cpp )
# target_link_libraries( power pthread )
# set_target_properties( power PROPERTIES  LIBRARY_OUTPUT_DIRECTORY ".." )

# add_library(weather SHARED Weather.cpp )
# target_link_libraries( weather curl )
# set_target_properties( weather PROPERTIES  LIBRARY_OUTPUT_DIRECTORY ".." )

# add_library(json SHARED JsonParser.cpp )
# set_target_properties( json PROPERTIES  LIBRARY_OUTPUT_DIRECTORY ".." )

if( ENABLE_CUDA )
    set( LINALG_SOURCES gram-cuda.cpp eigs-cuda.cpp )
    set( LINALG_LIBS cudart cusolver cublas )
    set( LINALG_INCLUDES  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )

else( ENABLE_CUDA )
    set( LINALG_SOURCES gram-eigen.cpp eigs-eigen.cpp )
    set( LINALG_LIBS "" )
    set( LINALG_INCLUDES ${EIGEN3_INCLUDE_DIRS} )
endif( ENABLE_CUDA) 

add_executable( window window.cpp ${LINALG_SOURCES} graphFactory.cpp graph.cpp )
if( ENABLE_CUDA )
    target_link_directories( window PRIVATE ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif( ENABLE_CUDA) 
target_include_directories ( window PUBLIC ${LINALG_INCLUDES} )
target_link_libraries( window PRIVATE ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${LINALG_LIBS} )
set_target_properties( window PROPERTIES  RUNTIME_OUTPUT_DIRECTORY ".." )

add_executable( bug bug.cpp ${LINALG_SOURCES} )
if( ENABLE_CUDA )
    target_link_directories( bug PRIVATE ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif( ENABLE_CUDA) 
target_include_directories ( bug PUBLIC ${LINALG_INCLUDES} )
target_link_libraries( bug PRIVATE ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${LINALG_LIBS} )
set_target_properties( bug PROPERTIES  RUNTIME_OUTPUT_DIRECTORY ".." )
###############################################
# Webserver
# add_executable(webserver webserver.cpp ${MONGOOSE_SRC}/mongoose.c History.cpp )
# target_include_directories ( webserver PUBLIC ${MONGOOSE_SRC} )

# if( OPENSSL_FOUND )
#     message( "Compile & link server with OpenSSL" ) 
#     target_compile_definitions(webserver PUBLIC MG_ENABLE_OPENSSL )
#     # target_link_libraries( webserver PRIVATE  )
# endif()


# target_compile_definitions(webserver PUBLIC MG_ENABLE_LOG=0 )
# target_link_libraries( webserver power weather json ${OPENSSL_LIBRARIES} )
# set_target_properties( webserver PROPERTIES  RUNTIME_OUTPUT_DIRECTORY ".." )

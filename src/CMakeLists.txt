# add_library(power SHARED Connection.cpp Device.cpp )
# target_link_libraries( power pthread )
# set_target_properties( power PROPERTIES  LIBRARY_OUTPUT_DIRECTORY ".." )

# add_library(weather SHARED Weather.cpp )
# target_link_libraries( weather curl )
# set_target_properties( weather PROPERTIES  LIBRARY_OUTPUT_DIRECTORY ".." )

# add_library(json SHARED JsonParser.cpp )
# set_target_properties( json PROPERTIES  LIBRARY_OUTPUT_DIRECTORY ".." )

add_executable( window window.cpp gemm.cpp svd.cpp eigs.cpp graphFactory.cpp graph.cpp )
target_link_directories( window PRIVATE ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_include_directories ( window PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )
target_link_libraries( window PRIVATE ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} cudart cusolver cublas )
set_target_properties( window PROPERTIES  RUNTIME_OUTPUT_DIRECTORY ".." )

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

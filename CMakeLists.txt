cmake_minimum_required (VERSION 3.20)

include(FetchContent)
include(CheckLanguage)

project (graphs LANGUAGES CXX C)

option(ENABLE_CUDA "Enable the cuda build" ON )
check_language(CUDA)

if( CMAKE_CUDA_COMPILER AND ${ENABLE_CUDA}  STREQUAL "ON" )
    message( "Using CUDA library - to force non-cuda -DENABLE_CUDA=OFF" )
    enable_language( CUDA )
else (CMAKE_CUDA_COMPILER)
    set( ENABLE_CUDA "")
    find_package(  Eigen3 REQUIRED )
    message( "Using Eigen template headers " ${EIGEN3_VERSION_STRING}  " - no CUDA found" )
endif()

include(GNUInstallDirs)

set( CMAKE_CXX_FLAGS "-std=c++17" )
add_compile_options( -g )
add_compile_options( -Wfatal-errors )

FetchContent_Declare( 
                mongoose 
                GIT_REPOSITORY  "https://github.com/cesanta/mongoose.git"
                GIT_TAG         "master"  # need 7.3 for cpp support
    )

FetchContent_MakeAvailable( mongoose )  
set( MONGOOSE_SRC ${FETCHCONTENT_BASE_DIR}/mongoose-src )

find_package( OpenSSL QUIET )
find_package( OpenGL REQUIRED COMPONENTS OpenGL )
find_package( GLUT REQUIRED )

 
subdirs( src )

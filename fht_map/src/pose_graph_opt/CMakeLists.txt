cmake_minimum_required( VERSION 2.8 )
project( pose_optimization )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

# 添加cmake模块以使用ceres库
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )


add_executable(pose_graph_2d main.cpp
angle_local_parameterization.h
  normalize_angle.h
  pose_graph_2d_error_term.h
  types.h)
target_link_libraries(pose_graph_2d ${CERES_LIBRARIES} gflags)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/..)

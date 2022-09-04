cmake_minimum_required(VERSION 2.8.3)
project(eigen_linear_mpc)

FIND_PACKAGE (PythonLibs 2.7)
# Find Eigen
find_package(Eigen3)
if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})
endif()

find_package(osqp REQUIRED)

find_package(OsqpEigen REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS 
                roscpp rospy std_msgs )

## Declare a catkin package
catkin_package(INCLUDE_DIRS include
                LIBRARIES eigen_linear_mpc
                DEPENDS
                EIGEN3)

include_directories(include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS}
    )

add_library(eigen_linear_mpc
src/EigenLinearMpc.cpp
src/OsqpEigenOptimization.cpp
)


target_link_libraries(eigen_linear_mpc 
    PRIVATE osqp::osqp
    OsqpEigen::OsqpEigen
    ${catkin_LIBRARIES}
    )

install(TARGETS eigen_linear_mpc
ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

add_executable(test_example
  src/test_example.cpp
  src/EigenLinearMpc.cpp
  src/OsqpEigenOptimization.cpp
)

target_link_libraries(test_example 
    PRIVATE osqp::osqp
    OsqpEigen::OsqpEigen
    ${catkin_LIBRARIES}
    ${PYTHON_LIBRARIES}
)
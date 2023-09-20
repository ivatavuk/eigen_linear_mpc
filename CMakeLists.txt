cmake_minimum_required(VERSION 2.8.3)
project(eigen_linear_mpc)

FIND_PACKAGE (PythonLibs 2.7)

find_package(OsqpEigen REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra)

option(BUILD_SHARED_LIBS "Build libraries as shared as opposed to static" ON)

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
  "Choose the type of build, recommanded options are: Debug or Release" FORCE)
endif()

include(CMakePackageConfigHelpers)

include_directories(include
  ${PYTHON_INCLUDE_DIRS}
)

set(LIBRARY_TARGET_NAME LinMpcEigen)
set(LIBRARY_VERSION 0.0)
set(LIBRARY_HEADERS 
  include/LinMpcEigen.hpp 
  include/QpProblem.hpp
  include/OsqpEigenOptimization.hpp
)

add_library(${LIBRARY_TARGET_NAME}
src/LinMpcEigen.cpp
src/OsqpEigenOptimization.cpp
)
target_link_libraries(${LIBRARY_TARGET_NAME}
    PUBLIC OsqpEigen::OsqpEigen
)
install(TARGETS ${LIBRARY_TARGET_NAME}
  EXPORT  LinMpcEigenTargets
  COMPONENT runtime
  LIBRARY   DESTINATION   lib   
  ARCHIVE   DESTINATION   lib
  RUNTIME   DESTINATION   bin
  INCLUDES  DESTINATION   include/${LIBRARY_TARGET_NAME})

install(FILES ${LIBRARY_HEADERS}
  DESTINATION "include/${LIBRARY_TARGET_NAME}")

install(EXPORT LinMpcEigenTargets
  FILE LinMpcEigenTargets.cmake
  NAMESPACE LinMpcEigen::
  DESTINATION lib/cmake/${LIBRARY_TARGET_NAME})

write_basic_package_version_file(
  "LinMpcEigenConfigVersion.cmake"
  VERSION ${LIBRARY_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES "cmake/LinMpcEigenConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/LinMpcEigenConfigVersion.cmake"
  DESTINATION lib/cmake/${LIBRARY_TARGET_NAME})

add_custom_target(uninstall
"${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake"
)

#-----------------tests-----------------
add_executable(test_example
  src/test_example.cpp
  src/LinMpcEigen.cpp
  src/OsqpEigenOptimization.cpp
)

target_link_libraries(test_example 
  PRIVATE ${LIBRARY_TARGET_NAME}
  ${PYTHON_LIBRARIES}
)

add_executable(test_example_2
  src/test_example_2.cpp
  src/LinMpcEigen.cpp
  src/OsqpEigenOptimization.cpp
)

target_link_libraries(test_example_2 
  PRIVATE ${LIBRARY_TARGET_NAME}
  ${PYTHON_LIBRARIES}
)

add_executable(test_example_3
  src/test_example_3.cpp
  src/LinMpcEigen.cpp
  src/OsqpEigenOptimization.cpp
)

target_link_libraries(test_example_3 
  PRIVATE ${LIBRARY_TARGET_NAME}
  ${PYTHON_LIBRARIES}
)
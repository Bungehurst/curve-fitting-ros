cmake_minimum_required(VERSION 3.0.2)

project(curve_fitting)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
    cmake_modules
    rosconsole 
    roscpp
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
# find_package(G2O REQUIRED)

add_subdirectory(3rdparty/matplotplusplus)
add_subdirectory(3rdparty/g2o)

include_directories(
    include/${PROJECT_NAME}
    ${catkin_INCLUDE_DIRS} 
    ${EIGEN_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIR}
)

add_definitions(${EIGEN_DEFINITIONS})

catkin_package()

add_executable(gncurvefit gaussianNewton.cpp)
target_link_libraries(gncurvefit PUBLIC matplot 
                        ${catkin_LIBRARIES} 
                        ${OpenCV_LIBRARIES} 
                        ${CERES_LIBRARIES}
                        )

add_executable(cerescurvefit ceresCurvefitting.cpp)
target_link_libraries(cerescurvefit PUBLIC matplot 
                        ${catkin_LIBRARIES} 
                        ${OpenCV_LIBRARIES} 
                        ${CERES_LIBRARIES}
                        )

add_executable(g2ocurvefit g2oCurvefitting.cpp)
target_link_libraries(g2ocurvefit PUBLIC matplot 
                        ${catkin_LIBRARIES} 
                        ${OpenCV_LIBRARIES} 
                        ${PROJECT_SOURCE_DIR}/3rdparty/g2o/lib/libg2o.so
                        )
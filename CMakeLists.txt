cmake_minimum_required(VERSION 3.9)

project(SRCNN_cpp)

set(i_want_openmp OFF)
set(EXECUTABLE_OUTPUT_PATH "..")
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)

# include OpenCV
find_package(OpenCV REQUIRED)

# include OpenMP 
find_package(OpenMP)

include_directories(src)
add_executable(SRCNN_cpp 
    main.cpp 
    src/srcnn.cpp
    src/gaussian.cpp)

if(OpenMP_CXX_FOUND AND i_want_openmp)
    target_link_libraries(SRCNN_cpp PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
else()
    target_link_libraries(SRCNN_cpp ${OpenCV_LIBS})
endif()

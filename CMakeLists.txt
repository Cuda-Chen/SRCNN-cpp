cmake_minimum_required(VERSION 3.9)

project(SRCNN_cpp)

set(i_want_openmp ON)
set(EXECUTABLE_OUTPUT_PATH "..")
set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_BUILD_TYPE Release)

# include OpenCV
find_package(OpenCV REQUIRED)

# include OpenMP 
find_package(OpenMP)

include_directories(src)
add_executable(SRCNN_cpp 
    main.cpp 
    src/srcnn.cpp
    src/gaussian.cpp)
#set_property(TARGET SRCNN_cpp PROPERTY COMPILE_FLAGS "-Ofast")
#set_property(TARGET SRCNN_cpp PROPERTY COMPILE_FLAGS "-Ofast -ftree-vectorize -march=native -mtune=native -mavx2 -DIM2COL=1")

if(OpenMP_CXX_FOUND AND i_want_openmp)
    message("SRCNN_cpp is built with OpenCV and OpenMP is enabled.")
    set_property(TARGET SRCNN_cpp PROPERTY COMPILE_FLAGS "-Ofast -ftree-vectorize -march=native -mtune=native -mavx2 -DIM2COL=1")
    target_link_libraries(SRCNN_cpp PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
else()
    message("SRCNN_cpp is built with only OpenCV.")
    set_property(TARGET SRCNN_cpp PROPERTY COMPILE_FLAGS "-Ofast -ftree-vectorize -march=native -mtune=native -mavx2")
    target_link_libraries(SRCNN_cpp ${OpenCV_LIBS})
endif()

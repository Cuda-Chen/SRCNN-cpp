cmake_minimum_required(VERSION 3.1)

project(SRCNN_cpp)

set(EXECUTABLE_OUTPUT_PATH "..")
set(CMAKE_CXX_STANDARD 11)

# OpenCV stuff
find_package(OpenCV REQUIRED)

include_directories(src)
add_executable(SRCNN_cpp 
    main.cpp 
    src/srcnn.cpp)

target_link_libraries(SRCNN_cpp ${OpenCV_LIBS})

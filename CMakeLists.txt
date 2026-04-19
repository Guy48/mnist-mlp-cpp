cmake_minimum_required(VERSION 3.22)
project(mnist-mlp-cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mnist_mlp
    src/Activation.cpp
    src/Dense.cpp
    src/Matrix.cpp
    src/MlpNetwork.cpp
)

target_include_directories(mnist_mlp PUBLIC include)

add_executable(mnist_cli src/main.cpp)
target_link_libraries(mnist_cli PRIVATE mnist_mlp)

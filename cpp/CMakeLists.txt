cmake_minimum_required(VERSION 3.5)
project(scenecompletion)

set(CMAKE_CXX_STANDARD 14)

add_definitions(-DGSL_THROW_ON_CONTRACT_VIOLATION)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Needed to build shared objects for python ctypes.

#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -mfpmath=both")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

include_directories(${CMAKE_SOURCE_DIR})


add_subdirectory(third_party)
add_subdirectory(lib)
add_subdirectory(apps)
add_subdirectory(tests)

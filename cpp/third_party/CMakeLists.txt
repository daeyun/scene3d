set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/third_party/cmake/Modules/")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(BUILD_SHARED_LIBS OFF)

set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)


add_library(eigen3 INTERFACE)
target_include_directories(eigen3 SYSTEM BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/eigen
        )

add_library(cpp_gsl INTERFACE)
target_include_directories(cpp_gsl SYSTEM BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/GSL/include
        )

add_library(spdlog INTERFACE)
target_include_directories(spdlog SYSTEM BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/spdlog/include
        )
target_link_libraries(spdlog INTERFACE Threads::Threads)

add_library(assimp INTERFACE)
target_link_libraries(assimp INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/install/assimp/lib/libassimp.a
        ${CMAKE_SOURCE_DIR}/third_party/install/assimp/lib/libIrrXML.a
        z
        )
target_include_directories(assimp SYSTEM BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/install/assimp/include
        )
#target_compile_options(assimp INTERFACE ${OpenMP_CXX_FLAGS})

add_library(nanort INTERFACE)
target_include_directories(nanort SYSTEM BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/nanort
        )


add_library(blosc INTERFACE)
target_link_libraries(blosc INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/install/c-blosc/lib/libblosc.a
        )
target_include_directories(blosc SYSTEM BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/install/c-blosc/include
        )

add_library(cxxopts INTERFACE)
target_include_directories(cxxopts SYSTEM BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/cxxopts/include
        )

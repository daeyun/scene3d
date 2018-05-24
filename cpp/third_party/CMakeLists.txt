set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/third_party/cmake/Modules/")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(BUILD_SHARED_LIBS OFF)

set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)


add_library(eigen3 INTERFACE)
target_include_directories(eigen3 BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/eigen
        )

add_library(cpp_gsl INTERFACE)
target_include_directories(cpp_gsl BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/GSL/include
        )

add_library(spdlog INTERFACE)
target_include_directories(spdlog BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/spdlog/include
        )
target_link_libraries(spdlog INTERFACE Threads::Threads)

add_library(assimp INTERFACE)
target_link_libraries(assimp INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/install/assimp/lib/libassimp.a
        ${CMAKE_SOURCE_DIR}/third_party/install/assimp/lib/libIrrXML.a
        z
        )
target_include_directories(assimp BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/install/assimp/include
        )
#target_compile_options(assimp INTERFACE ${OpenMP_CXX_FLAGS})

add_library(nanort INTERFACE)
target_include_directories(nanort BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/nanort
        )


add_library(blosc INTERFACE)
target_link_libraries(blosc INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/install/c-blosc/lib/libblosc.a
        )
target_include_directories(blosc BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/install/c-blosc/include
        )

add_library(cxxopts INTERFACE)
target_include_directories(cxxopts BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/cxxopts/include
        )

add_library(catch2 INTERFACE)
target_include_directories(catch2 BEFORE INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/Catch2/single_include
        )

# mve

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(TIFF REQUIRED)

add_library(mve INTERFACE)
target_link_libraries(mve INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/mve/libs/mve/libmve.a
        ${CMAKE_SOURCE_DIR}/third_party/repos/mve/libs/dmrecon/libmve_dmrecon.a
        ${CMAKE_SOURCE_DIR}/third_party/repos/mve/libs/ogl/libmve_ogl.a
        ${CMAKE_SOURCE_DIR}/third_party/repos/mve/libs/sfm/libmve_sfm.a
        ${CMAKE_SOURCE_DIR}/third_party/repos/mve/libs/fssr/libmve_fssr.a
        ${CMAKE_SOURCE_DIR}/third_party/repos/mve/libs/util/libmve_util.a
        ${PNG_LIBRARY} ${JPEG_LIBRARY} ${TIFF_LIBRARY}
        z
        lzma
        jbig
        )
target_include_directories(mve INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/repos/mve/libs/
        ${PNG_INCLUDE_DIR}
        ${JPEG_INCLUDE_DIR}
        ${TIFF_INCLUDE_DIR}
        )
target_compile_options(mve INTERFACE ${OpenMP_CXX_FLAGS})

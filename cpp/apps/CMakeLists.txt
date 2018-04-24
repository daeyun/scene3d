add_executable(render render.cpp)
target_link_libraries(render common cpp_gsl spdlog assimp file_io nanort cxxopts)

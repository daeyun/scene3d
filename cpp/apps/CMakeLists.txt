add_executable(render render.cpp)
target_link_libraries(render common cpp_gsl ray_mesh_intersection spdlog assimp file_io nanort cxxopts)

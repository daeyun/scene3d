add_executable(render render.cpp)
target_link_libraries(render common cpp_gsl spdlog assimp file_io cxxopts ray_mesh_intersection)
set_target_properties(render PROPERTIES LINK_FLAGS "-static")

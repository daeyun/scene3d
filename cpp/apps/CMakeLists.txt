add_executable(render render.cpp)
target_link_libraries(render common cpp_gsl spdlog assimp file_io cxxopts ray_mesh_intersection)
set_target_properties(render PROPERTIES LINK_FLAGS "-static")

add_executable(depth2mesh_perspective depth2mesh_perspective.cc)
target_link_libraries(depth2mesh_perspective mve)

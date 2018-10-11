#add_executable(render render.cpp)
#target_link_libraries(render common cpp_gsl spdlog assimp file_io cxxopts ray_mesh_intersection csv_parser suncg_utils)

# Should be uncommented when building release version.
#set_target_properties(render PROPERTIES LINK_FLAGS "-static")

add_executable(depth2mesh_perspective depth2mesh_perspective.cc)
target_link_libraries(depth2mesh_perspective mve)

add_executable(render_suncg render_suncg.cpp)
target_link_libraries(render_suncg common cpp_gsl spdlog assimp file_io cxxopts ray_mesh_intersection csv_parser suncg_utils benchmark pcl)

add_executable(gt_coverage gt_coverage.cc)
target_link_libraries(gt_coverage common cpp_gsl spdlog assimp cxxopts depth_render_utils meshdist suncg_utils benchmark)

# Should be uncommented when building release version.
#set_target_properties(render_suncg PROPERTIES LINK_FLAGS "-static")

#add_executable(render render.cpp)
#target_link_libraries(render common cpp_gsl spdlog assimp file_io cxxopts ray_mesh_intersection csv_parser suncg_utils)

# Should be uncommented when building release version.
#set_target_properties(render PROPERTIES LINK_FLAGS "-static")

add_executable(depth2mesh_perspective depth2mesh_perspective.cc)
target_link_libraries(depth2mesh_perspective mve)

add_executable(render_suncg render_suncg.cpp)
target_link_libraries(render_suncg common cpp_gsl spdlog assimp file_io cxxopts ray_mesh_intersection csv_parser suncg_utils benchmark pcl)

add_executable(find_floor_height find_floor_height.cpp)
target_link_libraries(find_floor_height common cpp_gsl spdlog assimp file_io cxxopts ray_mesh_intersection csv_parser suncg_utils benchmark pcl)

add_executable(gt_coverage gt_coverage.cc)
target_link_libraries(gt_coverage common cpp_gsl spdlog assimp cxxopts depth_render_utils meshdist suncg_utils benchmark)

add_executable(pred_recon pred_recon.cc)
target_link_libraries(pred_recon common cpp_gsl spdlog assimp cxxopts depth_render_utils meshdist suncg_utils benchmark)

add_executable(generate_gt_mesh generate_gt_mesh.cc)
target_link_libraries(generate_gt_mesh common cpp_gsl spdlog assimp cxxopts depth_render_utils meshdist suncg_utils benchmark string_utils)

add_executable(extract_frustum_mesh extract_frustum_mesh.cc)
target_link_libraries(extract_frustum_mesh common cpp_gsl spdlog file_io assimp cxxopts depth_render_utils benchmark string_utils)

add_executable(ply2obj mesh2obj.cc)
target_link_libraries(ply2obj common cpp_gsl spdlog file_io cxxopts)

# Should be uncommented when building release version.
#set_target_properties(render_suncg PROPERTIES LINK_FLAGS "-static")

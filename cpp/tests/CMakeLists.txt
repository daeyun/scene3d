add_definitions(-DGSL_THROW_ON_CONTRACT_VIOLATION)

find_package(Boost REQUIRED COMPONENTS system filesystem)

add_executable(ray_tracing_test ray_tracing_test.cc test_main.cc)
target_link_libraries(ray_tracing_test common catch2 Boost::filesystem CGAL ray_mesh_intersection_cgal)

add_executable(depth_renderer_test depth_renderer_test.cc test_main.cc)
target_link_libraries(depth_renderer_test common catch2 Boost::filesystem multi_layer_depth_renderer file_io ray_mesh_intersection suncg_utils pcl)

add_executable(depth_render_utils_test depth_render_utils_test.cc test_main.cc)
target_link_libraries(depth_render_utils_test common catch2 Boost::filesystem multi_layer_depth_renderer file_io depth_render_utils suncg_utils pcl)

add_executable(pcl_test pcl_test.cc test_main.cc)
target_link_libraries(pcl_test common catch2 Boost::filesystem multi_layer_depth_renderer file_io depth_render_utils pcl meshdist benchmark suncg_utils pcl)

add_executable(camera_test camera_test.cc test_main.cc)
target_link_libraries(camera_test common catch2 Boost::filesystem multi_layer_depth_renderer file_io depth_render_utils pcl vectorization_utils pcl)

add_executable(suncg_utils_test suncg_utils_test.cc test_main.cc)
target_link_libraries(suncg_utils_test common catch2 file_io depth_render_utils pcl vectorization_utils camera suncg_utils)

add_executable(epipolar_test epipolar_test.cc test_main.cc)
target_link_libraries(epipolar_test common catch2 file_io depth_render_utils pcl vectorization_utils camera epipolar)

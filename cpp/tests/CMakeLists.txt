find_package(Boost REQUIRED COMPONENTS system filesystem)

add_executable(ray_tracing_test ray_tracing_test.cc test_main.cc)
target_link_libraries(ray_tracing_test common catch2 Boost::filesystem CGAL ray_mesh_intersection_cgal)

add_executable(depth_renderer_test depth_renderer_test.cc test_main.cc)
target_link_libraries(depth_renderer_test common catch2 Boost::filesystem multi_layer_depth_renderer file_io ray_mesh_intersection)

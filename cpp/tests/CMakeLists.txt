find_package(Boost REQUIRED COMPONENTS system filesystem)

add_executable(ray_tracing_test ray_tracing_test.cc test_main.cc)
target_link_libraries(ray_tracing_test common catch2 Boost::filesystem CGAL ray_mesh_intersection)

find_package(Boost REQUIRED COMPONENTS system filesystem)

add_library(common common.cc)
target_link_libraries(common PUBLIC cpp_gsl eigen3 spdlog)

add_library(file_io file_io.cc file_io.h)
target_link_libraries(file_io spdlog assimp common cpp_gsl Boost::filesystem blosc)

add_library(camera camera.cc camera.h)
target_link_libraries(camera common cpp_gsl file_io)

add_library(ray_mesh_intersection_cgal ray_mesh_intersection_cgal.cc ray_mesh_intersection_cgal.h)
target_link_libraries(ray_mesh_intersection_cgal PRIVATE common CGAL)

add_library(ray_mesh_intersection ray_mesh_intersection.cc ray_mesh_intersection.h)
target_link_libraries(ray_mesh_intersection PUBLIC common nanort)

add_library(multi_layer_depth_renderer multi_layer_depth_renderer.cc multi_layer_depth_renderer.h)
target_link_libraries(multi_layer_depth_renderer PUBLIC common nanort ray_mesh_intersection file_io camera)

add_library(depth_render_utils depth_render_utils.h)
target_link_libraries(depth_render_utils PUBLIC common multi_layer_depth_renderer)

add_library(depth depth.h)
target_link_libraries(depth PUBLIC common camera)

add_library(pcl pcl.h pcl.cc)
target_link_libraries(pcl PUBLIC common camera depth suncg_utils)

add_library(benchmark benchmark.h benchmark.cc)
target_link_libraries(benchmark PUBLIC common)

add_library(meshdist meshdist.h meshdist.cc)
target_link_libraries(meshdist PUBLIC common camera depth benchmark CGAL)
target_compile_options(meshdist BEFORE PUBLIC "-lCGAL")

add_library(vectorization_utils vectorization_utils.h)
target_link_libraries(vectorization_utils PUBLIC common)

add_library(string_utils string_utils.h)
target_link_libraries(string_utils PUBLIC common)

add_library(suncg_utils suncg_utils.h suncg_utils.cc)
target_link_libraries(suncg_utils PUBLIC common file_io camera nlohmann_json string_utils assimp Boost::filesystem csv_parser lrucache pthread benchmark pcl)

add_library(epipolar epipolar.h)
target_link_libraries(epipolar PUBLIC common file_io camera benchmark pcl)

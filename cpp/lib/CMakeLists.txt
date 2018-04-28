find_package(Boost REQUIRED COMPONENTS system filesystem)

add_library(common common.cc)
target_link_libraries(common PUBLIC cpp_gsl eigen3 spdlog)

add_library(file_io file_io.cc file_io.h)
target_link_libraries(file_io spdlog assimp common cpp_gsl Boost::filesystem blosc)

add_library(camera camera.cc camera.h)
target_link_libraries(camera common cpp_gsl)

add_library(ray_mesh_intersection_cgal ray_mesh_intersection_cgal.cc ray_mesh_intersection_cgal.h)
target_link_libraries(ray_mesh_intersection_cgal PRIVATE common CGAL)

add_library(ray_mesh_intersection ray_mesh_intersection.cc ray_mesh_intersection.h)
target_link_libraries(ray_mesh_intersection PUBLIC common nanort)

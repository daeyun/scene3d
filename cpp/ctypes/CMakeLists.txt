add_library(epipolar_transform_ctypes SHARED epipolar_transform.cc)
target_link_libraries(epipolar_transform_ctypes eigen3 epipolar benchmark pthread)

add_library(training_utils SHARED training_utils.cc)
target_link_libraries(training_utils eigen3 benchmark pthread suncg_utils)

add_library(depth_mesh_ctypes SHARED depth_mesh_utils_ctypes.cc)
target_link_libraries(depth_mesh_ctypes eigen3 benchmark pthread depth_mesh_utils)

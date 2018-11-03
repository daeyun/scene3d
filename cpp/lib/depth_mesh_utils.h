//
// Created by daeyun on 5/14/17.
//
#pragma once

#include <list>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <omp.h>

#include "lib/common.h"
#include "lib/mesh.h"

namespace scene3d {

// `camera_filename` is a text file containing one camera per line.
// `camera_index` is 0 if there's only one camera.
// `dd_factor` is 0 if there's only one camera.
void DepthToMesh(const float *depth_data,
                 uint32_t source_height,
                 uint32_t source_width,
                 const char *camera_filename,
                 uint32_t camera_index,
                 float dd_factor,
                 const char *out_filename);

void MeshPrecisionRecall(const char **gt_mesh_filenames,   // These meshes will be merged.
                         uint32_t num_gt_mesh_filenames,
                         const char **pred_mesh_filenames, // These meshes will be merged.
                         uint32_t num_pred_mesh_filenames,
                         float sampling_density,
                         const float *thresholds,
                         uint32_t num_thresholds,
                         std::vector<float> *out_precision,
                         std::vector<float> *out_recall);

}
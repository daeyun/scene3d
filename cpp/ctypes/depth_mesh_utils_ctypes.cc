//
// Created by daeyun on 9/9/18.
//

#include <string>
#include <omp.h>
#include "lib/common.h"
#include "lib/epipolar.h"
#include "lib/benchmark.h"
#include "lib/depth_mesh_utils.h"

extern "C" {
void depth_to_mesh(const float *depth_data,
                   uint32_t source_height,
                   uint32_t source_width,
                   const char *camera_filename,
                   uint32_t camera_index,
                   float dd_factor,
                   const char *out_filename);

void mesh_precision_recall(const char **gt_mesh_filename,     // These meshes will be merged.
                           uint32_t num_gt_mesh_filenames,
                           const char **pred_mesh_filenames,  // These meshes will be merged.
                           uint32_t num_pred_mesh_filenames,
                           float sampling_density,
                           const float *thresholds,
                           uint32_t num_thresholds,
                           float *out_precision,
                           float *out_recall);
}

void depth_to_mesh(const float *depth_data,
                   uint32_t source_height,
                   uint32_t source_width,
                   const char *camera_filename,
                   uint32_t camera_index,
                   float dd_factor,
                   const char *out_filename) {
  if (spdlog::get("console") == nullptr) {
    spdlog::stdout_color_mt("console");
  }

  auto start_time = scene3d::TimeSinceEpoch<std::milli>();
  scene3d::DepthToMesh(depth_data,
                       source_height,
                       source_width,
                       camera_filename,
                       camera_index,
                       dd_factor,
                       out_filename);
  LOGGER->info("Elapsed (DepthToMesh, total): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);

}

void mesh_precision_recall(const char **gt_mesh_filename,
                           uint32_t num_gt_mesh_filenames,
                           const char **pred_mesh_filenames,
                           uint32_t num_pred_mesh_filenames,
                           float sampling_density,
                           const float *thresholds,
                           uint32_t num_thresholds,
                           float *out_precision,
                           float *out_recall) {
  if (spdlog::get("console") == nullptr) {
    spdlog::stdout_color_mt("console");
  }

  auto start_time = scene3d::TimeSinceEpoch<std::milli>();

  std::vector<float> out_precision_vector;
  std::vector<float> out_recall_vector;

  scene3d::MeshPrecisionRecall(gt_mesh_filename,
                               num_gt_mesh_filenames,
                               pred_mesh_filenames,
                               num_pred_mesh_filenames,
                               sampling_density,
                               thresholds,
                               num_thresholds,
                               &out_precision_vector,
                               &out_recall_vector);

  memcpy(out_precision, out_precision_vector.data(), out_precision_vector.size() * sizeof(float));
  memcpy(out_recall, out_recall_vector.data(), out_recall_vector.size() * sizeof(float));

  LOGGER->info("Elapsed (MeshPrecisionRecall, total): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);
}

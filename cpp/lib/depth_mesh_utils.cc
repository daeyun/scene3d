//
// Created by daeyun on 5/10/17.
//

#include "depth_mesh_utils.h"

#include <Eigen/Dense>
#include <type_traits>
#include <chrono>

#include "lib/common.h"
#include "lib/random_utils.h"
#include "lib/benchmark.h"
#include "lib/depth_mesh_utils.h"
#include "lib/depth.h"
#include "lib/meshdist.h"
#include "lib/pcl.h"
#include "lib/string_utils.h"

namespace scene3d {
void DepthToMesh(const float *depth_data, uint32_t source_height, uint32_t source_width, const char *camera_filename, uint32_t camera_index, float dd_factor, const char *out_filename) {
  const string camera_filename_str(camera_filename);
  LOGGER->debug("Camera filename: {}", camera_filename_str);
  LOGGER->debug("source dimension: ({}, {})", source_height, source_width);

  Expects(source_height < 4096);
  Expects(source_width < 4096);

  vector<unique_ptr<scene3d::Camera>> cameras;
  ReadCameras(camera_filename_str, &cameras);
  Expects(!cameras.empty());
  Expects(camera_index < cameras.size());

  Image<float> depth(depth_data, source_height, source_width, NAN);

  TriMesh mesh;
  TriangulateDepth(depth, *cameras[camera_index], dd_factor, &mesh.faces, &mesh.vertices);

  const string out_filename_str(out_filename);
  PrepareDirForFile(out_filename_str);
  WritePly(out_filename_str, mesh.faces, mesh.vertices, true);
  if (scene3d::EndsWith(out_filename_str, ".ply")) {
    WritePly(out_filename_str, mesh.faces, mesh.vertices, true);
  } else if (scene3d::EndsWith(out_filename_str, ".obj")) {
    WriteObj(out_filename_str, mesh.faces, mesh.vertices);
  } else {
    throw std::runtime_error("Invalid output file format.");
  }
  LOGGER->info("Wrote {}", out_filename_str);
}

vector<string> LoadFilenames(const char **strings, uint32_t num) {
  vector<string> ret;
  ret.reserve((size_t) num);
  for (int i = 0; i < num; ++i) {
    ret.emplace_back(strings[i]);
    if (!Exists(ret.at(static_cast<size_t>(i)))) {
      LOGGER->error("File does not exist: {}", ret.at(static_cast<size_t>(i)));
    }
  }
  return ret;
}

void MeshPrecisionRecall(const char **gt_mesh_filenames,
                         uint32_t num_gt_mesh_filenames,
                         const char **pred_mesh_filenames,
                         uint32_t num_pred_mesh_filenames,
                         float sampling_density,
                         const float *thresholds,
                         uint32_t num_thresholds,
                         std::vector<float> *out_precision,
                         std::vector<float> *out_recall) {
  Expects(num_thresholds > 0);
  Expects(num_pred_mesh_filenames > 0);
  Expects(sampling_density > 0.0f);

  vector<string> pred_mesh_filenames_vector = LoadFilenames(pred_mesh_filenames, num_pred_mesh_filenames);
  vector<string> gt_mesh_filenames_vector = LoadFilenames(gt_mesh_filenames, num_gt_mesh_filenames);

  vector<float> squared_thresholds;
  for (int j = 0; j < num_thresholds; ++j) {
    Expects(thresholds[j] >= 0.0f);
    squared_thresholds.push_back(thresholds[j] * thresholds[j]);
  }

  auto load_and_merge_meshes = [&](const vector<string> &filenames, vector<meshdist_cgal::Triangle> *out) {
    for (const auto &fname : filenames) {
      ReadTriangles(fname,
                    [&](const array<array<float, 3>, 3> triangle) {
                      out->emplace_back(
                          Vec3{triangle[0][0], triangle[0][1], triangle[0][2]},
                          Vec3{triangle[1][0], triangle[1][1], triangle[1][2]},
                          Vec3{triangle[2][0], triangle[2][1], triangle[2][2]}
                      );
                    });
    }
  };

  const auto prev_log_level = LOGGER->level();
  LOGGER->set_level(spdlog::level::debug);  // MeshToMeshDistanceOneDirection will log elapsed times.

  double start_time;
  start_time = scene3d::TimeSinceEpoch<std::milli>();
  vector<meshdist_cgal::Triangle> gt_triangles;
  load_and_merge_meshes(gt_mesh_filenames_vector, &gt_triangles);
  LOGGER->info("Elapsed (ReadTriangles, GT): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);

  start_time = scene3d::TimeSinceEpoch<std::milli>();
  vector<meshdist_cgal::Triangle> pred_triangles;  // Polygon soup.
  load_and_merge_meshes(pred_mesh_filenames_vector, &pred_triangles);
  LOGGER->info("Elapsed (ReadTriangles, Pred): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);

  std::vector<float> recall_squared_distances;
  meshdist_cgal::MeshToMeshDistanceOneDirection(
      gt_triangles,    // from
      pred_triangles,  // to
      sampling_density,
      &recall_squared_distances);

  std::vector<float> precision_squared_distances;
  meshdist_cgal::MeshToMeshDistanceOneDirection(
      pred_triangles,  // from
      gt_triangles,    // to
      sampling_density,
      &precision_squared_distances);

  LOGGER->set_level(prev_log_level);  // restore log level.

  auto compute_inlier_ratio = [&](const std::vector<float> &distances, float threshold) -> double {
    if (distances.empty()) {
      return 0;
    }

    unsigned int inlier_count = 0;
    for (const float &d : distances) {
      if (d < threshold) {
        ++inlier_count;
      }
    }
    return static_cast<double>(inlier_count) / static_cast<double>(distances.size());
  };

  start_time = scene3d::TimeSinceEpoch<std::milli>();
  for (const float squared_threshold : squared_thresholds) {
    out_recall->push_back(static_cast<float>(compute_inlier_ratio(recall_squared_distances, squared_threshold)));
    out_precision->push_back(static_cast<float>(compute_inlier_ratio(precision_squared_distances, squared_threshold)));
  }
  LOGGER->info("Elapsed (inliers): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);
}
}

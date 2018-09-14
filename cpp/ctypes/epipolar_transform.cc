//
// Created by daeyun on 9/9/18.
//

#include <string>
#include <omp.h>
#include "lib/common.h"
#include "lib/epipolar.h"
#include "lib/benchmark.h"

extern "C" {
void epipolar_feature_transform(const float *feature_map_data,
                                const float *front_depth_data,
                                const float *back_depth_data,
                                uint32_t source_height,
                                uint32_t source_width,
                                uint32_t source_channels,
                                const char *camera_filename,
                                uint32_t target_height,
                                uint32_t target_width,
                                float *out);

void epipolar_feature_transform_parallel(const float *feature_map_data,
                                         const float *front_depth_data,
                                         const float *back_depth_data,
                                         uint32_t batch_size,
                                         uint32_t source_height,
                                         uint32_t source_width,
                                         uint32_t source_channels,
                                         const char **camera_filenames,
                                         uint32_t target_height,
                                         uint32_t target_width,
                                         float *out);
}

void epipolar_feature_transform(
    const float *feature_map_data,
    const float *front_depth_data,
    const float *back_depth_data,
    uint32_t source_height,
    uint32_t source_width,
    uint32_t source_channels,
    const char *camera_filename,
    uint32_t target_height,
    uint32_t target_width,
    float *out
) {
  if (spdlog::get("console") == nullptr) {
    spdlog::stdout_color_mt("console");
  }

  vector<float> transformed;

  auto start_time = scene3d::TimeSinceEpoch<std::milli>();
  scene3d::EpipolarFeatureTransform(feature_map_data,
                                    front_depth_data,
                                    back_depth_data,
                                    source_height,
                                    source_width,
                                    source_channels,
                                    camera_filename,
                                    target_height,
                                    target_width,
                                    &transformed);
  LOGGER->info("Elapsed: {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);

  size_t size_bytes = transformed.size() * sizeof(float);
  memcpy(out, transformed.data(), size_bytes);
}

void epipolar_feature_transform_parallel(const float *feature_map_data,
                                         const float *front_depth_data,
                                         const float *back_depth_data,
                                         uint32_t batch_size,
                                         uint32_t source_height,
                                         uint32_t source_width,
                                         uint32_t source_channels,
                                         const char **camera_filenames,
                                         uint32_t target_height,
                                         uint32_t target_width,
                                         float *out) {
  if (spdlog::get("console") == nullptr) {
    spdlog::stdout_color_mt("console");
  }

  std::vector<std::string> filenames_vector;
  std::vector<std::unique_ptr<vector<float>>> transformed_batch;
  filenames_vector.reserve((size_t) batch_size);
  transformed_batch.reserve((size_t) batch_size);
  for (int i = 0; i < batch_size; ++i) {
    filenames_vector.emplace_back(camera_filenames[i]);
    transformed_batch.push_back(std::make_unique<vector<float>>());
  }

#pragma omp parallel for schedule(dynamic) num_threads(12)
  for (int i = 0; i < batch_size; ++i) {
    const size_t read_offset = i * source_height * source_width;
    LOGGER->info("[thread {}] {}: {}", omp_get_thread_num(), i, filenames_vector[i]);
    auto start_time = scene3d::TimeSinceEpoch<std::milli>();
    scene3d::EpipolarFeatureTransform(feature_map_data + read_offset * source_channels,
                                      front_depth_data + read_offset,
                                      back_depth_data + read_offset,
                                      source_height,
                                      source_width,
                                      source_channels,
                                      filenames_vector[i].c_str(),
                                      target_height,
                                      target_width,
                                      transformed_batch[i].get());
    LOGGER->info("[thread {}] {}: Elapsed: {} ms", omp_get_thread_num(), i, scene3d::TimeSinceEpoch<std::milli>() - start_time);
  }

  auto start_time = scene3d::TimeSinceEpoch<std::milli>();
#pragma omp parallel for schedule(dynamic) num_threads(6)
  for (int i = 0; i < batch_size; ++i) {
    size_t item_count = transformed_batch[i]->size();
    size_t out_offset = i * item_count;  // `item_count` needs to be H*W*C.
    memcpy(out + out_offset, transformed_batch[i]->data(), item_count * sizeof(float));
  }
  LOGGER->info("Elapsed (memcpy): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);

}

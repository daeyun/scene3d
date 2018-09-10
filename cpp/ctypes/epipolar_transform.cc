//
// Created by daeyun on 9/9/18.
//

#include <string.h>
#include "lib/common.h"
#include "lib/epipolar.h"
#include "lib/benchmark.h"

extern "C" {
void epipolar_feature_transform(const float *feature_map_data,
                                const float *front_depth_data,
                                const float *back_depth_data,
                                uint32_t source_channels,
                                uint32_t source_height,
                                uint32_t source_width,
                                const char *camera_filename,
                                uint32_t target_height,
                                uint32_t target_width,
                                float *out);
}

void epipolar_feature_transform(
    const float *feature_map_data,
    const float *front_depth_data,
    const float *back_depth_data,
    uint32_t source_channels,
    uint32_t source_height,
    uint32_t source_width,
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
                                    source_channels,
                                    source_height,
                                    source_width,
                                    camera_filename,
                                    target_height,
                                    target_width,
                                    &transformed);
  LOGGER->info("Elapsed: {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);

  size_t size = transformed.size() * sizeof(float);
  memcpy(out, transformed.data(), size);
}

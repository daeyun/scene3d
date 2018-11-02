#pragma once

#include <algorithm>

#include "camera.h"
#include "pcl.h"

namespace scene3d {

struct XYLineSegment {
  // Can be negative if projected outside of image frame.
  array<int32_t, 2> xy1{0, 0};
  array<int32_t, 2> xy2{0, 0};
  bool has_xy2{false};  // If false, this is a segment of length 1.
};

// For each non-nan pixel in the front and back input depth images, find the starting and ending XY coordinates in the target camera.
void EpipolarLineSegmentCoordinates(const Image<float> &front_depth, const Image<float> &back_depth,
                                    const Camera &source_camera, const Camera &target_camera,
                                    unsigned int target_height, unsigned int target_width, Image<unique_ptr<XYLineSegment>> *epipolar_mapping);

void LineCoordinates(int x1, int y1, int x2, int y2, vector<array<int, 2>> *xy);

void LineCoordinatesValidRange(int x1, int y1, int x2, int y2, int height, int width, vector<array<int, 2>> *xy);

// Memory ordering must be (H, W, C).
void EpipolarFeatureTransform(const float *feature_map_data,
                              const float *front_depth_data,
                              const float *back_depth_data,
                              uint32_t source_height,
                              uint32_t source_width,
                              uint32_t source_channels,
                              const char *camera_filename,
                              uint32_t target_height,
                              uint32_t target_width,
                              std::vector<float> *transformed);

// Memory ordering must be (N, H, W).
void RenderDepthFromAnotherView(const float *depth_data,
                                uint32_t source_height,
                                uint32_t source_width,
                                uint32_t num_images,
                                const char *camera_filename,
                                uint32_t target_height,
                                uint32_t target_width,
                                float depth_disc_pixels,
                                std::vector<float> *transformed);

void FrustumVisibilityMapFromOverheadView(const char *camera_filename,
                                          uint32_t target_height,
                                          uint32_t target_width,
                                          std::vector<float> *transformed);

}  // namespace scene3d

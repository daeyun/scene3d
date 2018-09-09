#pragma once

#include "camera.h"
#include "pcl.h"

namespace scene3d {

struct XYLineSegment {
  // Can be negative if projected outside of image frame.
  array<int32_t, 2> xy1{0, 0};
  array<int32_t, 2> xy2{0, 0};
  bool has_xy2{false};  // If false, this is a segment of length 1.
};

// For each finite pixel in the front and back input depth images, find the starting and ending image coordinates from the target camera.
void EpipolarLineSegmentCoordinates(const Image<float> &front_depth, const Image<float> &back_depth,
                                    const Camera &source_camera, const Camera &target_camera,
                                    unsigned int target_height, unsigned int target_width, Image<unique_ptr<XYLineSegment>> *epipolar_mapping) {
  Points3d front_depth_pcl, back_depth_pcl;
  Points2i source_xy_front, source_xy_back;
  PclFromDepthInWorldCoords(front_depth, source_camera, &source_xy_front, &front_depth_pcl);
  PclFromDepthInWorldCoords(back_depth, source_camera, &source_xy_back, &back_depth_pcl);
  Ensures(front_depth_pcl.cols() >= back_depth_pcl.cols());
  Ensures(front_depth_pcl.cols() == source_xy_front.cols());
  Ensures(back_depth_pcl.cols() == source_xy_back.cols());

  Points2i target_xy_front, target_xy_back;
  target_camera.WorldToImage(front_depth_pcl, target_height, target_width, &target_xy_front, nullptr);
  target_camera.WorldToImage(back_depth_pcl, target_height, target_width, &target_xy_back, nullptr);
  Ensures(target_xy_front.cols() >= target_xy_back.cols());
  Ensures(source_xy_front.cols() == target_xy_front.cols());
  Ensures(source_xy_back.cols() == target_xy_back.cols());

  epipolar_mapping->Resize(front_depth.height(), front_depth.width());

  for (int i = 0; i < source_xy_front.cols(); ++i) {
    unsigned int x = source_xy_front.col(i)[0];
    unsigned int y = source_xy_front.col(i)[1];
    Expects(epipolar_mapping->at(y, x) == nullptr);
    auto segment = make_unique<XYLineSegment>();
    segment->xy1 = {static_cast<int32_t>(target_xy_front.col(i)[0]), static_cast<int32_t>(target_xy_front.col(i)[1])};
    epipolar_mapping->at(y, x) = move(segment);
  }

  for (int i = 0; i < source_xy_back.cols(); ++i) {
    unsigned int x = source_xy_back.col(i)[0];
    unsigned int y = source_xy_back.col(i)[1];
    Expects(epipolar_mapping->at(y, x) != nullptr);
    auto &segment = epipolar_mapping->at(y, x);
    Expects(!segment->has_xy2);
    segment->xy2 = {static_cast<int32_t>(target_xy_back.col(i)[0]), static_cast<int32_t>(target_xy_back.col(i)[1])};
    segment->has_xy2 = true;
  }
}
}

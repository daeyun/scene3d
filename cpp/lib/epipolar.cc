#include "epipolar.h"
#include "benchmark.h"

namespace scene3d {

void EpipolarLineSegmentCoordinates(const Image<float> &front_depth,
                                    const Image<float> &back_depth,
                                    const Camera &source_camera,
                                    const Camera &target_camera,
                                    unsigned int target_height,
                                    unsigned int target_width,
                                    Image<unique_ptr<XYLineSegment>> *epipolar_mapping) {
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

void LineCoordinates(int x1, int y1, int x2, int y2, vector<array<int, 2>> *xy) {
  // Bresenham's line algorithm
  // based on http://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#C.2B.2B
  const bool is_steep = (std::abs(y2 - y1) > std::abs(x2 - x1));
  if (is_steep) {
    std::swap(x1, y1);
    std::swap(x2, y2);
  }

  const bool is_reverse = x1 > x2;
  if (is_reverse) {
    std::swap(x1, x2);
    std::swap(y1, y2);
  }

  const double dx = x2 - x1;
  const double dy = std::abs(y2 - y1);

  double error = dx * 0.5;
  const int ystep = (y1 < y2) ? 1 : -1;
  int y = y1;

  const int max_x = x2;

  for (int x = x1; x <= max_x; ++x) {  // Inclusive x2.
    if (is_steep) {
      xy->push_back(array<int, 2>{y, x});
    } else {
      xy->push_back(array<int, 2>{x, y});
    }

    error -= dy;
    if (error < 0) {
      y += ystep;
      error += dx;
    }
  }

  if (is_reverse) {
    std::reverse(xy->begin(), xy->end());
  }
}

void LineCoordinatesValidRange(int x1, int y1, int x2, int y2, int height, int width, vector<array<int, 2>> *xy) {
  // Bresenham's line algorithm
  // based on http://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#C.2B.2B
  const bool is_steep = (std::abs(y2 - y1) > std::abs(x2 - x1));
  if (is_steep) {
    std::swap(x1, y1);
    std::swap(x2, y2);
  }

  const bool is_reverse = x1 > x2;
  if (is_reverse) {
    std::swap(x1, x2);
    std::swap(y1, y2);
  }

  const double dx = x2 - x1;
  const double dy = std::abs(y2 - y1);

  double error = dx * 0.5;
  const int ystep = (y1 < y2) ? 1 : -1;
  int y = y1;

  const int max_x = x2;

  for (int x = x1; x <= max_x; ++x) {  // Inclusive x2.
    if (x >= 0 && y >= 0 && x < width && y < height) {
      if (is_steep) {
        xy->push_back(array<int, 2>{y, x});
      } else {
        xy->push_back(array<int, 2>{x, y});
      }
    }

    error -= dy;
    if (error < 0) {
      y += ystep;
      error += dx;
    }
  }

  if (is_reverse) {
    std::reverse(xy->begin(), xy->end());
  }
}

void EpipolarFeatureTransform(const float *feature_map_data,
                              const float *front_depth_data,
                              const float *back_depth_data,
                              uint32_t source_height,
                              uint32_t source_width,
                              uint32_t source_channels,
                              const char *camera_filename,
                              uint32_t target_height,
                              uint32_t target_width,
                              std::vector<float> *transformed) {
  const string camera_filename_str(camera_filename);
  LOGGER->debug("Camera filename: {}", camera_filename_str);
  LOGGER->debug("source dimension: ({}, {}, {})", source_height, source_width, source_channels);
  LOGGER->debug("target dimension: ({}, {}, {})", target_height, target_width, source_channels);

  Expects(source_height < 4096);
  Expects(source_width < 4096);
  Expects(source_channels < 409600);
  Expects(target_height < 4096);
  Expects(target_width < 4096);

  Image<float> front_depth(front_depth_data, source_height, source_width, NAN);
  Image<float> back_depth(back_depth_data, source_height, source_width, NAN);

  // Assume the file contains two cameras, in the order of source and target.
  vector<unique_ptr<scene3d::Camera>> cameras;
  ReadCameras(camera_filename_str, &cameras);
  Expects(cameras.size() == 2);
  Expects(cameras[0] && cameras[1]);
  Expects(cameras[0]->is_perspective());  // Sanity check. This is not a strict requirement.
  Expects(!cameras[1]->is_perspective());

  Image<unique_ptr<XYLineSegment>> epipolar_mapping;
  EpipolarLineSegmentCoordinates(front_depth, back_depth, *cameras[0], *cameras[1], target_height, target_width, &epipolar_mapping);

  MultiLayerImage<array<int, 2>> reverse_mapping(target_height, target_width, array<int, 2>{0, 0});

  for (int y = 0; y < source_height; ++y) {
    for (int x = 0; x < source_width; ++x) {
      const auto &segment = epipolar_mapping.at(y, x);
      if (segment == nullptr) {
        continue;
      }
      vector<array<int, 2>> xys;
      if (segment->has_xy2) {
        const auto &xy1 = segment->xy1;
        const auto &xy2 = segment->has_xy2 ? segment->xy2 : segment->xy1;
        LineCoordinatesValidRange(xy1[0], xy1[1], xy2[0], xy2[1], target_height, target_width, &xys);
      } else {
        if (segment->xy1[0] >= 0 && segment->xy1[1] >= 0 && segment->xy1[0] < target_width && segment->xy1[1] < target_height) {
          xys.push_back(array<int, 2>{segment->xy1[0], segment->xy1[1]});
        }
      }

      for (const auto &target_xy: xys) {
        reverse_mapping.values((unsigned int) target_xy[1], (unsigned int) target_xy[0])->push_back(array<int, 2>{x, y});
      }
    }
  }

  const uint32_t num_channels = source_channels;
  transformed->resize(target_height * target_width * num_channels);
  std::fill(transformed->begin(), transformed->end(), 0);

  for (int y = 0; y < target_height; ++y) {
    for (int x = 0; x < target_width; ++x) {
      const auto *source_xys = reverse_mapping.values((unsigned int) y, (unsigned int) x);
      if (source_xys == nullptr || source_xys->empty()) {
        continue;
      }

      float *target_vector_start = &transformed->at((y * target_width + x) * num_channels);
      for (const auto &source_xy : *source_xys) {
        const float *source_vector_start = feature_map_data + (source_xy[1] * source_width + source_xy[0]) * num_channels;
        std::transform(target_vector_start, target_vector_start + num_channels, source_vector_start, target_vector_start, std::plus<>());
      }

      const auto f = static_cast<float>(1.0 / source_xys->size());
      std::transform(target_vector_start, target_vector_start + num_channels, target_vector_start, [f](float value) {
        return value * f;
      });

    }
  }
}
}
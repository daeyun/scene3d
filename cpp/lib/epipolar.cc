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
  Points1d target_depth_value_front, target_depth_value_back;  // This may not be needed, depending on the gating function.
  target_camera.WorldToImage(front_depth_pcl, target_height, target_width, &target_xy_front, &target_depth_value_front);
  target_camera.WorldToImage(back_depth_pcl, target_height, target_width, &target_xy_back, &target_depth_value_back);
  Ensures(target_xy_front.cols() >= target_xy_back.cols());
  Ensures(source_xy_front.cols() == target_xy_front.cols());
  Ensures(source_xy_back.cols() == target_xy_back.cols());
  Ensures(target_depth_value_front.cols() == target_xy_front.cols());
  Ensures(target_depth_value_back.cols() == target_xy_back.cols());

  epipolar_mapping->Resize(front_depth.height(), front_depth.width());

  for (int i = 0; i < source_xy_front.cols(); ++i) {
    unsigned int x = source_xy_front.col(i)[0];
    unsigned int y = source_xy_front.col(i)[1];
    Expects(epipolar_mapping->at(y, x) == nullptr);
    auto segment = make_unique<XYLineSegment>();
    segment->xy1 = {static_cast<int32_t>(target_xy_front.col(i)[0]), static_cast<int32_t>(target_xy_front.col(i)[1])};
    segment->depth1 = target_depth_value_front[i];
    epipolar_mapping->at(y, x) = move(segment);
  }

  for (int i = 0; i < source_xy_back.cols(); ++i) {
    unsigned int x = source_xy_back.col(i)[0];
    unsigned int y = source_xy_back.col(i)[1];
    Expects(epipolar_mapping->at(y, x) != nullptr);
    auto &segment = epipolar_mapping->at(y, x);
    Expects(!segment->has_xy2);
    segment->xy2 = {static_cast<int32_t>(target_xy_back.col(i)[0]), static_cast<int32_t>(target_xy_back.col(i)[1])};
    segment->depth2 = target_depth_value_back[i];
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

void LineCoordinatesWithDepthValidRange(int x1, int y1, float d1, int x2, int y2, float d2, int height, int width, vector<array<int, 2>> *xy, vector<float> *depths) {
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
  const double depth_step = ((max_x - x1) > 0) ? (static_cast<double>(d2 - d1) / (max_x - x1)) : 0;
  double current_depth = d1;

  for (int x = x1; x <= max_x; ++x) {  // Inclusive x2.
    if (x >= 0 && y >= 0 && x < width && y < height) {
      if (is_steep) {
        xy->push_back(array<int, 2>{y, x});
      } else {
        xy->push_back(array<int, 2>{x, y});
      }
      depths->push_back(static_cast<float>(current_depth));
    }

    error -= dy;
    if (error < 0) {
      y += ystep;
      error += dx;
    }

    current_depth += depth_step;  // This is usually prone to floating point errors. But for our purposes, it is sufficient.
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
                              std::vector<float> *transformed,
                              GatingFunction gating_function) {
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

  switch (gating_function) {
    case GatingFunction::Average: {
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
      break;
    }
    case GatingFunction::ZBuffering: {
      MultiLayerImage<array<int, 2>> reverse_mapping(target_height, target_width, array<int, 2>{0, 0});
      Image<float> closest_depth(target_height, target_width, 0);
      closest_depth.Fill(std::numeric_limits<float>::infinity());

      for (int y = 0; y < source_height; ++y) {
        for (int x = 0; x < source_width; ++x) {
          const auto &segment = epipolar_mapping.at(y, x);
          if (segment == nullptr) {
            continue;
          }
          vector<array<int, 2>> xys;
          vector<float> depths;
          if (segment->has_xy2) {
            const auto &xy1 = segment->xy1;
            const auto &xy2 = segment->has_xy2 ? segment->xy2 : segment->xy1;
            const auto &d1 = segment->depth1;
            const auto &d2 = segment->has_xy2 ? segment->depth2 : segment->depth1;
            LineCoordinatesWithDepthValidRange(xy1[0], xy1[1], d1, xy2[0], xy2[1], d2, target_height, target_width, &xys, &depths);
          } else {
            if (segment->xy1[0] >= 0 && segment->xy1[1] >= 0 && segment->xy1[0] < target_width && segment->xy1[1] < target_height) {
              xys.push_back(array<int, 2>{segment->xy1[0], segment->xy1[1]});
              depths.push_back(segment->depth1);
            }
          }
          Ensures(xys.size() == depths.size());

          for (int i = 0; i < xys.size(); ++i) {
            const auto &target_xy = xys[i];
            const auto &target_d = depths[i];
            if (target_d < closest_depth.at((unsigned int) target_xy[1], (unsigned int) target_xy[0])) {
              closest_depth.at((unsigned int) target_xy[1], (unsigned int) target_xy[0]) = target_d;
              // TODO: This can be more efficient. Use Image<array<int, 2>> instead.
              auto *values = reverse_mapping.values((unsigned int) target_xy[1], (unsigned int) target_xy[0]);
              if (values->empty()) {
                values->push_back(array<int, 2>{x, y});
              } else {
                values->at(0) = array<int, 2>{x, y};
              }
            }
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

          Ensures(source_xys->size() == 1);

          float *target_vector_start = &transformed->at((y * target_width + x) * num_channels);
          const auto &source_xy = source_xys->at(0);
          const float *source_vector_start = feature_map_data + (source_xy[1] * source_width + source_xy[0]) * num_channels;
          std::memcpy(target_vector_start, source_vector_start, sizeof(float) * num_channels);
        }
      }
      break;
    }
  }
}

uint64_t PackXY(uint32_t x, uint32_t y) {
  return static_cast<uint64_t>(x) << 32 | static_cast<uint64_t>(y);
};

void RenderDepthFromAnotherView(const float *depth_data,
                                uint32_t source_height,
                                uint32_t source_width,
                                uint32_t num_images,
                                const char *camera_filename,
                                uint32_t target_height,
                                uint32_t target_width,
                                float depth_disc_pixels,
                                std::vector<float> *transformed) {
  const string camera_filename_str(camera_filename);
  LOGGER->debug("Camera filename: {}", camera_filename_str);
  LOGGER->debug("source dimension: ({}, {}, {})", num_images, source_height, source_width);
  LOGGER->debug("target dimension: ({}, {}, {})", num_images, target_height, target_width);

  Expects(source_height < 4096);
  Expects(source_width < 4096);
  Expects(num_images < 409600);
  Expects(target_height < 4096);
  Expects(target_width < 4096);

  vector<unique_ptr<scene3d::Camera>> cameras;
  ReadCameras(camera_filename_str, &cameras);
  Expects(cameras.size() == 2);
  Expects(cameras[0] && cameras[1]);
  Expects(cameras[0]->is_perspective());  // Sanity check. This is not a strict requirement.
  Expects(!cameras[1]->is_perspective());

  for (int i = 0; i < num_images; ++i) {
    const float *data = depth_data + (i * source_height * source_width);
    Image<float> depth(data, source_height, source_width, NAN);

    Points2i xy;
    Points3d out;
    PclFromDepthInWorldCoords(depth, *cameras[0], &xy, &out);
    Expects(xy.cols() == out.cols());

    Points2i image_xy;
    Points1d cam_depth_value;
    cameras[1]->WorldToImage(out, target_height, target_width, &image_xy, &cam_depth_value);
    Expects(xy.cols() == image_xy.cols());
    Expects(out.cols() == cam_depth_value.cols());

    Image<float> out_depth(target_height, target_width, NAN);
    out_depth.Fill(NAN);

    auto assign_depth_value = [&](unsigned int x, unsigned int y, float value) {
      if ((x >= 0 && x < out_depth.width()) && (y >= 0 && y < out_depth.height())) {
        if (!std::isfinite(out_depth.at(y, x)) || out_depth.at(y, x) > value) {
          out_depth.at(y, x) = value;
        }
      }
    };

    std::unordered_map<uint64_t, size_t> indices;

    for (int j = 0; j < image_xy.cols(); ++j) {
      const unsigned int x = image_xy.col(j)[0];
      const unsigned int y = image_xy.col(j)[1];
      assign_depth_value(x, y, cam_depth_value[j]);

      const unsigned int x2 = xy.col(j)[0];
      const unsigned int y2 = xy.col(j)[1];
      indices[PackXY(x2, y2)] = static_cast<size_t>(j);
    }

    for (uint32_t y = 0; y < depth.height() - 1; ++y) {
      for (uint32_t x = 0; x < depth.width() - 1; ++x) {
        if (!std::isfinite(depth.at(y, x))) {
          continue;
        }
        const size_t start_index = indices[PackXY(x, y)];

        if (std::isfinite(depth.at(y, x + 1))) {
          const size_t end_index = indices[PackXY(x + 1, y)];
          unsigned int start_x = image_xy.col(start_index)[0];
          unsigned int start_y = image_xy.col(start_index)[1];
          float start_value = cam_depth_value.col(start_index)[1];
          unsigned int end_x = image_xy.col(end_index)[0];
          unsigned int end_y = image_xy.col(end_index)[1];
          float end_value = cam_depth_value.col(end_index)[1];

          if (std::hypot((float) start_x - (float) end_x, (float) start_y - (float) end_y) > depth_disc_pixels) {
            continue;
          }

          vector<array<int, 2>> line_xy;
          LineCoordinates(start_x, start_y, end_x, end_y, &line_xy);
          auto j_max = static_cast<const int>(line_xy.size() - 1);
          for (int j = 0; j <= j_max; ++j) {  // End is inclusive.
            float a = static_cast<float>(j) / static_cast<float>(j_max);
            auto lx = static_cast<unsigned int>(line_xy[j][0]);
            auto ly = static_cast<unsigned int>(line_xy[j][1]);
            float interpolated_value = start_value * (1 - a) + end_value * a;
            assign_depth_value(lx, ly, interpolated_value);
          }
        }
        if (std::isfinite(depth.at(y + 1, x))) {
          const size_t end_index = indices[PackXY(x, y + 1)];
          unsigned int start_x = image_xy.col(start_index)[0];
          unsigned int start_y = image_xy.col(start_index)[1];
          float start_value = cam_depth_value.col(start_index)[1];
          unsigned int end_x = image_xy.col(end_index)[0];
          unsigned int end_y = image_xy.col(end_index)[1];
          float end_value = cam_depth_value.col(end_index)[1];

          if (std::hypot((float) start_x - (float) end_x, (float) start_y - (float) end_y) > depth_disc_pixels) {
            continue;
          }

          vector<array<int, 2>> line_xy;
          LineCoordinates(start_x, start_y, end_x, end_y, &line_xy);
          auto j_max = static_cast<const int>(line_xy.size() - 1);
          for (int j = 0; j <= j_max; ++j) {  // End is inclusive.
            float a = static_cast<float>(j) / static_cast<float>(j_max);
            auto lx = static_cast<unsigned int>(line_xy[j][0]);
            auto ly = static_cast<unsigned int>(line_xy[j][1]);
            float interpolated_value = start_value * (1 - a) + end_value * a;
            assign_depth_value(lx, ly, interpolated_value);
          }
        }
      }
    }

    transformed->insert(transformed->end(), out_depth.data(), out_depth.data() + out_depth.size());
  }

}
void FrustumVisibilityMapFromOverheadView(const char *camera_filename,
                                          uint32_t target_height,
                                          uint32_t target_width,
                                          std::vector<float> *transformed) {
  const string camera_filename_str(camera_filename);
  LOGGER->debug("Camera filename: {}", camera_filename_str);

  vector<unique_ptr<scene3d::Camera>> cameras;
  ReadCameras(camera_filename_str, &cameras);
  Expects(cameras.size() == 2);
  Expects(cameras[0] && cameras[1]);
  Expects(cameras[0]->is_perspective());
  Expects(!cameras[1]->is_perspective()); // The second camera needs to be an overhead camera.


  Image<float> out_image(target_height, target_width, 0);
  out_image.Fill(0);

  Points2i xy(2, 4);
  xy(0, 0) = 0;
  xy(1, 0) = 120;
  xy(0, 1) = 0;
  xy(1, 1) = 120;
  xy(0, 2) = 319;
  xy(1, 2) = 120;
  xy(0, 3) = 319;
  xy(1, 3) = 120;

  Points1d values(1, 4);
  values(0, 0) = 0.02;
  values(0, 1) = 25;
  values(0, 2) = 0.02;
  values(0, 3) = 25;

  Points3d out;
  cameras[0]->ImageToWorld(xy, values, 240, 320, &out);

  Points2i image_xy;
  Points1d cam_depth_value;
  cameras[1]->WorldToImage(out, target_height, target_width, &image_xy, &cam_depth_value);

  vector<array<int, 2>> line_xy_left;
  {
    unsigned int start_x = image_xy.col(0)[0];
    unsigned int start_y = image_xy.col(0)[1];
    unsigned int end_x = image_xy.col(1)[0];
    unsigned int end_y = image_xy.col(1)[1];
    LineCoordinates(start_x, start_y, end_x, end_y, &line_xy_left);
  }
  vector<array<int, 2>> line_xy_right;
  {
    unsigned int start_x = image_xy.col(2)[0];
    unsigned int start_y = image_xy.col(2)[1];
    unsigned int end_x = image_xy.col(3)[0];
    unsigned int end_y = image_xy.col(3)[1];
    LineCoordinates(start_x, start_y, end_x, end_y, &line_xy_right);
  }

  Ensures(line_xy_left.size() == line_xy_right.size());

  for (int j = 0; j < line_xy_left.size(); ++j) {
    int ly = line_xy_left[j][1];

    if (ly >= 0 && ly < out_image.height()) {
      int lx_start = std::max(line_xy_left[j][0], 0);
      int lx_end = std::min(static_cast<int>(line_xy_right[j][0]), static_cast<int>(target_width) - 1);
      for (int lx = lx_start; lx <= lx_end; ++lx) {
        if (lx >= 0 && lx < out_image.width()) {
          out_image.at(ly, lx) = 1.0;
        }
      }
    }
  }

  transformed->insert(transformed->end(), out_image.data(), out_image.data() + out_image.size());
}
}
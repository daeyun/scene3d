//
// Created by daeyun on 8/27/18.
//


#include "catch.hpp"
#include "spdlog/spdlog.h"

#include "lib/common.h"
#include "lib/camera.h"
#include "lib/file_io.h"
#include "lib/depth_render_utils.h"
#include "lib/pcl.h"
#include "lib/vectorization_utils.h"
#include "lib/epipolar.h"
#include "lib/benchmark.h"

using namespace scene3d;

TEST_CASE("simple reprojection") {
  string filename = "resources/depth_render/0004d52d1aeeb8ae6de39d6bd993e992/000003_ldi.bin";
  string overhead_filename = "resources/depth_render/0004d52d1aeeb8ae6de39d6bd993e992/000003_ldi-o.bin";
  string camera_filename = "resources/depth_render/0004d52d1aeeb8ae6de39d6bd993e992/000003_cam.txt";

  vector<int> shape;
  vector<float> ldi_data;
  ReadTensorData(filename, &shape, &ldi_data);

  REQUIRE(shape.size() == 3);
  REQUIRE(shape[0] == 4);
  REQUIRE(shape[1] == 240);
  REQUIRE(shape[2] == 320);

  float *front_layer_data = ldi_data.data();
  float *back_layer_data = ldi_data.data() + shape[1] * shape[2];

  const auto height = static_cast<unsigned int>(shape[1]);
  const auto width = static_cast<unsigned int>(shape[2]);

  Image<float> front_layer(front_layer_data, height, width, NAN);
  Image<float> back_layer(back_layer_data, height, width, NAN);

  REQUIRE(front_layer.height() == height);
  REQUIRE(front_layer.width() == width);

  vector<int> overhead_shape;
  vector<float> overhead_ldi_data;
  ReadTensorData(overhead_filename, &overhead_shape, &overhead_ldi_data);

  REQUIRE(overhead_shape.size() == 3);
  REQUIRE(overhead_shape[0] == 4);
  REQUIRE(overhead_shape[1] == 300);
  REQUIRE(overhead_shape[2] == 300);

  const auto overhead_height = static_cast<unsigned int>(overhead_shape[1]);
  const auto overhead_width = static_cast<unsigned int>(overhead_shape[2]);

  vector<unique_ptr<scene3d::Camera>> cameras;
  ReadCameras(camera_filename, &cameras);

  Points2i source_xy, target_xy_front, target_xy_back;

  REQUIRE(cameras[0]->is_perspective());
  REQUIRE(!cameras[1]->is_perspective());

  auto stime = TimeSinceEpoch<std::milli>();

  Image<unique_ptr<XYLineSegment>> epipolar_mapping;
  EpipolarLineSegmentCoordinates(
      front_layer,
      back_layer,
      *cameras[0],
      *cameras[1],
      overhead_height,
      overhead_width,
      &epipolar_mapping);

  LOGGER->info("Elapsed (EpipolarLineSegmentCoordinates): {}", TimeSinceEpoch<std::milli>() - stime);

  REQUIRE(epipolar_mapping.height() == front_layer.height());
  REQUIRE(epipolar_mapping.width() == front_layer.width());

  REQUIRE(epipolar_mapping.at(100, 169) != nullptr);
  REQUIRE(epipolar_mapping.at(100, 170) == nullptr);
  REQUIRE(epipolar_mapping.at(100, 171) == nullptr);
  REQUIRE(epipolar_mapping.at(100, 172) == nullptr);
  REQUIRE(epipolar_mapping.at(100, 173) == nullptr);
  REQUIRE(epipolar_mapping.at(100, 174) != nullptr);

  REQUIRE(!epipolar_mapping.at(100, 174)->has_xy2);
  REQUIRE(epipolar_mapping.at(100, 174)->xy1[1] < 0);
  REQUIRE(epipolar_mapping.at(100, 174)->xy1[0] > 100);
  REQUIRE(epipolar_mapping.at(100, 174)->xy1[0] < 250);

  REQUIRE(epipolar_mapping.at(130, 170)->has_xy2);
  REQUIRE(epipolar_mapping.at(130, 170)->has_xy2);

  unsigned int x = 150;
  unsigned int y = 150;

  Image<float> overhead_front_layer(overhead_ldi_data.data(), overhead_height, overhead_width, NAN);

  float reprojected_depth_front = overhead_front_layer.at(static_cast<unsigned int>(epipolar_mapping.at(y, x)->xy1[1]), static_cast<unsigned int>(epipolar_mapping.at(y, x)->xy1[0]));
  float reprojected_depth_back = overhead_front_layer.at(static_cast<unsigned int>(epipolar_mapping.at(y, x)->xy2[1]), static_cast<unsigned int>(epipolar_mapping.at(y, x)->xy2[0]));

  REQUIRE(reprojected_depth_front > 0.7);
  REQUIRE(reprojected_depth_front < 0.75);
  REQUIRE(reprojected_depth_back > 0.7);
  REQUIRE(reprojected_depth_back < 0.75);
}

TEST_CASE("simple reprojection 2") {
  string filename = "resources/depth_render/915a66f48ec925febf644f962375720d/000000_ldi.bin";
  string overhead_filename = "resources/depth_render/915a66f48ec925febf644f962375720d/000000_ldi-o.bin";
  string camera_filename = "resources/depth_render/915a66f48ec925febf644f962375720d/000000_cam.txt";

  vector<int> shape;
  vector<float> ldi_data;
  ReadTensorData(filename, &shape, &ldi_data);

  REQUIRE(shape.size() == 3);
  REQUIRE(shape[0] == 4);
  REQUIRE(shape[1] == 240);
  REQUIRE(shape[2] == 320);

  float *front_layer_data = ldi_data.data();
  float *back_layer_data = ldi_data.data() + shape[1] * shape[2];

  const auto height = static_cast<unsigned int>(shape[1]);
  const auto width = static_cast<unsigned int>(shape[2]);

  Image<float> front_layer(front_layer_data, height, width, NAN);
  Image<float> back_layer(back_layer_data, height, width, NAN);

  REQUIRE(front_layer.height() == height);
  REQUIRE(front_layer.width() == width);

  vector<int> overhead_shape;
  vector<float> overhead_ldi_data;
  ReadTensorData(overhead_filename, &overhead_shape, &overhead_ldi_data);

  REQUIRE(overhead_shape.size() == 3);
  REQUIRE(overhead_shape[0] == 4);
  REQUIRE(overhead_shape[1] == 300);
  REQUIRE(overhead_shape[2] == 300);

  const auto overhead_height = static_cast<unsigned int>(shape[1]);
  const auto overhead_width = static_cast<unsigned int>(shape[2]);

  vector<unique_ptr<scene3d::Camera>> cameras;
  ReadCameras(camera_filename, &cameras);

  Points2i source_xy, target_xy_front, target_xy_back;

  REQUIRE(cameras[0]->is_perspective());
  REQUIRE(!cameras[1]->is_perspective());

  Image<unique_ptr<XYLineSegment>> epipolar_mapping;
  EpipolarLineSegmentCoordinates(
      front_layer,
      back_layer,
      *cameras[0],
      *cameras[1],
      overhead_height,
      overhead_width,
      &epipolar_mapping);

  REQUIRE(epipolar_mapping.height() == front_layer.height());
  REQUIRE(epipolar_mapping.width() == front_layer.width());
}

TEST_CASE("camera test") {
  string filename = "resources/depth_render/0004d52d1aeeb8ae6de39d6bd993e992/000003_ldi.bin";
  string overhead_filename = "resources/depth_render/0004d52d1aeeb8ae6de39d6bd993e992/000003_ldi-o.bin";
  string camera_filename = "resources/depth_render/0004d52d1aeeb8ae6de39d6bd993e992/000003_cam.txt";

  vector<int> shape;
  vector<float> ldi_data;
  ReadTensorData(filename, &shape, &ldi_data);

  REQUIRE(shape.size() == 3);
  REQUIRE(shape[0] == 4);
  REQUIRE(shape[1] == 240);
  REQUIRE(shape[2] == 320);

  float *front_layer_data = ldi_data.data();
  float *back_layer_data = ldi_data.data() + shape[1] * shape[2];

  const auto height = static_cast<unsigned int>(shape[1]);
  const auto width = static_cast<unsigned int>(shape[2]);

  Image<float> front_layer(front_layer_data, height, width, NAN);
  Image<float> back_layer(back_layer_data, height, width, NAN);

  REQUIRE(front_layer.height() == height);
  REQUIRE(front_layer.width() == width);

  vector<int> overhead_shape;
  vector<float> overhead_ldi_data;
  ReadTensorData(overhead_filename, &overhead_shape, &overhead_ldi_data);

  REQUIRE(overhead_shape.size() == 3);
  REQUIRE(overhead_shape[0] == 4);
  REQUIRE(overhead_shape[1] == 300);
  REQUIRE(overhead_shape[2] == 300);

  Image<float> overhead_front_layer(overhead_ldi_data.data(), 300, 300, NAN);
  const auto overhead_height = static_cast<unsigned int>(shape[1]);
  const auto overhead_width = static_cast<unsigned int>(shape[2]);

  vector<unique_ptr<scene3d::Camera>> cameras;
  ReadCameras(camera_filename, &cameras);

  overhead_front_layer.Transform([&](size_t i, float value) -> float {
    return cameras[1]->position()[1] - value - 0.05;
  });

  PointCloud pcl;
  PclFromDepthInWorldCoords(front_layer, *cameras[0], &pcl);

  Points2i xy;
  cameras[1]->WorldToImage(pcl.points(), 300, 300, &xy);

  // Depth point cloud reprojected to overhead image space.
  Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> transposed_points = xy.transpose().cast<int>().eval();
  SerializeTensor<int>("/tmp/scene3d_test/xy.bin", transposed_points.data(), {static_cast<int>(xy.cols()), 2});
}

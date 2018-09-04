//
// Created by daeyun on 8/27/18.
//

#include "catch.hpp"
#include "spdlog/spdlog.h"

#include "lib/common.h"
#include "lib/depth_render_utils.h"
#include "lib/camera.h"
#include "lib/file_io.h"
#include "lib/suncg_utils.h"

using namespace scene3d;

TEST_CASE("perspective") {
  std::string obj_filename = "resources/depth_render/dummy.obj";
  std::string json_filename = "resources/depth_render/dummy.json";
  std::string category_filename = "resources/depth_render/dummy_ModelCategoryMapping.csv";

  // Read the obj, json, and category mappings.
  auto scene = make_unique<suncg::Scene>(json_filename, obj_filename, category_filename);
  scene->Build();

  scene3d::RayTracer ray_tracer(scene->faces, scene->vertices);
  ray_tracer.PrintStats();

  unsigned int height = 240;
  unsigned int width = 320;
  double x_fov = 30.0 / 180.0 * M_PI;  // 60 degrees left-to-right.
  double near = 0.01;
  double far = 20;
  Vec3 cam_eye = {0, 7.660254037844387, 0};  // This is the distance where the plane takes up 80% of the image width.
  Vec3 cam_up = {0, 0, -1};
  Vec3 cam_lookat = {0, 0, 0};

  auto frustum = MakePerspectiveFrustumParams(static_cast<double>(height) / width, x_fov, near, far);
  auto camera = PerspectiveCamera(cam_eye, cam_lookat, cam_up, frustum);

  auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
      &ray_tracer,
      &camera,
      width,
      height,
      scene.get()
  );

  auto ml_depth = MultiLayerImage<float>(height, width);
  auto ml_prim_ids = MultiLayerImage<uint32_t>(height, width);
  RenderMultiLayerDepthImage(&renderer, &ml_depth, &ml_prim_ids);

  REQUIRE(3 == ml_depth.values(120, 160)->size());
  REQUIRE(Approx(cam_eye[1] - 1) == ml_depth.at(120, 160, 0));
  REQUIRE(Approx(cam_eye[1] + 1) == ml_depth.at(120, 160, 1));

  Image<float> depth(height, width);
  ml_depth.ExtractLayer(0, &depth);

  REQUIRE(Approx(cam_eye[1] - 1) == depth.at(120, 160));

  // Hard-coded values based on the camera position and the mesh file.
  // It should look like this: https://daeyun.s3.amazonaws.com/images/3DPtFTTqr3k2g.png
  for (int x = 0; x < 32; ++x) {
    REQUIRE(std::isnan(depth.at(120, x)));
  }
  for (int x = 32; x < 118; ++x) {
    REQUIRE(Approx(cam_eye[1] + 1) == depth.at(120, x));
  }
  for (int x = 118; x < 118 + 84; ++x) {
    REQUIRE(Approx(cam_eye[1] - 1) == depth.at(120, x));
  }
  REQUIRE(cam_eye[1] - 1 < depth.at(120, 118 + 84));
  REQUIRE(std::isfinite(depth.at(120, width - 32 - 1)));
  for (int x = width - 32; x < width; ++x) {
    REQUIRE(std::isnan(depth.at(120, x)));
  }
  for (int y = 0; y < 78; ++y) {
    REQUIRE(Approx(cam_eye[1] + 1) == depth.at(y, 160));
  }
  for (int y = 78; y < 78 + 84; ++y) {
    REQUIRE(Approx(cam_eye[1] - 1) == depth.at(y, 160));
  }
  REQUIRE(Approx(cam_eye[1] + 1) == depth.at(78 + 84, 160));

  SerializeTensor<float>("/tmp/scene3d_test/dummy_obj_depth_00.bin", depth.data(), {static_cast<int>(height), static_cast<int>(width)});
}

TEST_CASE("orthographic") {
  std::string obj_filename = "resources/depth_render/dummy.obj";
  std::string json_filename = "resources/depth_render/dummy.json";
  std::string category_filename = "resources/depth_render/dummy_ModelCategoryMapping.csv";

  // Read the obj, json, and category mappings.
  auto scene = make_unique<suncg::Scene>(json_filename, obj_filename, category_filename);
  scene->Build();

  scene3d::RayTracer ray_tracer(scene->faces, scene->vertices);
  ray_tracer.PrintStats();

  unsigned int height = 240;
  unsigned int width = 320;
  double near = 0.01;
  double far = 20;
  Vec3 cam_eye = {0, 6, 0};
  Vec3 cam_up = {0, 0, -1};
  Vec3 cam_lookat = {0, 0, 0};

  FrustumParams frustum;

  frustum.near = near;
  frustum.far = far;
  frustum.right = 5;
  frustum.left = -frustum.right;
  frustum.top = frustum.right * (static_cast<double>(height) / width);
  frustum.bottom = -frustum.top;

  auto camera = OrthographicCamera(cam_eye, cam_lookat, cam_up, frustum);

  auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
      &ray_tracer,
      &camera,
      width,
      height,
      scene.get()
  );

  auto ml_depth = MultiLayerImage<float>(height, width);
  auto ml_prim_ids = MultiLayerImage<uint32_t>(height, width);
  RenderMultiLayerDepthImage(&renderer, &ml_depth, &ml_prim_ids);

  REQUIRE(3 == ml_depth.values(120, 160)->size());
  REQUIRE(Approx(cam_eye[1] - 1) == ml_depth.at(120, 160, 0));
  REQUIRE(Approx(cam_eye[1] + 1) == ml_depth.at(120, 160, 1));

  Image<float> depth(height, width);
  ml_depth.ExtractLayer(0, &depth);

  REQUIRE(Approx(cam_eye[1] - 1) == depth.at(120, 160));

  // Hard-coded values based on the camera position and the mesh file.
  // It should look like this: https://daeyun.s3.amazonaws.com/images/3DPtb3ccFBzpV.png
  for (int x = 0; x < 32; ++x) {
    REQUIRE(std::isnan(depth.at(120, x)));
  }
  for (int x = 32; x < 32 * 4; ++x) {
    REQUIRE(Approx(cam_eye[1] + 1) == depth.at(120, x));
  }
  for (int x = 32 * 4; x < 32 * 6; ++x) {
    REQUIRE(Approx(cam_eye[1] - 1) == depth.at(120, x));
  }
  REQUIRE(cam_eye[1] - 1 < depth.at(120, 32 * 6));
  REQUIRE(std::isfinite(depth.at(120, 32 * 9 - 1)));
  for (int x = 32 * 9; x < width; ++x) {
    REQUIRE(std::isnan(depth.at(120, x)));
  }
  for (int y = 0; y < 88; ++y) {
    REQUIRE(Approx(cam_eye[1] + 1) == depth.at(y, 160));
  }
  for (int y = 88; y < 88 + 32 * 2; ++y) {
    REQUIRE(Approx(cam_eye[1] - 1) == depth.at(y, 160));
  }
  REQUIRE(Approx(cam_eye[1] + 1) == depth.at(88 + 32 * 2, 160));

  SerializeTensor<float>("/tmp/scene3d_test/dummy_obj_depth_01.bin", depth.data(), {static_cast<int>(height), static_cast<int>(width)});
}


#include <fstream>
#include "catch.hpp"
#include "spdlog/spdlog.h"

#include "lib/common.h"
#include "lib/multi_layer_depth_renderer.h"
#include "lib/file_io.h"
#include "lib/suncg_utils.h"

using namespace scene3d;

TEST_CASE("multi hit test") {
  // These paths are relative to the project root directory.
  std::string camera_filename = "resources/depth_render/house_obj_camera.txt";
  std::string obj_filename = "resources/house/0004d52d1aeeb8ae6de39d6bd993e992/house.obj";
  std::string json_filename = "resources/house/0004d52d1aeeb8ae6de39d6bd993e992/house_p.json";
  std::string category_filename = "resources/ModelCategoryMapping.csv";

  // Read the obj, json, and category mappings.
  auto scene = make_unique<suncg::Scene>(json_filename, obj_filename, category_filename);
  scene->Build();

  std::vector<suncg::CameraParams> suncg_cameras;
  suncg::ReadCameraFile(camera_filename, &suncg_cameras);

  scene3d::RayTracer ray_tracer(scene->faces, scene->vertices);
  ray_tracer.PrintStats();

  unsigned int width = 320;
  unsigned int height = 240;

  int camera_index = 3;

  double near = 0.01;
  double far = 100;
  PerspectiveCamera camera = MakeCamera(suncg_cameras[camera_index], near, far);

  auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
      &ray_tracer,
      &camera,
      width,
      height,
      scene.get()
  );

  renderer.ray_tracer()->PrintStats();

  SECTION("Floor pixel") {
    vector<float> values;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(100, 200, &values, &prim_ids);

    LOGGER->info("Num hits at (100, 200): {}", values.size());
    for (int i = 0; i < values.size(); ++i) {
      LOGGER->info("{}", values[i]);
    }

    REQUIRE(0 == background_index);
    REQUIRE(1 == values.size());
    REQUIRE(Approx(2.93651) == values[0]);
  }

  SECTION("no hit. outdoor") {
    vector<float> values;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(172, 109, &values, &prim_ids);

    REQUIRE(0 > background_index);
    REQUIRE(0 == values.size());
  }

  SECTION("two hits. no background hit") {
    vector<float> values;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(172, 129, &values, &prim_ids);

    REQUIRE(0 > background_index);
    REQUIRE(2 == values.size());
  }

  SECTION("one hit. background (door)") {
    vector<float> values;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(156, 109, &values, &prim_ids);

    REQUIRE(0 == background_index);
    REQUIRE(1 == values.size());
    REQUIRE(Approx(4.9071) == values[0]);
  }

  SECTION("three hits, two hits through arm of sofa, then background (floor)") {
    vector<float> values;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(83, 215, &values, &prim_ids);

    REQUIRE(2 == background_index);
    REQUIRE(3 == values.size());
    REQUIRE(Approx(2.63747) == values[2]);
  }

  SECTION("many hits. two objects, last of which has a coinciding plane on the floor. then background (floor)") {
    vector<float> values;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(214, 170, &values, &prim_ids);

    LOGGER->info("Num hits at (214, 170): {}", values.size());
    for (int i = 0; i < values.size(); ++i) {
      LOGGER->info("{}", values[i]);

      Vec3 origin = renderer.RayOrigin(0, 0);  // x,y doesn't matter because this is perspective.
      Vec3 dir = renderer.RayDirection(214, 170);

      Vec3 p = origin + values[i] * dir;
      LOGGER->info("{}, {}, {}", p[0], p[1], p[2]);
    }

    REQUIRE(0 < background_index);
    REQUIRE(4 < values.size());
    REQUIRE(Approx(values[values.size() - 2]).margin(0.0025) == values[values.size() - 1]);

    Vec3 origin = renderer.RayOrigin(0, 0);
    Vec3 dir = renderer.RayDirection(214, 170);

    double height_1 = (origin + values[values.size() - 1] * dir)[1];
    double height_2 = (origin + values[values.size() - 2] * dir)[1];
    double height_3 = (origin + values[values.size() - 3] * dir)[1];

    REQUIRE(Approx(height_1).margin(0.001) == height_2);
    REQUIRE(Approx(0.0171942492).margin(1e-5) == height_3 - height_2);  // Height of the object on the floor.
  }
}

TEST_CASE("thickness in inner normal direction") {
  // These paths are relative to the project root directory.
  std::string camera_filename = "resources/depth_render/house_obj_camera.txt";
  std::string obj_filename = "resources/house/0004d52d1aeeb8ae6de39d6bd993e992/house.obj";
  std::string json_filename = "resources/house/0004d52d1aeeb8ae6de39d6bd993e992/house_p.json";
  std::string category_filename = "resources/ModelCategoryMapping.csv";

  // Read the obj, json, and category mappings.
  auto scene = make_unique<suncg::Scene>(json_filename, obj_filename, category_filename);
  scene->Build();

  std::vector<suncg::CameraParams> suncg_cameras;
  suncg::ReadCameraFile(camera_filename, &suncg_cameras);

  scene3d::RayTracer ray_tracer(scene->faces, scene->vertices);
  ray_tracer.PrintStats();

  unsigned int width = 320;
  unsigned int height = 240;

  int camera_index = 3;

  double near = 0.01;
  double far = 100;
  PerspectiveCamera camera = MakeCamera(suncg_cameras[camera_index], near, far);

  auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
      &ray_tracer,
      &camera,
      width,
      height,
      scene.get()
  );

  renderer.ray_tracer()->PrintStats();

  SECTION("Top of coffee table") {
    {
      vector<float> values;
      vector<uint32_t> prim_ids;
      renderer.ObjectCenteredRayDisplacement(150, 200, &values, &prim_ids);
      REQUIRE(values.size() > 2);
      float t = values[1] - values[0];
      LOGGER->info("First thickness value at (150, 200): {}", t);
      REQUIRE(t > 0);
      REQUIRE(Approx(0.03) == t);
    }

    {
      vector<float> values;
      vector<uint32_t> prim_ids;
      renderer.ObjectCenteredRayDisplacement(205, 187, &values, &prim_ids);
      REQUIRE(values.size() > 2);
      float t = values[1] - values[0];
      LOGGER->info("First thickness value at (205, 187): {}", t);
      REQUIRE(t > 0);
      REQUIRE(Approx(0.03) == t);
    }
  }
}

TEST_CASE("orthographic coordinates") {
  std::vector<std::array<unsigned int, 3>> faces{
      {0, 1, 2},
  };
  std::vector<std::array<float, 3>> vertices{
      {0, 0, 0},
      {0, 1, 0},
      {0, 0, 1},
  };
  std::vector<std::string> prim_id_to_node_name;
  scene3d::RayTracer ray_tracer(faces, vertices);

  {
    FrustumParams frustum;
    frustum.left = -8;
    frustum.right = 8;
    frustum.top = 16;
    frustum.bottom = -16;
    frustum.near = 0.01;
    frustum.far = 100;

    Vec3 eye = {2, 3, 1};
    Vec3 view_dir = {0, 0, -1};
    Vec3 up = {0, 1, 0};

    auto camera = OrthographicCamera(eye, eye + view_dir, up, frustum);

    auto renderer = scene3d::SimpleMultiLayerDepthRenderer(
        &ray_tracer,
        &camera,
        16,
        32
    );

    Vec3 ray_origin = renderer.RayOrigin(0, 0);
    const Vec3 expected_ray_dir = Vec3{-7.5 + 2, 15.5 + 3, 1};

    REQUIRE(Approx(expected_ray_dir[0]) == ray_origin[0]);
    REQUIRE(Approx(expected_ray_dir[1]) == ray_origin[1]);
    REQUIRE(Approx(expected_ray_dir[2]) == ray_origin[2]);
  }

}

TEST_CASE("simple scene depth test") {
  // These paths are relative to the project root directory.
  std::string obj_filename = "resources/depth_render/dummy.obj";
  std::string json_filename = "resources/depth_render/dummy.json";
  std::string category_filename = "resources/depth_render/dummy_ModelCategoryMapping.csv";

  // Read the obj, json, and category mappings.
  auto scene = make_unique<suncg::Scene>(json_filename, obj_filename, category_filename);
  scene->Build();

  scene3d::RayTracer ray_tracer(scene->faces, scene->vertices);
  ray_tracer.PrintStats();

  Vec3 eye = {0, 5, 0};
  Vec3 up = {0, 0, -1};
  Vec3 view_dir = {0, -1, 0};
  double x_fov = 60 / 180.0 * M_PI;  // 120

  auto frustum = MakePerspectiveFrustumParams(1, x_fov, 0.01, 100);
  auto camera = PerspectiveCamera(eye, eye + view_dir, up, frustum);

  LOGGER->info("Reading file {}", obj_filename);

  unsigned int width = 200;
  unsigned int height = 200;

  auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
      &ray_tracer,
      &camera,
      width,
      height,
      scene.get()
  );

  renderer.ray_tracer()->PrintStats();

  {
    vector<float> values;
    vector<unsigned int> prim_ids;
    renderer.DepthValues(100, 100, &values, &prim_ids);
    REQUIRE(3 == values.size());
    REQUIRE(Approx(4.0) == values[0]);
    REQUIRE(Approx(6.0) == values[1]);
  }

  {
    vector<float> values;
    vector<unsigned int> prim_ids;
    renderer.DepthValues(95, 107, &values, &prim_ids);
    REQUIRE(3 == values.size());
    REQUIRE(Approx(4.0) == values[0]);
    REQUIRE(Approx(6.0) == values[1]);
  }

  {
    vector<float> values;
    vector<unsigned int> prim_ids;
    renderer.DepthValues(70, 72, &values, &prim_ids);
    REQUIRE(1 == values.size());
    REQUIRE(Approx(6.0) == values[0]);
  }

  {
    vector<float> values;
    vector<unsigned int> prim_ids;
    renderer.DepthValues(80, 82, &values, &prim_ids);
    REQUIRE(1 == values.size());
    REQUIRE(Approx(6.0) == values[0]);
  }

  {
    vector<float> values;
    vector<unsigned int> prim_ids;
    renderer.DepthValues(121, 79, &values, &prim_ids);
    REQUIRE(1 == values.size());
    REQUIRE(Approx(6.0) == values[0]);
  }
}

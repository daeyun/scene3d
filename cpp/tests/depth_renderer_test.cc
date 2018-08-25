#include <fstream>
#include "catch.hpp"
#include "spdlog/spdlog.h"

#include "lib/common.h"
#include "lib/multi_layer_depth_renderer.h"
#include "lib/file_io.h"

struct CameraParams {
  Vec3 cam_eye;
  Vec3 cam_view_dir;
  Vec3 cam_up;
  double x_fov;
  double y_fov;
  double score;  // scene coverage score. not used at the moment.
};


TEST_CASE("multi hit test") {
  // These paths are relative to the project root directory.
  std::string camera_filename = "resources/depth_render/house_obj_camera.txt";
  std::string obj_filename = "resources/depth_render/house.obj";

  std::vector<std::array<unsigned int, 3>> faces;
  std::vector<std::array<float, 3>> vertices;
  std::vector<int> prim_id_to_node_id;
  std::vector<std::string> prim_id_to_node_name;

  LOGGER->info("Reading file {}", camera_filename);

  std::ifstream source;
  source.open(camera_filename, std::ios_base::in);
  if (!source) {
    throw std::runtime_error("Can't open file.");
  }

  std::vector<CameraParams> suncg_cameras;
  for (std::string line; std::getline(source, line);) {
    if (line.empty()) {
      continue;
    }

    std::istringstream in(line);
    CameraParams cam;

    in >> cam.cam_eye[0] >> cam.cam_eye[1] >> cam.cam_eye[2];
    in >> cam.cam_view_dir[0] >> cam.cam_view_dir[1] >> cam.cam_view_dir[2];
    in >> cam.cam_up[0] >> cam.cam_up[1] >> cam.cam_up[2];
    in >> cam.x_fov >> cam.y_fov >> cam.score;

    LOGGER->info("camera {}, eye {}, {}, {}, fov {}, {}", suncg_cameras.size(), cam.cam_eye[0], cam.cam_eye[1], cam.cam_eye[2], cam.x_fov, cam.y_fov);

    cam.cam_view_dir.normalize();

    suncg_cameras.push_back(cam);
  }

  LOGGER->info("Reading file {}", obj_filename);

  bool ok = scene3d::ReadFacesAndVertices(obj_filename, &faces, &vertices, &prim_id_to_node_id, &prim_id_to_node_name);

  // Sanity check.
  Ensures(faces.size() == prim_id_to_node_id.size());
  Ensures(faces.size() == prim_id_to_node_name.size());

  LOGGER->info("{} faces, {} vertices", faces.size(), vertices.size());

  scene3d::RayTracer ray_tracer(faces, vertices);

  unsigned int width = 320;
  unsigned int height = 240;
  unsigned int max_hits = 0;  // unlimited

  int camera_index = 3;

  auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
      &ray_tracer,
      suncg_cameras[camera_index].cam_eye,
      suncg_cameras[camera_index].cam_view_dir,
      suncg_cameras[camera_index].cam_up,
      suncg_cameras[camera_index].x_fov,
      suncg_cameras[camera_index].y_fov,
      width,
      height,
      max_hits,
      prim_id_to_node_name,
      false,
      0, 0, 0, 0
  );

  renderer.ray_tracer()->PrintStats();

  SECTION("Floor pixel") {
    vector<float> values;
    vector<string> model_ids;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(100, 200, &values, &model_ids, &prim_ids);

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
    vector<string> model_ids;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(172, 109, &values, &model_ids, &prim_ids);

    REQUIRE(0 > background_index);
    REQUIRE(0 == values.size());
  }

  SECTION("two hits. no background hit") {
    vector<float> values;
    vector<string> model_ids;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(172, 129, &values, &model_ids, &prim_ids);

    REQUIRE(0 > background_index);
    REQUIRE(2 == values.size());
  }

  SECTION("one hit. background (door)") {
    vector<float> values;
    vector<string> model_ids;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(156, 109, &values, &model_ids, &prim_ids);

    REQUIRE(0 == background_index);
    REQUIRE(1 == values.size());
    REQUIRE(Approx(4.9071) == values[0]);
  }

  SECTION("three hits, two hits through arm of sofa, then background (floor)") {
    vector<float> values;
    vector<string> model_ids;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(83, 215, &values, &model_ids, &prim_ids);

    REQUIRE(2 == background_index);
    REQUIRE(3 == values.size());
    REQUIRE(Approx(2.63747) == values[2]);
  }

  SECTION("many hits. two objects, last of which has a coinciding plane on the floor. then background (floor)") {
    vector<float> values;
    vector<string> model_ids;
    vector<unsigned int> prim_ids;
    int background_index = renderer.DepthValues(214, 170, &values, &model_ids, &prim_ids);

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
  std::string obj_filename = "resources/depth_render/house.obj";

  std::vector<std::array<unsigned int, 3>> faces;
  std::vector<std::array<float, 3>> vertices;
  std::vector<int> prim_id_to_node_id;
  std::vector<std::string> prim_id_to_node_name;

  LOGGER->info("Reading file {}", camera_filename);

  std::ifstream source;
  source.open(camera_filename, std::ios_base::in);
  if (!source) {
    throw std::runtime_error("Can't open file.");
  }

  std::vector<CameraParams> suncg_cameras;
  for (std::string line; std::getline(source, line);) {
    if (line.empty()) {
      continue;
    }

    std::istringstream in(line);
    CameraParams cam;

    in >> cam.cam_eye[0] >> cam.cam_eye[1] >> cam.cam_eye[2];
    in >> cam.cam_view_dir[0] >> cam.cam_view_dir[1] >> cam.cam_view_dir[2];
    in >> cam.cam_up[0] >> cam.cam_up[1] >> cam.cam_up[2];
    in >> cam.x_fov >> cam.y_fov >> cam.score;

    LOGGER->info("camera {}, eye {}, {}, {}, fov {}, {}", suncg_cameras.size(), cam.cam_eye[0], cam.cam_eye[1], cam.cam_eye[2], cam.x_fov, cam.y_fov);

    cam.cam_view_dir.normalize();

    suncg_cameras.push_back(cam);
  }

  LOGGER->info("Reading file {}", obj_filename);

  bool ok = scene3d::ReadFacesAndVertices(obj_filename, &faces, &vertices, &prim_id_to_node_id, &prim_id_to_node_name);

  // Sanity check.
  Ensures(faces.size() == prim_id_to_node_id.size());
  Ensures(faces.size() == prim_id_to_node_name.size());

  LOGGER->info("{} faces, {} vertices", faces.size(), vertices.size());

  scene3d::RayTracer ray_tracer(faces, vertices);

  unsigned int width = 320;
  unsigned int height = 240;
  unsigned int max_hits = 0;  // unlimited. not relevant at the moment.

  int camera_index = 3;

  auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
      &ray_tracer,
      suncg_cameras[camera_index].cam_eye,
      suncg_cameras[camera_index].cam_view_dir,
      suncg_cameras[camera_index].cam_up,
      suncg_cameras[camera_index].x_fov,
      suncg_cameras[camera_index].y_fov,
      width,
      height,
      max_hits,
      prim_id_to_node_name,
      false,
      0, 0, 0, 0
  );

  renderer.ray_tracer()->PrintStats();

  float t = 0;
  SECTION("Top of coffee table") {
    t = renderer.ObjectCenteredVolume(150, 200);
    LOGGER->info("First thickness value at (150, 200): {}", t);
    REQUIRE(t > 0);
    REQUIRE(Approx(0.03) == t);

    t = renderer.ObjectCenteredVolume(205, 187);
    LOGGER->info("First thickness value at (205, 187): {}", t);
    REQUIRE(t > 0);
    REQUIRE(Approx(0.03) == t);
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
    CameraParams cam;
    cam.cam_eye = {2, 3, 1};
    cam.cam_view_dir = {0, 0, -1};
    cam.cam_up = {0, 1, 0};

    auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
        &ray_tracer,
        cam.cam_eye,
        cam.cam_view_dir,
        cam.cam_up,
        cam.x_fov,
        cam.y_fov,
        16,
        32,
        0,
        prim_id_to_node_name,
        true,
        -8, 8, 16, -16
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

  std::vector<std::array<unsigned int, 3>> faces;
  std::vector<std::array<float, 3>> vertices;
  std::vector<int> prim_id_to_node_id;
  std::vector<std::string> prim_id_to_node_name;

  CameraParams cam_params;
  cam_params.cam_eye = {0, 5, 0};
  cam_params.cam_up = {0, 0, -1};
  cam_params.cam_view_dir = {0, -1, 0};
  cam_params.x_fov = 60 / 180.0 * M_PI;
  cam_params.y_fov = 60 / 180.0 * M_PI;

  LOGGER->info("Reading file {}", obj_filename);

  bool ok = scene3d::ReadFacesAndVertices(obj_filename, &faces, &vertices, &prim_id_to_node_id, &prim_id_to_node_name);

  // Sanity check.
  Ensures(faces.size() == prim_id_to_node_id.size());
  Ensures(faces.size() == prim_id_to_node_name.size());

  LOGGER->info("{} faces, {} vertices", faces.size(), vertices.size());

  scene3d::RayTracer ray_tracer(faces, vertices);

  unsigned int width = 200;
  unsigned int height = 200;
  unsigned int max_hits = 0;  // unlimited. not relevant at the moment.

  auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
      &ray_tracer,
      cam_params.cam_eye,
      cam_params.cam_view_dir,
      cam_params.cam_up,
      cam_params.x_fov,
      cam_params.y_fov,
      width,
      height,
      max_hits,
      prim_id_to_node_name,
      false,
      0, 0, 0, 0
  );

  renderer.ray_tracer()->PrintStats();

  {
    vector<float> values;
    vector<string> model_ids;
    vector<unsigned int> prim_ids;
    renderer.DepthValues(100, 100, &values, &model_ids, &prim_ids);
    REQUIRE(3 == values.size());
    REQUIRE(Approx(4.0) == values[0]);
    REQUIRE(Approx(6.0) == values[1]);
  }

  {
    vector<float> values;
    vector<string> model_ids;
    vector<unsigned int> prim_ids;
    renderer.DepthValues(95, 107, &values, &model_ids, &prim_ids);
    REQUIRE(3 == values.size());
    REQUIRE(Approx(4.0) == values[0]);
    REQUIRE(Approx(6.0) == values[1]);
  }

  {
    vector<float> values;
    vector<string> model_ids;
    vector<unsigned int> prim_ids;
    renderer.DepthValues(70, 72, &values, &model_ids, &prim_ids);
    REQUIRE(1 == values.size());
    REQUIRE(Approx(6.0) == values[0]);
  }

  {
    vector<float> values;
    vector<string> model_ids;
    vector<unsigned int> prim_ids;
    renderer.DepthValues(80, 82, &values, &model_ids, &prim_ids);
    REQUIRE(1 == values.size());
    REQUIRE(Approx(6.0) == values[0]);
  }

  {
    vector<float> values;
    vector<string> model_ids;
    vector<unsigned int> prim_ids;
    renderer.DepthValues(121, 79, &values, &model_ids, &prim_ids);
    REQUIRE(1 == values.size());
    REQUIRE(Approx(6.0) == values[0]);
  }
}

#include <fstream>
#include "catch.hpp"
#include "spdlog/spdlog.h"

#include "lib/common.h"
#include "lib/multi_layer_depth_renderer.h"
#include "lib/file_io.h"

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

  struct CameraParams {
    Vec3 cam_eye;
    Vec3 cam_view_dir;
    Vec3 cam_up;
    double x_fov;
    double y_fov;
    double score;  // scene coverage score. not used at the moment.
  };

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
      prim_id_to_node_name
  );

  renderer.ray_tracer()->PrintStats();

  SECTION("Floor pixel") {
    vector<float> values;
    int background_index = renderer.depth_values(100, 200, &values);

    LOGGER->info("Num hits at (100, 200): {}", values.size());
    for (int i = 0; i < values.size(); ++i) {
      LOGGER->info("{}", values[i]);
    }

    REQUIRE(0 == background_index);
    REQUIRE(1 == values.size());
    REQUIRE(Approx(3.14823) == values[0]);
  }

  SECTION("no hit. outdoor") {
    vector<float> values;
    int background_index = renderer.depth_values(172, 109, &values);

    REQUIRE(0 > background_index);
    REQUIRE(0 == values.size());
  }

  SECTION("two hits. no background hit") {
    vector<float> values;
    int background_index = renderer.depth_values(172, 129, &values);

    REQUIRE(0 > background_index);
    REQUIRE(2 == values.size());
  }

  SECTION("one hit. background (door)") {
    vector<float> values;
    int background_index = renderer.depth_values(156, 109, &values);

    REQUIRE(0 == background_index);
    REQUIRE(1 == values.size());
    REQUIRE(Approx(4.9115796) == values[0]);
  }

  SECTION("three hits, two hits through arm of sofa, then background (floor)") {
    vector<float> values;
    int background_index = renderer.depth_values(83, 215, &values);

    REQUIRE(2 == background_index);
    REQUIRE(3 == values.size());
    REQUIRE(Approx(2.9170308) == values[2]);
  }

  SECTION("many hits. two objects, last of which has a coinciding plane on the floor. then background (floor)") {
    vector<float> values;
    int background_index = renderer.depth_values(214, 170, &values);

    LOGGER->info("Num hits at (214, 170): {}", values.size());
    for (int i = 0; i < values.size(); ++i) {
      LOGGER->info("{}", values[i]);

      Vec3 origin = renderer.ray_origin();
      Vec3 dir = renderer.ray_direction(214, 170);

      Vec3 p = origin + values[i] * dir;
      LOGGER->info("{}, {}, {}", p[0], p[1], p[2]);
    }

    REQUIRE(0 < background_index);
    REQUIRE(4 < values.size());
    REQUIRE(Approx(values[values.size() - 2]).margin(0.0025) == values[values.size() - 1]);

    Vec3 origin = renderer.ray_origin();
    Vec3 dir = renderer.ray_direction(214, 170);

    double height_1 = (origin + values[values.size() - 1] * dir)[1];
    double height_2 = (origin + values[values.size() - 2] * dir)[1];
    double height_3 = (origin + values[values.size() - 3] * dir)[1];

    REQUIRE(Approx(height_1).margin(0.001) == height_2);
    REQUIRE(Approx(0.01788).margin(1e-5) == height_3 - height_2);  // Height of the object on the floor.
  }
}


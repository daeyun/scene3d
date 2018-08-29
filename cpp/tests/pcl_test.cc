//
// Created by daeyun on 8/27/18.
//

#include "catch.hpp"
#include "spdlog/spdlog.h"

#include "lib/common.h"
#include "lib/camera.h"
#include "lib/depth_render_utils.h"
#include "lib/pcl.h"
#include "lib/meshdist.h"
#include "lib/benchmark.h"

using namespace scene3d;

TEST_CASE("pcl from depth") {
  std::string obj_filename = "resources/depth_render/dummy.obj";

  PointCloud pcl;

  // Render depth image first.
  SECTION("perspective") {
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

    auto ml_depth = MultiLayerImage<float>(height, width);
    RenderMultiLayerDepthImage(obj_filename, camera, height, width, &ml_depth);

    REQUIRE(3 == ml_depth.values(120, 160)->size());
    REQUIRE(Approx(cam_eye[1] - 1) == ml_depth.at(120, 160, 0));
    REQUIRE(Approx(cam_eye[1] + 1) == ml_depth.at(120, 160, 1));

    Image<float> depth(height, width);
    ml_depth.ExtractLayer(0, &depth);

    PclFromDepth(depth, camera, &pcl);

    pcl.Save("/tmp/scene3d_test/dummy_obj_pcl_00.bin");
  }

  SECTION("orthographic") {
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

    auto ml_depth = MultiLayerImage<float>(height, width);
    RenderMultiLayerDepthImage(obj_filename, camera, height, width, &ml_depth);

    REQUIRE(3 == ml_depth.values(120, 160)->size());
    REQUIRE(Approx(cam_eye[1] - 1) == ml_depth.at(120, 160, 0));
    REQUIRE(Approx(cam_eye[1] + 1) == ml_depth.at(120, 160, 1));

    Image<float> depth(height, width);
    ml_depth.ExtractLayer(0, &depth);

    PclFromDepth(depth, camera, &pcl);

    pcl.Save("/tmp/scene3d_test/dummy_obj_pcl_01.bin");
  }

  std::vector<std::array<unsigned int, 3>> faces;
  std::vector<std::array<float, 3>> vertices;
  std::vector<int> prim_id_to_node_id;
  std::vector<std::string> prim_id_to_node_name;
  bool ok = ReadFacesAndVertices(obj_filename, &faces, &vertices, &prim_id_to_node_id, &prim_id_to_node_name);
  Ensures(ok);
  vector<meshdist_cgal::Triangle> triangles;
  for (const auto &face : faces) {
    auto a = vertices[face[0]];
    auto b = vertices[face[1]];
    auto c = vertices[face[2]];
    triangles.push_back(meshdist_cgal::Triangle{
        Vec3{a[0], a[1], a[2]},
        Vec3{b[0], b[1], b[2]},
        Vec3{c[0], c[1], c[2]}
    });
  }
  std::vector<std::array<float, 3>> points;
  for (int j = 0; j < pcl.NumPoints(); ++j) {
    Vec3 p = pcl.at(j);
    points.push_back(std::array<float, 3>{(float) p[0], (float) p[1], (float) p[2]});
  }

  float pcl_to_triangle_dist = meshdist_cgal::PointsToMeshDistanceOneDirection(points, triangles);
  LOGGER->info("pcl_to_triangle_dist {}", pcl_to_triangle_dist);

  REQUIRE(pcl_to_triangle_dist < 1e-6);

  float triangle_to_pcl_dist = meshdist_cgal::MeshToPointsDistanceOneDirection(triangles, points, 50);
  LOGGER->info("triangle_to_pcl_dist {}", triangle_to_pcl_dist);

  REQUIRE(triangle_to_pcl_dist < 0.6);

}


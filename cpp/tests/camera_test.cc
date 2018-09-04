//
// Created by daeyun on 8/27/18.
//

#include "catch.hpp"
#include "spdlog/spdlog.h"

#include "lib/common.h"
#include "lib/camera.h"
#include "lib/depth_render_utils.h"
#include "lib/pcl.h"
#include "lib/vectorization_utils.h"

using namespace scene3d;

TEST_CASE("vectorization, perspective") {
  for (int k = 0; k < 20; ++k) {
    Vec3 eye = Vec3::Random() * (k + 2);
    Vec3 up = Vec3::Random().normalized();
    Vec3 lookat = Vec3::Random();
    unsigned int height = 120;
    unsigned int width = 160;
    double near = 0.01;
    double far = 1000;
    double x_fov = 30.0 / 180.0 * M_PI;  // 60 degrees left-to-right.
    auto frustum = MakePerspectiveFrustumParams(static_cast<double>(height) / width, x_fov, near, far);

    auto camera = PerspectiveCamera(eye, lookat, up, frustum);
    double xf = 0, yf = 0;
    REQUIRE(camera.fov(&xf, &yf));
    REQUIRE(Approx(xf) == x_fov);
    REQUIRE(xf - yf > 0.1);
    REQUIRE(camera.is_perspective());

    int num_points = 100 + k * 5;

    Points3d pts = Points3d::Random(3, num_points);
    Points3d out_xyz_vectorized;
    Points3d out_xyz(3, num_points);
    Points2i out_xy_vectorized;
    Points2i out_xy(2, num_points);
    Points1d out_d_vectorized;
    Points1d out_d(1, num_points);

    SECTION("world to cam") {
      camera.WorldToCam(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.WorldToCam(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("cam to world") {
      camera.CamToWorld(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.CamToWorld(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("world to cam normal") {
      camera.WorldToCamNormal(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.WorldToCamNormal(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("cam to world normal") {
      camera.CamToWorldNormal(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.CamToWorldNormal(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("cam to frustum") {
      camera.CamToFrustum(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.CamToFrustum(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("frustum to cam") {
      camera.FrustumToCam(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.FrustumToCam(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("world to frustum") {
      camera.WorldToFrustum(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.WorldToFrustum(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("frustum to world") {
      camera.FrustumToWorld(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.FrustumToWorld(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("cam to image") {
      camera.CamToImage(pts, height, width, &out_xy_vectorized, &out_d_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec2i xy;
        double d;
        camera.CamToImage(pts.col(j), height, width, &xy, &d);
        out_xy.col(j) = xy;
        out_d[j] = d;
      }
      REQUIRE(out_xy_vectorized.isApprox(out_xy));
      REQUIRE(out_d_vectorized.isApprox(out_d));
    }

    SECTION("world to image") {
      camera.WorldToImage(pts, height, width, &out_xy_vectorized, &out_d_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec2i xy;
        double d;
        camera.WorldToImage(pts.col(j), height, width, &xy, &d);
        out_xy.col(j) = xy;
        out_d[j] = d;
      }
      REQUIRE(out_xy_vectorized.isApprox(out_xy));
      REQUIRE(out_d_vectorized.isApprox(out_d));
    }
  }
}

TEST_CASE("vectorization, orthographic") {
  for (int k = 0; k < 20; ++k) {
    Vec3 eye = Vec3::Random() * (k + 1);
    Vec3 up = Vec3::Random().normalized();
    Vec3 lookat = Vec3::Random();
    unsigned int height = 120;
    unsigned int width = 160;
    double near = 0.01;
    double far = 1000;

    // not symmetric.
    FrustumParams frustum;
    double fsize = 2;
    frustum.left = -fsize;
    frustum.right = fsize * 1.2;
    frustum.bottom = -fsize * 0.75;
    frustum.top = fsize * 0.9;
    frustum.near = near;
    frustum.far = far;

    auto camera = OrthographicCamera(eye, lookat, up, frustum);
    REQUIRE(!camera.fov(nullptr, nullptr));
    REQUIRE(!camera.is_perspective());

    int num_points = 100 + k * 5;

    Points3d pts = Points3d::Random(3, num_points);
    Points3d out_xyz_vectorized;
    Points3d out_xyz(3, num_points);
    Points2i out_xy_vectorized;
    Points2i out_xy(2, num_points);
    Points1d out_d_vectorized;
    Points1d out_d(1, num_points);

    SECTION("world to cam") {
      camera.WorldToCam(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.WorldToCam(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("cam to world") {
      camera.CamToWorld(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.CamToWorld(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("world to cam normal") {
      camera.WorldToCamNormal(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.WorldToCamNormal(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("cam to world normal") {
      camera.CamToWorldNormal(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.CamToWorldNormal(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("cam to frustum") {
      camera.CamToFrustum(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.CamToFrustum(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("frustum to cam") {
      camera.FrustumToCam(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.FrustumToCam(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("world to frustum") {
      camera.WorldToFrustum(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.WorldToFrustum(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("frustum to world") {
      camera.FrustumToWorld(pts, &out_xyz_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec3 v;
        camera.FrustumToWorld(pts.col(j), &v);
        out_xyz.col(j) = v;
      }
      REQUIRE(out_xyz_vectorized.isApprox(out_xyz));
    }

    SECTION("cam to image") {
      camera.CamToImage(pts, height, width, &out_xy_vectorized, &out_d_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec2i xy;
        double d;
        camera.CamToImage(pts.col(j), height, width, &xy, &d);
        out_xy.col(j) = xy;
        out_d[j] = d;
      }
      REQUIRE(out_xy_vectorized.isApprox(out_xy));
      REQUIRE(out_d_vectorized.isApprox(out_d));
    }

    SECTION("world to image") {
      camera.WorldToImage(pts, height, width, &out_xy_vectorized, &out_d_vectorized);
      for (int j = 0; j < num_points; ++j) {
        Vec2i xy;
        double d;
        camera.WorldToImage(pts.col(j), height, width, &xy, &d);
        out_xy.col(j) = xy;
        out_d[j] = d;
      }
      REQUIRE(out_xy_vectorized.isApprox(out_xy));
      REQUIRE(out_d_vectorized.isApprox(out_d));
    }
  }
}

TEST_CASE("transitivity, perspective") {
  for (int k = 0; k < 20; ++k) {
    Vec3 eye = Vec3::Random() * (k + 2);
    Vec3 up = Vec3::Random().normalized();
    Vec3 lookat = Vec3::Random();
    unsigned int height = 12000;
    unsigned int width = 16000;
    double near = 0.01;
    double far = 1000;
    double x_fov = 30.0 / 180.0 * M_PI;  // 60 degrees left-to-right.
    auto frustum = MakePerspectiveFrustumParams(static_cast<double>(height) / width, x_fov, near, far);

    auto camera = PerspectiveCamera(eye, lookat, up, frustum);
    double xf = 0, yf = 0;
    REQUIRE(camera.fov(&xf, &yf));
    REQUIRE(Approx(xf) == x_fov);
    REQUIRE(xf - yf > 0.1);
    REQUIRE(camera.is_perspective());

    int num_points = 200 + k * 5;

    Points3d world_xyz = Points3d::Random(3, num_points);

    Points3d cam_xyz, frustum_xyz, frustum_xyz2;
    camera.WorldToCam(world_xyz, &cam_xyz);
    camera.CamToFrustum(cam_xyz, &frustum_xyz);
    camera.WorldToFrustum(world_xyz, &frustum_xyz2);

    REQUIRE(frustum_xyz.isApprox(frustum_xyz2));

    Points3d cam_xyz2, world_xyz2, world_xyz3;
    camera.FrustumToCam(frustum_xyz, &cam_xyz2);
    REQUIRE(cam_xyz.isApprox(cam_xyz2));
    camera.CamToWorld(cam_xyz2, &world_xyz2);
    REQUIRE(world_xyz.isApprox(world_xyz2, 1e-10));  // Less precise.
    camera.FrustumToWorld(frustum_xyz, &world_xyz3);
    REQUIRE(world_xyz.isApprox(world_xyz3, 1e-10));

    Points2i xy1;
    Points1d d1;
    camera.WorldToImage(world_xyz, height, width, &xy1, &d1);
    Points2i xy2;
    Points1d d2;
    camera.CamToImage(cam_xyz, height, width, &xy2, &d2);
    REQUIRE(xy1.isApprox(xy2));
    REQUIRE(d1.isApprox(d2));
  }
}

TEST_CASE("transitivity, orthographic") {
  for (int k = 0; k < 20; ++k) {
    Vec3 eye = Vec3::Random() * (k + 1);
    Vec3 up = Vec3::Random().normalized();
    Vec3 lookat = Vec3::Random();
    unsigned int height = 120;
    unsigned int width = 160;
    double near = 0.01;
    double far = 1000;

    // not symmetric.
    FrustumParams frustum;
    double fsize = 2;
    frustum.left = -fsize;
    frustum.right = fsize * 1.2;
    frustum.bottom = -fsize * 0.75;
    frustum.top = fsize * 0.9;
    frustum.near = near;
    frustum.far = far;

    auto camera = OrthographicCamera(eye, lookat, up, frustum);
    REQUIRE(!camera.fov(nullptr, nullptr));
    REQUIRE(!camera.is_perspective());

    int num_points = 200 + k * 5;

    Points3d world_xyz = Points3d::Random(3, num_points);

    Points3d cam_xyz, frustum_xyz, frustum_xyz2;
    camera.WorldToCam(world_xyz, &cam_xyz);
    camera.CamToFrustum(cam_xyz, &frustum_xyz);
    camera.WorldToFrustum(world_xyz, &frustum_xyz2);

    REQUIRE(frustum_xyz.isApprox(frustum_xyz2));

    Points3d cam_xyz2, world_xyz2, world_xyz3;
    camera.FrustumToCam(frustum_xyz, &cam_xyz2);
    REQUIRE(cam_xyz.isApprox(cam_xyz2));
    camera.CamToWorld(cam_xyz2, &world_xyz2);
    REQUIRE(world_xyz.isApprox(world_xyz2, 1e-10));  // Less precise.
    camera.FrustumToWorld(frustum_xyz, &world_xyz3);
    REQUIRE(world_xyz.isApprox(world_xyz3, 1e-10));

    Points2i xy1;
    Points1d d1;
    camera.WorldToImage(world_xyz, height, width, &xy1, &d1);
    Points2i xy2;
    Points1d d2;
    camera.CamToImage(cam_xyz, height, width, &xy2, &d2);
    REQUIRE(xy1.isApprox(xy2));
    REQUIRE(d1.isApprox(d2));
  }
}

TEST_CASE("image to cam, perspective") {
  std::string obj_filename = "resources/depth_render/plane_grid.obj";
  unsigned int width = 2000;
  unsigned int height = 1500;
  double x_fov = 30.0 / 180.0 * M_PI;  // 60 degrees left-to-right.
  double near = 0.01;
  double far = 20;
  Vec3 eye = {0, 4, 0};  // This is the distance where the plane takes up 80% of the image width.
  Vec3 up = {0, 0, -1};
  Vec3 lookat = {0, 0, 0};
  auto frustum = MakePerspectiveFrustumParams(static_cast<double>(height) / width, x_fov, near, far);
  auto camera = PerspectiveCamera(eye, lookat, up, frustum);

  std::vector<std::array<unsigned int, 3>> faces;
  std::vector<std::array<float, 3>> vertices;
  std::vector<int> prim_id_to_node_id;
  std::vector<std::string> prim_id_to_node_name;
  bool ok = ReadFacesAndVertices(obj_filename, &faces, &vertices, &prim_id_to_node_id, &prim_id_to_node_name);
  Ensures(ok);

  scene3d::RayTracer ray_tracer(faces, vertices);
  ray_tracer.PrintStats();

  auto renderer = scene3d::SimpleMultiLayerDepthRenderer(
      &ray_tracer,
      &camera,
      width,
      height
  );

  auto ml_depth = MultiLayerImage<float>(height, width);
  auto ml_prim_ids = MultiLayerImage<uint32_t>(height, width);
  RenderMultiLayerDepthImage(&renderer, &ml_depth, &ml_prim_ids);

  Image<float> depth(height, width);
  ml_depth.ExtractLayer(0, &depth);

  Points3d points;
  ToEigen(vertices, &points);

  Points2i xy;
  Points1d d;

  camera.WorldToImage(points, height, width, &xy, &d);

  for (int i = 0; i < xy.cols(); ++i) {
    double image_depth = depth.at(xy.col(i)[1], xy.col(i)[0]);
    double proj_depth = d[i];
    if (std::isfinite(image_depth)) {
      REQUIRE(Approx(image_depth).margin(1e-2) == proj_depth);
    }

    // If the point is projected to a background, there must be a non-background within 1 pixel.
    double min_diff = kInfinity;
    for (int j = -1; j < 2; ++j) {
      for (int k = -1; k < 2; ++k) {
        if (i == j) {
          continue;
        }
        double neighbor_depth = depth.at(xy.col(i)[1] + j, xy.col(i)[0] + k);
        if (std::isfinite(neighbor_depth)) {
          double diff = std::abs(neighbor_depth - proj_depth);
          if (diff < min_diff) {
            min_diff = diff;
          }
        }
      }
    }
    REQUIRE(min_diff < 1e-2);
  }

  Points3d restored_world;
  camera.ImageToWorld(xy, d, height, width, &restored_world);

  // The y dimension (z in camera coordinates) should be precise because it was never discretized.
  REQUIRE(points.row(1).isApprox(restored_world.row(1)));

  REQUIRE(points.row(0).isApprox(restored_world.row(0), 1e-2));
  REQUIRE(points.row(2).isApprox(restored_world.row(2), 1e-2));

  Points2i xy2;
  Points1d d2;
  camera.WorldToImage(restored_world, height, width, &xy2, &d2);
  Points3d restored_world2;
  camera.ImageToWorld(xy, d, height, width, &restored_world2);
  // Now project the restored point cloud to image coordinates and then back to world coordinates. There should be no discretization error this time.
  REQUIRE(restored_world.isApprox(restored_world2));

}

TEST_CASE("image to cam, orthographic") {
  std::string obj_filename = "resources/depth_render/plane_grid.obj";
  unsigned int width = 2000;
  unsigned int height = 1500;
  double near = 0.01;
  double far = 20;
  Vec3 eye = {0, 4, 0};
  Vec3 up = {0, 0, -1};
  Vec3 lookat = {0, 0, 0};

  // not symmetric.
  FrustumParams frustum;
  double fsize = 2;
  frustum.left = -fsize;
  frustum.right = fsize * 1.1;
  frustum.bottom = -fsize * 0.75;
  frustum.top = fsize * 0.9;
  frustum.near = near;
  frustum.far = far;

  auto camera = OrthographicCamera(eye, lookat, up, frustum);
  REQUIRE(!camera.fov(nullptr, nullptr));
  REQUIRE(!camera.is_perspective());

  std::vector<std::array<unsigned int, 3>> faces;
  std::vector<std::array<float, 3>> vertices;
  std::vector<int> prim_id_to_node_id;
  std::vector<std::string> prim_id_to_node_name;
  bool ok = ReadFacesAndVertices(obj_filename, &faces, &vertices, &prim_id_to_node_id, &prim_id_to_node_name);
  Ensures(ok);

  scene3d::RayTracer ray_tracer(faces, vertices);
  ray_tracer.PrintStats();

  auto renderer = scene3d::SimpleMultiLayerDepthRenderer(
      &ray_tracer,
      &camera,
      width,
      height
  );

  auto ml_depth = MultiLayerImage<float>(height, width);
  auto ml_prim_ids = MultiLayerImage<uint32_t>(height, width);
  RenderMultiLayerDepthImage(&renderer, &ml_depth, &ml_prim_ids);

  Image<float> depth(height, width);
  ml_depth.ExtractLayer(0, &depth);

  Points3d points;
  ToEigen(vertices, &points);

  Points2i xy;
  Points1d d;

  camera.WorldToImage(points, height, width, &xy, &d);

  for (int i = 0; i < xy.cols(); ++i) {
    double image_depth = depth.at(xy.col(i)[1], xy.col(i)[0]);
    double proj_depth = d[i];
    if (std::isfinite(image_depth)) {
      REQUIRE(Approx(image_depth).margin(1e-2) == proj_depth);
    }

    // If the point is projected to a background, there must be a non-background within 1 pixel.
    double min_diff = kInfinity;
    for (int j = -1; j < 2; ++j) {
      for (int k = -1; k < 2; ++k) {
        if (i == j) {
          continue;
        }
        double neighbor_depth = depth.at(xy.col(i)[1] + j, xy.col(i)[0] + k);
        if (std::isfinite(neighbor_depth)) {
          double diff = std::abs(neighbor_depth - proj_depth);
          if (diff < min_diff) {
            min_diff = diff;
          }
        }
      }
    }
    REQUIRE(min_diff < 1e-2);
  }

  Points3d restored_world;
  camera.ImageToWorld(xy, d, height, width, &restored_world);

  // The y dimension (z in camera coordinates) should be precise because it was never discretized.
  REQUIRE(points.row(1).isApprox(restored_world.row(1)));

  REQUIRE(points.row(0).isApprox(restored_world.row(0), 1e-2));
  REQUIRE(points.row(2).isApprox(restored_world.row(2), 1e-2));

  Points2i xy2;
  Points1d d2;
  camera.WorldToImage(restored_world, height, width, &xy2, &d2);
  Points3d restored_world2;
  camera.ImageToWorld(xy, d, height, width, &restored_world2);
  // Now project the restored point cloud to image coordinates and then back to world coordinates. There should be no discretization error this time.
  REQUIRE(restored_world.isApprox(restored_world2));

}

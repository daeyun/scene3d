#include <utility>

#pragma once

#include "common.h"
#include "file_io.h"
#include "camera.h"
#include "depth.h"

namespace scene3d {

class PointCloud {
 public:
  PointCloud() = default;
  explicit PointCloud(Points3d points) : points_(std::move(points)) {
    Ensures(NumPoints() > 0);
  }
  explicit PointCloud(const std::vector<std::array<float, 3>> &points) {
    Ensures(!points.empty());
    points_.resize(3, points.size());
    for (int i = 0; i < points.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        points_(j, i) = points[i][j];
      }
    }
  }

  const Points3d &points() const { return points_; }

  Vec3 at(unsigned int i) const {
    Ensures(i < points_.cols());
    return points_.col(i);
  }

  unsigned int NumPoints() const {
    return static_cast<unsigned int>(points_.cols());
  };

  void Save(const std::string &filename) const {
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> transposed_points = points_.transpose().cast<float>().eval();
    SerializeTensor<float>(filename, transposed_points.data(), {static_cast<int>(NumPoints()), 3});
  }

 private:
  Points3d points_;
  vector<uint8_t> label_;
};

static void PclFromDepth(const DepthImage &depth, const Camera &camera, PointCloud *out) {
  vector<Vec3> xyz;
  for (unsigned int y = 0; y < depth.height(); ++y) {
    for (unsigned int x = 0; x < depth.width(); ++x) {
      float value = depth.at(y, x);
      if (std::isfinite(value)) {
        xyz.emplace_back(x, y, value);
      }
    }
  }

  Points3d cam_pts(3, xyz.size());
  for (int i = 0; i < xyz.size(); ++i) {
    cam_pts.col(i) = xyz[i];
  }

  // Calculate coordinates in camera space.
  if (camera.is_perspective()) {
    cam_pts.row(0) = ((cam_pts.row(0).array() + 0.5) * (camera.frustum().right - camera.frustum().left) / depth.width() + camera.frustum().left) / camera.frustum().near * cam_pts.row(2).array();
    cam_pts.row(1) = ((cam_pts.row(1).array() + 0.5) * -(camera.frustum().top - camera.frustum().bottom) / depth.height() + camera.frustum().top) / camera.frustum().near * cam_pts.row(2).array();
    cam_pts.row(2) *= -1;
  } else {
    cam_pts.row(0) = (cam_pts.row(0).array() + 0.5) * (camera.frustum().right - camera.frustum().left) / depth.width() + camera.frustum().left;
    cam_pts.row(1) = (cam_pts.row(1).array() + 0.5) * -(camera.frustum().top - camera.frustum().bottom) / depth.height() + camera.frustum().top;
    cam_pts.row(2) *= -1;
  };

  Points3d ret;
  camera.CamToWorld(cam_pts, &ret);
  *out = PointCloud(ret);
}

}

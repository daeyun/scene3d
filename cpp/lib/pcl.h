#include <utility>

#pragma once

#include "common.h"
#include "file_io.h"
#include "camera.h"
#include "depth.h"

namespace scene3d {

class BoundingBox {
 public:
  BoundingBox(const Vec3 &corner0, const Vec3 &corner1) {
    bmin = {
        std::min(corner0[0], corner1[0]),
        std::min(corner0[1], corner1[1]),
        std::min(corner0[2], corner1[2]),
    };
    bmax = {
        std::max(corner0[0], corner1[0]),
        std::max(corner0[1], corner1[1]),
        std::max(corner0[2], corner1[2]),
    };
  }

  Vec3 bmin;
  Vec3 bmax;
};

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

void ValidPixelCoordinates(const Image<float> &depth, Points2i *out_xy, Points1d *out_values) {
  vector<Vec2i> xy;
  vector<double> d;
  for (unsigned int y = 0; y < depth.height(); ++y) {
    for (unsigned int x = 0; x < depth.width(); ++x) {
      float value = depth.at(y, x);
      if (std::isfinite(value)) {
        xy.emplace_back(x, y);
        d.push_back(value);
      }
    }
  }

  out_xy->resize(2, xy.size());
  out_values->resize(1, xy.size());
  for (int i = 0; i < xy.size(); ++i) {
    out_xy->col(i) = xy[i];
    (*out_values)[i] = d[i];
  }
}

static void PclFromDepth(const Image<float> &depth, const Camera &camera, PointCloud *out) {
  Points2i cam_pts;
  Points1d cam_d;

  ValidPixelCoordinates(depth, &cam_pts, &cam_d);

  // Calculate coordinates in camera space.
  Points3d cam_out;
  camera.ImageToCam(cam_pts, cam_d, depth.height(), depth.width(), &cam_out);

  Points3d ret;
  camera.CamToWorld(cam_out, &ret);
  *out = PointCloud(ret);
}

}

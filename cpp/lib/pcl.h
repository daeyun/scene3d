#pragma once

#include <utility>
#include <limits>

#include "common.h"
#include "file_io.h"
#include "camera.h"
#include "depth.h"
#include "suncg_utils.h"

namespace scene3d {

// Axis-aligned bounding box.
class AABB {
 public:
  AABB(const Vec3 &corner0, const Vec3 &corner1) {
    Grow(corner0);
    Grow(corner1);
  }

  AABB() = default;

  void Grow(const Vec3 &p) {
    if (is_initialized_) {
      for (int i = 0; i < 3; ++i) {
        if (p[i] < bmin[i]) {
          bmin[i] = p[i];
        } else if (p[i] > bmax[i]) {
          bmax[i] = p[i];
        }
      }
    } else {
      bmin = p;
      bmax = p;
      is_initialized_ = true;
    }
  }

  void Expand(double ratio) {
    if (!is_initialized_) {
      return;
    }
    Vec3 c = Center();
    Vec3 r = 0.5 * (bmax - bmin);
    bmin = c - r * ratio;
    bmax = c + r * ratio;
  }

  double XZArea() const {
    if (!is_initialized_) {
      return 0;
    }
    Vec3 diff = bmax - bmin;
    double ret = diff(0) * diff(2);
    Ensures(std::isfinite(ret));
    return ret;
  }

  Vec3 Center() const {
    Expects(is_initialized_);
    return 0.5 * (bmin + bmax);
  }

  Vec3 bmin;
  Vec3 bmax;

 private:
  bool is_initialized_ = false;
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

void ValidPixelCoordinates(const Image<float> &depth, Points2i *out_xy, Points1d *out_values);

void PclFromDepthInCamCoords(const Image<float> &depth, const Camera &camera, Points3d *out);

void PclFromDepthInCamCoords(const Image<float> &depth, const Camera &camera, Points2i *xy, Points3d *out);

void PclFromDepthInWorldCoords(const Image<float> &depth, const Camera &camera, Points3d *out);

void PclFromDepthInWorldCoords(const Image<float> &depth, const Camera &camera, Points2i *xy, Points3d *out);

void PclFromDepth(const Image<float> &depth, const Camera &camera, PointCloud *out);

void MeanAndStd(const Points3d &points, Vec3 *mean, Vec3 *stddev);

unique_ptr<OrthographicCamera> ComputeOverheadCamera(const MultiLayerImage<float> &ml_depth,
                                                     const MultiLayerImage<uint16_t> &ml_model_indices,
                                                     const MultiLayerImage<uint32_t> &ml_prim_ids,
                                                     const PerspectiveCamera &camera,
                                                     suncg::Scene *scene,
                                                     double overhead_hw_ratio,
                                                     AABB *average_bounding_box,
                                                     vector<AABB> *candidate_boxes,
                                                     PerspectiveCamera *aligner);

void SaveAABB(const string &txt_filename, const vector<AABB> &boxes);

}

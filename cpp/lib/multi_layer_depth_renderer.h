#pragma once

#include "common.h"
#include "ray_mesh_intersection.h"
#include "camera.h"

namespace scene3d {
class MultiLayerDepthRenderer {
 public:

  // `ray_tracer` is managed externally.
  MultiLayerDepthRenderer(const scene3d::RayTracer *ray_tracer,
                          const Vec3 &cam_eye,
                          const Vec3 &cam_view_dir,
                          const Vec3 &cam_up,
                          double xf,
                          double yf,
                          size_t width,
                          size_t height,
                          size_t max_hits)
      : ray_tracer_(ray_tracer), xf_(xf), yf_(yf), width_(width), height_(height), max_hits_(max_hits) {
    scene3d::FrustumParams frustum;
    frustum.far = 10000;
    frustum.near = 0.01;
    camera_ = std::make_unique<scene3d::PerspectiveCamera>(cam_eye, cam_eye + cam_view_dir, cam_up, frustum);

    // Distance to the image plane according to the x fov.
    double xl = 0.5 * width_ / std::tan(xf);
    // Distance to the image plane according to the y fov.
    double yl = 0.5 * height_ / std::tan(yf);

    // For now, we assume the aspect ratio is always 1.0. So the distance to image plane should end up being the same according to both x and y.
    // Otherwise the image size or focal length is wrong. This can also happen because of precision error.
    // 0.01 is an arbitrary threshold.
    if (std::abs(xl - yl) > 0.01) {
      LOGGER->warn("xf: {}, yf: {}, width: {}, height: {}, xl: {}, yl: {}", xf_, yf_, width_, height_, xl, yl);
      throw std::runtime_error("Inconsistent distance to image plane.");
    }

    // Compute the average of the two distances. There are probably other, better ways to do this.
    image_focal_length_ = (xl + yl) * 0.5;

    image_optical_center_ = Vec3{width_ * 0.5, height_ * 0.5, 0};

    const Vec3 cam_ray_origin{0, 0, 0};
    camera_->CamToWorld(cam_ray_origin, &ray_origin_);
  }

  const scene3d::RayTracer *ray_tracer() const {
    return ray_tracer_;
  }

  Vec3 ray_origin() { return ray_origin_; }

  // Implementation specific.
  virtual int depth_values(int x, int y, vector<float> *out) const = 0;

 protected:
  const scene3d::RayTracer *ray_tracer_;
  std::unique_ptr<scene3d::Camera> camera_;
  double xf_;  // TODO(daeyun): refactor so that the fov is part of PerspectiveCamera.
  double yf_;
  size_t width_;
  size_t height_;
  size_t max_hits_;
  double image_focal_length_;
  Vec3 image_optical_center_;
  Vec3 ray_origin_;
};

class SunCgMultiLayerDepthRenderer : public MultiLayerDepthRenderer {
 public:
  SunCgMultiLayerDepthRenderer(const scene3d::RayTracer *ray_tracer,
                               const Vec3 &cam_eye,
                               const Vec3 &cam_view_dir,
                               const Vec3 &cam_up,
                               double xf,
                               double yf,
                               size_t width,
                               size_t height,
                               size_t max_hits,
                               const std::vector<std::string> &prim_id_to_node_name)
      : MultiLayerDepthRenderer(ray_tracer, cam_eye, cam_view_dir, cam_up, xf, yf, width, height, max_hits), prim_id_to_node_name_(prim_id_to_node_name) {}

  Vec3 ray_direction(int x, int y) const {
    Vec3 image_plane_coord{static_cast<double>(x) + 0.5, height_ - (static_cast<double>(y) + 0.5), -image_focal_length_};
    Vec3 cam_ray_direction = (image_plane_coord - image_optical_center_).normalized();
    Vec3 ray_direction;
    camera_->CamToWorldNormal(cam_ray_direction, &ray_direction);
    ray_direction.normalize();
    return ray_direction;
  }

  // x: x pixel coordinates [0, width).
  // y: y pixel coordinates [0, height).
  // out_values: Stack of depth values. e.g.  [FG, O1, O2, ... ,BG]. Can be empty if the ray hits nothing.
  // Returns the index of the first background found. Less than 0 if no background found.
  virtual int depth_values(int x, int y, vector<float> *out_values) const override {
    Vec3 ray_direction = this->ray_direction(x, y);

    int background_value_index = -1;

    // Depth values are collected in the callback function, in the order traversed.
    ray_tracer_->Traverse(ray_origin_, ray_direction, [&](float t, float u, float v, unsigned int prim_id) -> bool {
      bool is_background = IsBackground(prim_id);

      if (background_value_index < 0) {  // No background value previously found.
        out_values->push_back(t);
        // Only the first background hit counts. There could be double walls, etc.
        if (is_background) {
          background_value_index = static_cast<int>(out_values->size() - 1);
        }
      } else { // Background is already found.
        // But if this hit coincides with a previously found background and is not a background itself, add.
        const double kMargin = 0.001;  // This margin can be pretty big, for some reason; based on inspection of nested floors.
        bool is_coincided = std::abs(t - out_values->at(static_cast<size_t>(background_value_index))) < kMargin;
        if (is_coincided) {
          if (!is_background) {
            out_values->push_back(t);
          }
        } else {
          return false;  // Stop traversal.
        }
      }
      return true;
    });

    return background_value_index;
  }

  bool IsBackground(int prim_id) const {
    Expects(prim_id < prim_id_to_node_name_.size());
    auto node_name = prim_id_to_node_name_[prim_id];
    bool ret = false;
    for (const string &substr: kBackgroundNameSubstrings) {
      if (node_name.find(substr) != std::string::npos) {
        ret = true;
        break;
      }
    }
    if (!ret and node_name.find("Model#") != std::string::npos) {
      string model_id = node_name.substr(node_name.find('#') + 1);
      if (kSunCgDoorsAndWindows.find(model_id) != kSunCgDoorsAndWindows.end()) {
        // If the model id matches a window or a door, it is the background.
        ret = true;
      }
    }
    return ret;
  }

 private:
  const std::vector<string> kBackgroundNameSubstrings{"Floor", "Wall", "Ceiling", "Room", "Level", "floor", "background"};

  // TODO(daeyun): These are used for detecting doors and windows. There are probably better ways to do this.
  const std::set<std::string> kSunCgDoorsAndWindows
      {"122", "126", "133", "209", "210", "211", "212", "213", "214", "246", "247", "326", "327", "331", "361", "73", "752", "753", "754", "755", "756", "757", "758", "759", "760", "761",
       "762", "763", "764", "765", "766", "767", "768", "769", "770", "771", "s__1276", "s__1762", "s__1763", "s__1764", "s__1765", "s__1766", "s__1767", "s__1768", "s__1769", "s__1770", "s__1771",
       "s__1772", "s__1773", "s__2010", "s__2011", "s__2012", "s__2013", "s__2014", "s__2015", "s__2016", "s__2017", "s__2019"};
  const std::vector<std::string> prim_id_to_node_name_;
};

}

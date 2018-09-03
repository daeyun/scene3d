#pragma once

#include "common.h"
#include "ray_mesh_intersection.h"
#include "camera.h"

namespace scene3d {

struct Pixel {
  float depth;
  array<float, 3> normal;
  uint16_t model_index;
  uint16_t instance_index;
};

class MultiLayerDepthRenderer {
 public:

  // `ray_tracer` is managed externally.
  MultiLayerDepthRenderer(const scene3d::RayTracer *ray_tracer,
                          const Camera *camera,
                          size_t width,
                          size_t height,
                          size_t max_hits)
      : ray_tracer_(ray_tracer), camera_(camera), width_(width), height_(height), max_hits_(max_hits) {
    if (camera_->is_perspective()) {
      double xf, yf;
      // This whole block of code is from an older version. It makes sure aspect stretching does not happen.
      camera->fov(&xf, &yf);
      // Distance to the image plane according to the x fov.
      double xl = 0.5 * width_ / std::tan(xf);
      // Distance to the image plane according to the y fov.
      double yl = 0.5 * height_ / std::tan(yf);

      // For now, we assume the aspect ratio is always 1.0. So the distance to image plane should end up being the same according to both x and y.
      // Otherwise the image size or focal length is wrong. This can also happen because of precision error.
      // 0.01 is an arbitrary threshold.
      if (std::abs(xl - yl) > 0.01) {
        LOGGER->warn("xf: {}, yf: {}, width: {}, height: {}, xl: {}, yl: {}", xf, yf, width_, height_, xl, yl);
        throw std::runtime_error("Inconsistent distance to image plane.");
      }
      // Compute the average of the two distances. There are probably other, better ways to do this.
      image_focal_length_ = (xl + yl) * 0.5;
    }

    image_optical_center_ = Vec3{width_ * 0.5, height_ * 0.5, 0};

  }

  const scene3d::RayTracer *ray_tracer() const {
    return ray_tracer_;
  }

  Vec3 RayOrigin(int x, int y) const {
    if (camera_->is_perspective()) {
      return camera_->position();
    } else {
      double im_x = static_cast<double>(x) + 0.5;
      double im_y = height_ - (static_cast<double>(y) + 0.5);
      const auto &frustum = camera_->frustum();
      double cam_x = im_x / width_ * (frustum.right - frustum.left) + frustum.left;
      double cam_y = im_y / height_ * (frustum.top - frustum.bottom) + frustum.bottom;
      Vec3 ret;
      camera_->CamToWorld(Vec3{cam_x, cam_y, 0}, &ret);
      return ret;
    }
  }

  Vec3 RayDirection(int x, int y) const {
    Vec3 ray_direction;
    if (camera_->is_perspective()) {
      Vec3 image_plane_coord{static_cast<double>(x) + 0.5, height_ - (static_cast<double>(y) + 0.5), -image_focal_length()};
      Vec3 cam_ray_direction = (image_plane_coord - image_optical_center()).normalized();
      camera_->CamToWorldNormal(cam_ray_direction, &ray_direction);
    } else {
      ray_direction = camera_->viewing_direction();
    }
    ray_direction.normalize();
    return ray_direction;
  }

  // Implementation specific.
  virtual int DepthValues(int x, int y, vector<float> *out, vector<string> *model_ids, vector<unsigned int> *prim_ids) const = 0;

 protected:
  const scene3d::RayTracer *ray_tracer_;
  const scene3d::Camera *camera_;  // Managed externally
  size_t width_;
  size_t height_;
  size_t max_hits_;

 private:
  double image_focal_length() const {
    if (camera_->is_perspective()) {
      return image_focal_length_;
    } else {
      throw std::runtime_error("image focal length should not be used in orthographic projection.");
    }
  }

  const Vec3 &image_optical_center() const {
    if (camera_->is_perspective()) {
      return image_optical_center_;
    } else {
      throw std::runtime_error("image optical center should not be used in orthographic projection.");
    }
  }

  double image_focal_length_;
  Vec3 image_optical_center_;
};

class SunCgMultiLayerDepthRenderer : public MultiLayerDepthRenderer {
 public:
  SunCgMultiLayerDepthRenderer(const scene3d::RayTracer *ray_tracer,
                               const Camera *camera,
                               size_t width,
                               size_t height,
                               size_t max_hits,
                               const std::vector<std::string> &prim_id_to_node_name)
      : MultiLayerDepthRenderer(ray_tracer, camera, width, height, max_hits), prim_id_to_node_name_(prim_id_to_node_name) {}

  // x: x pixel coordinates [0, width).
  // y: y pixel coordinates [0, height).
  // out_values: Stack of depth values. e.g.  [FG, O1, O2, ... ,BG]. Can be empty if the ray hits nothing.
  // Returns the index of the first background found. Less than 0 if no background found.
  virtual int DepthValues(int x, int y, vector<float> *out_values, vector<string> *model_ids, vector<unsigned int> *prim_ids) const override {
    Vec3 ray_direction = this->RayDirection(x, y);

    int background_value_index = -1;

    // Depth values are collected in the callback function, in the order traversed.
    ray_tracer_->Traverse(this->RayOrigin(x, y), ray_direction, [&](float t, float u, float v, unsigned int prim_id) -> bool {
      bool is_background = IsBackground(prim_id);

      if (do_not_render_background_except_floor_) {
        if (is_background && !IsFloor(prim_id)) {
          return true;
        }
      }

      string model_id = GetModelId(prim_id);

      if (background_value_index < 0) {  // No background value previously found.
        out_values->push_back(t);
        model_ids->push_back(model_id);
        prim_ids->push_back(prim_id);
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
            model_ids->push_back(model_id);
            prim_ids->push_back(prim_id);
          }
        } else {
          return false;  // Stop traversal.
        }
      }
      return true;
    });

    // Convert ray displacement to depth.
    const double z = camera_->viewing_direction().dot(ray_direction);
    for (auto &t : *out_values) {
      t *= z;
    }

    return background_value_index;
  }

  // Returns the thickness of the first surface in the inward normal direction.
  virtual float ObjectCenteredVolume(int x, int y) const {
    Vec3 ray_direction = this->RayDirection(x, y);

    int count = 0;
    float ret = -1;

    // Depth values are collected in the callback function, in the order traversed.
    ray_tracer_->TraverseInwardNormalDirection(this->RayOrigin(x, y), ray_direction, [&](float t, float u, float v, unsigned int prim_id) -> bool {
      bool is_background = IsBackground(prim_id);

      if (is_background) {
        return false;  // Stop traversal.
      }
      if (count > 0) {
        ret = t;
        return false;  // Stop traversal.
      }

      count++;
      return true;
    });
    return ret;
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

  bool IsFloor(int prim_id) const {
    Expects(prim_id < prim_id_to_node_name_.size());
    auto node_name = prim_id_to_node_name_[prim_id];
    bool ret = false;
    for (const string &substr: kFloorNameSubstrings) {
      if (node_name.find(substr) != std::string::npos) {
        ret = true;
        break;
      }
    }
    return ret;
  }

  std::string GetModelId(int prim_id) const {
    Expects(prim_id < prim_id_to_node_name_.size());
    auto node_name = prim_id_to_node_name_[prim_id];
    if (node_name.find("Floor") != std::string::npos || node_name.find("floor") != std::string::npos) {
      return "Floor";
    }
    if (node_name.find("Wall") != std::string::npos) {
      return "Wall";
    }
    if (node_name.find("Ceiling") != std::string::npos) {
      return "Ceiling";
    }
    if (node_name.find("Box") != std::string::npos) {
      // Not sure if this ever happens.
      return "Box";
    }
    if (node_name.find("Empty") != std::string::npos) {
      // Not sure if this ever happens.
      return "Empty";
    }
    if (node_name.find("Ground") != std::string::npos) {
      // Ground category is not present in ModelCategoryMapping.csv, but it does happen.
      // TODO(daeyun): Decide what exactly needs to be done with Ground.
      return "Empty";
    }
    if (node_name.find("Model#") != std::string::npos) {
      string model_id = node_name.substr(node_name.find('#') + 1);
      return model_id;
    }

    LOGGER->error("model_id detection failed. node name was {}", node_name);
    throw std::runtime_error("model_id detection failed");
  }

  void set_do_not_render_background_except_floor(bool value) {
    do_not_render_background_except_floor_ = value;
  }

 private:
  const std::vector<string> kBackgroundNameSubstrings{"Floor", "Wall", "Ceiling", "Room", "Level", "floor", "background"};
  const std::vector<string> kFloorNameSubstrings{"Floor", "Level", "floor"};

  // TODO(daeyun): These are used for detecting doors and windows. There are probably better ways to do this.
//  const std::set<std::string> kSunCgDoorsAndWindows
//      {"122", "126", "133", "209", "210", "211", "212", "213", "214", "246", "247", "326", "327", "331", "361", "73", "752", "753", "754", "755", "756", "757", "758", "759", "760", "761",
//       "762", "763", "764", "765", "766", "767", "768", "769", "770", "771", "s__1276", "s__1762", "s__1763", "s__1764", "s__1765", "s__1766", "s__1767", "s__1768", "s__1769", "s__1770", "s__1771",
//       "s__1772", "s__1773", "s__2010", "s__2011", "s__2012", "s__2013", "s__2014", "s__2015", "s__2016", "s__2017", "s__2019"};

  // Now it includes stairs, columns, and rugs. otherwise same as above.
  // TODO(daeyun): rename. or refactor to use nyu category name.
  const std::set<std::string> kSunCgDoorsAndWindows
      {"122", "126", "133", "209", "210", "211", "212", "213", "214", "246", "247", "326", "327", "331", "361", "73", "752", "753", "754", "755", "756", "757", "758", "759", "760", "761",
       "762", "763", "764", "765", "766", "767", "768", "769", "770", "771", "s__1276", "s__1762", "s__1763", "s__1764", "s__1765", "s__1766", "s__1767", "s__1768", "s__1769", "s__1770", "s__1771",
       "s__1772", "s__1773", "s__2010", "s__2011", "s__2012", "s__2013", "s__2014", "s__2015", "s__2016", "s__2017", "s__2019",

       "151", "152", "253", "254", "782", "783", "784", "785", "s__499",  // "stairs"

       "365", "366", "367", // "column"

       "153", "235", "238", "s__1090", "s__1647", // "rug"
      };
  const std::vector<std::string> prim_id_to_node_name_;

  // TODO(daeyun): For now, this is only used in the "DepthValues" function. Not thickness image.
  bool do_not_render_background_except_floor_ = false;
};

}

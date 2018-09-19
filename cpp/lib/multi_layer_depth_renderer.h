#pragma once

#include "common.h"
#include "ray_mesh_intersection.h"
#include "camera.h"
#include "suncg_utils.h"

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
                          size_t height)
      : ray_tracer_(ray_tracer), camera_(camera), width_(width), height_(height) {
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
  virtual int DepthValues(int x, int y, vector<float> *out, vector<uint32_t> *prim_ids) const = 0;
  virtual int ObjectCenteredRayDisplacement(int x, int y, vector<float> *out, vector<uint32_t> *prim_ids) const = 0;

  size_t width() const {
    return width_;
  }
  size_t height() const {
    return height_;
  }

 protected:
  const scene3d::RayTracer *ray_tracer_;
  const scene3d::Camera *camera_;  // Managed externally
  size_t width_;
  size_t height_;

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

class SimpleMultiLayerDepthRenderer : public MultiLayerDepthRenderer {
 public:
  SimpleMultiLayerDepthRenderer(const scene3d::RayTracer *ray_tracer,
                                const Camera *camera,
                                size_t width,
                                size_t height)
      : MultiLayerDepthRenderer(ray_tracer, camera, width, height) {}

  virtual int DepthValues(int x, int y, vector<float> *out_values, vector<uint32_t> *prim_ids) const override {
    Vec3 ray_direction = this->RayDirection(x, y);

    int count = 0;

    // Depth values are collected in the callback function, in the order traversed.
    ray_tracer_->Traverse(this->RayOrigin(x, y), ray_direction, [&](float t, float u, float v, unsigned int prim_id) -> bool {
      out_values->push_back(t);
      prim_ids->push_back(prim_id);
      ++count;
    });

    // Convert ray displacement to depth.
    const double z = camera_->viewing_direction().dot(ray_direction);
    for (auto &t : *out_values) {
      t *= z;
    }

    return count;
  }

  virtual int ObjectCenteredRayDisplacement(int x, int y, vector<float> *out_values, vector<uint32_t> *prim_ids) const override {
    LOGGER->error("Not implemented");
    throw std::runtime_error("not implemented");
    return 0;
  }
};

class SunCgMultiLayerDepthRenderer : public MultiLayerDepthRenderer {
 public:
  SunCgMultiLayerDepthRenderer(const scene3d::RayTracer *ray_tracer,
                               const Camera *camera,
                               size_t width,
                               size_t height,
                               suncg::Scene *scene)
      : MultiLayerDepthRenderer(ray_tracer, camera, width, height), scene_(scene) {}

  // x: x pixel coordinates [0, width).
  // y: y pixel coordinates [0, height).
  // out_values: Stack of depth values. e.g.  [FG, O1, O2, ... ,BG]. Can be empty if the ray hits nothing.
  // Returns the index of the first background found. Less than 0 if no background found.
  virtual int DepthValues(int x, int y, vector<float> *out_values, vector<uint32_t> *prim_ids) const override {
    const Vec3 &ray_direction = this->RayDirection(x, y);

    const double kMargin = 0.001;  // This margin can be pretty big, for some reason; based on inspection of nested floors.

    int background_value_index = -1;  // -1 means no background is found in this pixel.

    // Depth values are collected in the callback function, in the order traversed.
    ray_tracer_->Traverse(this->RayOrigin(x, y), ray_direction, [&](float t, float u, float v, unsigned int prim_id) -> bool {
      const auto &instance = scene_->PrimIdToInstance(prim_id);
      bool is_background = scene_->IsPrimBackground(prim_id);
      bool is_floor = instance.type == suncg::InstanceType::Floor || instance.type == suncg::InstanceType::Ground;

      if (overhead_rendering_mode_) {
        if (is_background) {
          if (is_floor) {
            if (t < 0.5) {
              return true;
            }
          } else {
            return true;  // Skip and continue.
          }
        } else {
          const auto &category = scene_->PrimIdToCategory(prim_id);
          if (category.nyuv2_40class == "void" ||
              category.nyuv2_40class == "person" ||
              category.fine_grained_class == "plant" ||
              category.fine_grained_class == "chandelier" ||
              category.fine_grained_class == "ceiling_fan" ||
              category.fine_grained_class == "decoration" ||
              category.fine_grained_class == "surveillance_camera") {
            return true;
          }
        }
      }

      if (background_value_index < 0) {  // No background value previously found.
        out_values->push_back(t);
        prim_ids->push_back(prim_id);
        // Only the first background hit counts. There could be double walls, etc.
        if (is_background) {
          background_value_index = static_cast<int>(out_values->size() - 1);
        }
      } else { // Background is already found.
        // But if this hit coincides with a previously found background and is not a background itself, add.
        bool is_coincided = std::abs(t - out_values->at(static_cast<size_t>(background_value_index))) < kMargin;
        if (is_coincided) {
          if (!is_background) {
            out_values->push_back(t);
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

    Ensures(prim_ids->size() == out_values->size());

    if (!prim_ids->empty() && background_value_index >= 0 && background_value_index < prim_ids->size() - 1) {
      std::swap(prim_ids->at(prim_ids->size() - 1), prim_ids->at(background_value_index));
      background_value_index = static_cast<int>(prim_ids->size() - 1);
    }

    // TODO(daeyun): `prim_ids` can probably be incorrectly ordered if there are coinciding surfaces.

    return background_value_index;
  }

  virtual int ObjectCenteredRayDisplacement(int x, int y, vector<float> *out_values, vector<uint32_t> *prim_ids) const override {
    const Vec3 &ray_direction = this->RayDirection(x, y);

    const double kMargin = 0.001;  // This margin can be pretty big, for some reason; based on inspection of nested floors.

    int background_value_index = -1;  // -1 means no background is found in this pixel.

    // Depth values are collected in the callback function, in the order traversed.
    ray_tracer_->TraverseInwardNormalDirection(this->RayOrigin(x, y), ray_direction, [&](float t, float u, float v, unsigned int prim_id) -> bool {
      const auto &instance = scene_->PrimIdToInstance(prim_id);
      bool is_background = scene_->IsPrimBackground(prim_id);
      bool is_floor = instance.type == suncg::InstanceType::Floor || instance.type == suncg::InstanceType::Ground;

      if (overhead_rendering_mode_) {
        if (is_background) {
          if (is_floor) {
            if (t < 0.5) {
              return true;
            }
          } else {
            return true;  // Skip and continue.
          }
        } else {
          const auto &category = scene_->PrimIdToCategory(prim_id);
          if (category.nyuv2_40class == "void" ||
              category.nyuv2_40class == "person" ||
              category.fine_grained_class == "plant" ||
              category.fine_grained_class == "chandelier" ||
              category.fine_grained_class == "ceiling_fan" ||
              category.fine_grained_class == "decoration" ||
              category.fine_grained_class == "surveillance_camera") {
            return true;
          }
        }
      }

      if (background_value_index < 0) {  // No background value previously found.
        out_values->push_back(t);
        prim_ids->push_back(prim_id);
        // Only the first background hit counts. There could be double walls, etc.
        if (is_background) {
          background_value_index = static_cast<int>(out_values->size() - 1);
        }
      } else { // Background is already found.
        // But if this hit coincides with a previously found background and is not a background itself, add.
        bool is_coincided = std::abs(t - out_values->at(static_cast<size_t>(background_value_index))) < kMargin;
        if (is_coincided) {
          if (!is_background) {
            out_values->push_back(t);
            prim_ids->push_back(prim_id);
          }
        } else {
          return false;  // Stop traversal.
        }
      }
      return true;
    });

    // We want ray displacement, not depth. So no conversion here.

    Ensures(prim_ids->size() == out_values->size());

    if (!prim_ids->empty() && background_value_index >= 0 && background_value_index < prim_ids->size() - 1) {
      std::swap(prim_ids->at(prim_ids->size() - 1), prim_ids->at(background_value_index));
      background_value_index = static_cast<int>(prim_ids->size() - 1);
    }

    // TODO(daeyun): `prim_ids` can probably be incorrectly ordered if there are coinciding surfaces.

    return background_value_index;
  }

  void set_overhead_rendering_mode(bool value) {
    overhead_rendering_mode_ = value;
  }

 private:
  suncg::Scene *scene_;

  // TODO(daeyun): For now, this is only used in the "DepthValues" function. Not thickness image.
  bool overhead_rendering_mode_ = false;
};

}

//
// Created by daeyun on 4/11/17.
//

#pragma once

#include "common.h"

namespace scene3d {
struct FrustumParams {
  double left = -1;
  double right = 1;
  double bottom = -1;
  double top = 1;
  double near = 0.001;
  double far = 50;
};

// `hw_ratio` is height/width. e.g. 0.75
// `x_fov` must be half-angle, in radians.
FrustumParams MakePerspectiveFrustumParams(double hw_ratio, double x_fov, double near, double far) {
  FrustumParams ret;
  ret.right = std::abs(std::tan(x_fov) * near);
  ret.top = ret.right * hw_ratio;

  ret.left = -ret.right;
  ret.bottom = -ret.top;

  ret.near = near;
  ret.far = far;
  return ret;
}

class Camera {
 public:
  Camera(const Vec3 &camera_position,
         const Vec3 &lookat_position,
         const Vec3 &up,
         const FrustumParams &frustum)
      : position_(camera_position),
        lookat_position_(lookat_position),
        up_(up),
        frustum_(frustum) {
    auto viewing_direction = (lookat_position - camera_position).normalized();
    auto right = viewing_direction.cross(up).normalized();
    auto up_vector = right.cross(viewing_direction);

    view_mat_(0, 0) = right[0];
    view_mat_(0, 1) = right[1];
    view_mat_(0, 2) = right[2];
    view_mat_(1, 0) = up_vector[0];
    view_mat_(1, 1) = up_vector[1];
    view_mat_(1, 2) = up_vector[2];
    view_mat_(2, 0) = -viewing_direction[0];
    view_mat_(2, 1) = -viewing_direction[1];
    view_mat_(2, 2) = -viewing_direction[2];

    Vec3 translation = -(view_mat_.topLeftCorner<3, 3>() * camera_position);
    view_mat_(0, 3) = translation[0];
    view_mat_(1, 3) = translation[1];
    view_mat_(2, 3) = translation[2];

    view_mat_(3, 0) = 0;
    view_mat_(3, 1) = 0;
    view_mat_(3, 2) = 0;
    view_mat_(3, 3) = 1;

    view_mat_inv_.topLeftCorner<3, 3>() =
        view_mat_.topLeftCorner<3, 3>().transpose();
    view_mat_inv_.block<3, 1>(0, 3) = camera_position;

    viewing_direction_ = (lookat_position_ - position_).normalized();
  }

  void WorldToCam(const Vec3 &xyz, Vec3 *out) const {
    Vec4 hom;
    hom.head<3>().array() = xyz;
    hom(3) = 1.0;
    *out = view_mat_.topRows<3>() * hom;
  }

  void WorldToCam(const Points3d &xyz, Points3d *out) const {
    Points4d hom(4, xyz.cols());
    hom.topRows<3>().array() = xyz;
    hom.row(3).array() = 1.0;
    *out = view_mat_.topRows<3>() * hom;
  }

  void CamToWorld(const Vec3 &xyz, Vec3 *out) const {
    Vec4 hom;
    hom.head<3>().array() = xyz;
    hom(3) = 1.0;
    *out = view_mat_inv_.topRows<3>() * hom;
  }

  void CamToWorld(const Points3d &xyz, Points3d *out) const {
    Points4d hom(4, xyz.cols());
    hom.topRows<3>().array() = xyz;
    hom.row(3).array() = 1.0;
    *out = view_mat_inv_.topRows<3>() * hom;
  }

  void CamToWorldNormal(const Vec3 &xyz, Vec3 *out) const {
    const auto rot = view_mat_inv_.topLeftCorner<3, 3>();
    *out = rot * xyz;
  }

  void WorldToCamNormal(const Vec3 &xyz, Vec3 *out) const {
    const auto rot = view_mat_.topLeftCorner<3, 3>();
    *out = rot * xyz;
  }

  void FrustumToCam(const Vec3 &xyz, Vec3 *out) const {
    Vec4 hom;
    hom.head<3>().array() = xyz;
    hom(3) = 1.0;
    Vec4 p = projection_mat_inv() * hom;
    *out = p.head<3>().array() / p(3);
  }

  void CamToFrustum(const Vec3 &xyz, Vec3 *out) const {
    Vec4 hom;
    hom.head<3>().array() = xyz;
    hom(3) = 1.0;
    Vec4 p = projection_mat() * hom;
    *out = p.head<3>().array() / p(3);
  }

  void CamToFrustum(const Points3d &xyz, Points3d *out) const {
    Points4d hom(4, xyz.cols());
    hom.topRows<3>().array() = xyz;
    hom.row(3).array() = 1.0;
    Points4d p = projection_mat() * hom;
    *out = p.topRows<3>().array().rowwise() / p.row(3).array();
  }

  const Mat44 &view_mat() const {
    return view_mat_;
  }

  const Mat44 &view_mat_inv() const {
    return view_mat_inv_;
  }

  const FrustumParams &frustum() const {
    return frustum_;
  }

  const Vec3 &position() const {
    return position_;
  }

  const Vec3 &lookat_position() const {
    return lookat_position_;
  }

  const Vec3 &viewing_direction() const {
    return viewing_direction_;
  }

  const Vec3 &up() const {
    return up_;
  }

  virtual const Mat44 &projection_mat() const = 0;
  virtual const Mat44 &projection_mat_inv() const = 0;
  virtual bool is_perspective() const = 0;

  // Returns false if this is orthographic. `x_fov` and `y_fov` are half-angles, in radians.
  virtual bool fov(double *x_fov, double *y_fov) const = 0;

 private:
  Vec3 position_;
  Vec3 lookat_position_;
  Vec3 up_;
  Vec3 viewing_direction_;
  Mat44 view_mat_;
  Mat44 view_mat_inv_;
  FrustumParams frustum_;
};

class OrthographicCamera : public Camera {
 public:
  OrthographicCamera(const Vec3 &camera_position,
                     const Vec3 &lookat_position,
                     const Vec3 &up,
                     const FrustumParams &frustum_params)
      : Camera(camera_position, lookat_position, up, frustum_params) {
    auto rl = frustum().right - frustum().left;
    auto tb = frustum().top - frustum().bottom;
    auto fn = frustum().far - frustum().near;
    projection_mat_.setIdentity();
    projection_mat_(0, 0) = 2.0 / rl;
    projection_mat_(1, 1) = 2.0 / tb;
    projection_mat_(2, 2) = -2.0 / fn;
    projection_mat_(0, 3) = -(frustum().right + frustum().left) / rl;
    projection_mat_(1, 3) = -(frustum().top + frustum().bottom) / tb;
    projection_mat_(2, 3) = -(frustum().far + frustum().near) / fn;

    projection_mat_inv_ = projection_mat_.inverse();
  }

  const Mat44 &projection_mat() const override {
    return projection_mat_;
  }

  const Mat44 &projection_mat_inv() const override {
    return projection_mat_inv_;
  }

  bool is_perspective() const override {
    return false;
  }

  bool fov(double *x_fov, double *y_fov) const override {
    return false;
  }

 private:
  Mat44 projection_mat_;
  Mat44 projection_mat_inv_;
};

class PerspectiveCamera : public Camera {
 public:
  PerspectiveCamera(const Vec3 &camera_position,
                    const Vec3 &lookat_position,
                    const Vec3 &up,
                    const FrustumParams &frustum_params)
      : Camera(camera_position, lookat_position, up, frustum_params) {
    // Expects symmetric frustum.
    Expects((frustum().bottom + frustum().top) < 1e-7);
    Expects((frustum().left + frustum().right) < 1e-7);

    auto rl = frustum().right - frustum().left;
    auto tb = frustum().top - frustum().bottom;
    auto fn = frustum().far - frustum().near;
    projection_mat_.setZero();
    projection_mat_(0, 0) = 2.0 * frustum().near / rl;
    projection_mat_(1, 1) = 2.0 * frustum().near / tb;
    projection_mat_(2, 2) = -(frustum().far + frustum().near) / fn;
    projection_mat_(0, 2) = (frustum().right + frustum().left) / rl;
    projection_mat_(1, 2) = (frustum().top + frustum().bottom) / tb;
    projection_mat_(2, 3) = -2.0 * frustum().far * frustum().near / fn;
    projection_mat_(3, 2) = -1.0;
    projection_mat_inv_ = projection_mat_.inverse();
  }

  const Mat44 &projection_mat() const override {
    return projection_mat_;
  }

  const Mat44 &projection_mat_inv() const override {
    return projection_mat_inv_;
  }

  bool is_perspective() const override {
    return true;
  }

  bool fov(double *x_fov, double *y_fov) const override {
    // Frustum is expected to be symmetric.
    *x_fov = std::abs(std::atan2(frustum().right, frustum().near));
    *y_fov = std::abs(std::atan2(frustum().top, frustum().near));
    return true;
  }

 private:
  Mat44 projection_mat_;
  Mat44 projection_mat_inv_;
};
}

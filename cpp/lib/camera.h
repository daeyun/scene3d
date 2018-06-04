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

  void CamToWorld(const Vec3 &xyz, Vec3 *out) const {
    Vec4 hom;
    hom.head<3>().array() = xyz;
    hom(3) = 1.0;
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

  virtual const Mat44 &projection_mat() const = 0;
  virtual const Mat44 &projection_mat_inv() const = 0;

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

 private:
  Mat44 projection_mat_;
  Mat44 projection_mat_inv_;
};
}

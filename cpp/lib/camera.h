//
// Created by daeyun on 4/11/17.
//

#pragma once

#include <fstream>
#include <iomanip>

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
        frustum_(frustum) {
    Vec3 viewing_direction = (lookat_position - camera_position).normalized();
    Vec3 right = viewing_direction.cross(up).normalized();
    Vec3 up_vector = right.cross(viewing_direction).normalized();

    viewing_direction_ = viewing_direction;
    up_ = up_vector;  // Up vector can change if the initial value is not orthogonal.

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
  }

  void WorldToCam(const Vec3 &xyz, Vec3 *out) const {
    *out = view_mat_.topRows<3>() * xyz.homogeneous();
  }

  void WorldToCam(const Points3d &xyz, Points3d *out) const {
    *out = view_mat_.topRows<3>() * xyz.colwise().homogeneous();
  }

  void CamToWorld(const Vec3 &xyz, Vec3 *out) const {
    *out = view_mat_inv_.topRows<3>() * xyz.homogeneous();
  }

  void CamToWorld(const Points3d &xyz, Points3d *out) const {
    *out = view_mat_inv_.topRows<3>() * xyz.colwise().homogeneous();
  }

  void WorldToCamNormal(const Vec3 &xyz, Vec3 *out) const {
    *out = view_mat_.topLeftCorner<3, 3>() * xyz;
  }

  void WorldToCamNormal(const Points3d &xyz, Points3d *out) const {
    *out = view_mat_.topLeftCorner<3, 3>() * xyz;
  }

  void CamToWorldNormal(const Vec3 &xyz, Vec3 *out) const {
    *out = view_mat_inv_.topLeftCorner<3, 3>() * xyz;
  }

  void CamToWorldNormal(const Points3d &xyz, Points3d *out) const {
    *out = view_mat_inv_.topLeftCorner<3, 3>() * xyz;
  }

  // Camera coordinates to NDC.
  void CamToFrustum(const Vec3 &xyz, Vec3 *out) const {
    *out = (projection_mat() * xyz.homogeneous()).hnormalized();
  }

  void CamToFrustum(const Points3d &xyz, Points3d *out) const {
    *out = (projection_mat() * xyz.colwise().homogeneous()).colwise().hnormalized();
  }

  // NDC to Camera coordinates.
  void FrustumToCam(const Vec3 &xyz, Vec3 *out) const {
    *out = (projection_mat_inv() * xyz.homogeneous()).hnormalized();
  }

  void FrustumToCam(const Points3d &xyz, Points3d *out) const {
    *out = (projection_mat_inv() * xyz.colwise().homogeneous()).colwise().hnormalized();
  }

  // World coordinates to NDC.
  void WorldToFrustum(const Vec3 &xyz, Vec3 *out) const {
    *out = ((projection_mat() * view_mat_) * xyz.homogeneous()).hnormalized();
  }

  void WorldToFrustum(const Points3d &xyz, Points3d *out) const {
    *out = ((projection_mat() * view_mat_) * xyz.colwise().homogeneous()).colwise().hnormalized();
  }

  // NDC to World coordinates.
  void FrustumToWorld(const Vec3 &xyz, Vec3 *out) const {
    // For some reason (view_mat_inv_ * projection_mat_inv()) is unstable. They are mathematically the same.
    Mat44 pv_inv = (projection_mat() * view_mat_).inverse();  // TODO(daeyun): Avoid re-computing this.
    *out = (pv_inv * xyz.homogeneous()).hnormalized();
  }

  void FrustumToWorld(const Points3d &xyz, Points3d *out) const {
    Mat44 pv_inv = (projection_mat() * view_mat_).inverse();
    *out = (pv_inv * xyz.colwise().homogeneous()).colwise().hnormalized();
  }

  // `cam_depth_value` is optional.
  void CamToImage(const Vec3 &xyz, unsigned int height, unsigned int width, Vec2i *image_xy, double *cam_depth_value = nullptr) const {
    Vec3 ndc;
    CamToFrustum(xyz, &ndc);
    *image_xy = ((ndc.topRows<2>().array() + 1.0).colwise() * (Vec2{width, height} * 0.5).array()).cast<int>();
    image_xy->row(1) = height - image_xy->row(1).array() - 1;  // flip y.

    if (cam_depth_value) {
      // World-to-cam on z only.
      *cam_depth_value = -xyz[2];
    }
  }

  // `cam_depth_value` is optional.
  void CamToImage(const Points3d &xyz, unsigned int height, unsigned int width, Points2i *image_xy, Points1d *cam_depth_value = nullptr) const {
    Points3d ndc;
    CamToFrustum(xyz, &ndc);
    // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glViewport.xhtml
    *image_xy = ((ndc.topRows<2>().array() + 1.0).colwise() * (Vec2{width, height} * 0.5).array()).cast<int>();
    image_xy->row(1) = height - image_xy->row(1).array() - 1;  // flip y.

    if (cam_depth_value) {
      // World-to-cam on z only.
      *cam_depth_value = -xyz.row(2);
    }
  }

  // `cam_depth_value` is optional.
  void WorldToImage(const Vec3 &xyz, unsigned int height, unsigned int width, Vec2i *image_xy, double *cam_depth_value = nullptr) const {
    Vec3 ndc;
    WorldToFrustum(xyz, &ndc);
    *image_xy = ((ndc.topRows<2>().array() + 1.0).colwise() * (Vec2{width, height} * 0.5).array()).cast<int>();
    image_xy->row(1) = height - image_xy->row(1).array() - 1;  // flip y.

    if (cam_depth_value) {
      // World-to-cam on z only.
      *cam_depth_value = -view_mat_.row(2) * xyz.homogeneous();
    }
  }

  // `cam_depth_value` is optional.
  void WorldToImage(const Points3d &xyz, unsigned int height, unsigned int width, Points2i *image_xy, Points1d *cam_depth_value = nullptr) const {
    Points3d ndc;
    WorldToFrustum(xyz, &ndc);
    *image_xy = ((ndc.topRows<2>().array() + 1.0).colwise() * (Vec2{width, height} * 0.5).array()).cast<int>();
    image_xy->row(1) = height - image_xy->row(1).array() - 1;  // flip y.

    if (cam_depth_value) {
      // World-to-cam on z only.
      *cam_depth_value = -view_mat_.row(2) * xyz.colwise().homogeneous();
    }
  }

  void ImageToWorld(const Points2i &xy, const Points1d &cam_depth, unsigned int height, unsigned int width, Points3d *out) const {
    Points3d cam;
    ImageToCam(xy, cam_depth, height, width, &cam);
    CamToWorld(cam, out);
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

  virtual void ImageToCam(const Points2i &xy, const Points1d &cam_depth, unsigned int height, unsigned int width, Points3d *out) const = 0;

  virtual const Mat44 &projection_mat() const = 0;
  virtual const Mat44 &projection_mat_inv() const = 0;
  virtual bool is_perspective() const = 0;

  // Returns false if this is orthographic. `x_fov` and `y_fov` are half of end-to-end angles, in radians.
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

  // Skip NDC and directly find camera coordinates.
  void ImageToCam(const Points2i &xy, const Points1d &cam_depth, unsigned int height, unsigned int width, Points3d *out) const override {
    Points2d xy_double = xy.cast<double>();
    Vec2 xy_scale{(frustum().right - frustum().left) / width, -(frustum().top - frustum().bottom) / height};
    Vec2 xy_offset{frustum().left, frustum().top};
    out->resize(3, xy.cols());

    out->topRows<2>() = ((xy_double.array() + 0.5).array().colwise() * xy_scale.array()).array().colwise() + xy_offset.array();
    out->bottomRows<1>() = -cam_depth;
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

  // Skip NDC and directly find camera coordinates.
  void ImageToCam(const Points2i &xy, const Points1d &cam_depth, unsigned int height, unsigned int width, Points3d *out) const override {
    Points2d xy_double = xy.cast<double>();
    Vec2 xy_scale{(frustum().right - frustum().left) / width, -(frustum().top - frustum().bottom) / height};
    Vec2 xy_offset{frustum().left, frustum().top};
    Points1d z = cam_depth.array() / frustum().near;
    out->resize(3, xy.cols());

    out->topRows<2>() = (((xy_double.array() + 0.5).array().colwise() * xy_scale.array()).array().colwise() + xy_offset.array()).array().rowwise() * z.array();
    out->bottomRows<1>() = -cam_depth;
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

void SaveCamera(const string &txt_filename, const Camera &camera) {
  int precision = 12;
  std::ofstream ofile;
  ofile.open(txt_filename, std::ios::out);

  string prefix = (camera.is_perspective() ? "P" : "O");

  ofile <<
        prefix << " " << std::setprecision(precision) <<
        camera.position()[0] << " " <<
        camera.position()[1] << " " <<
        camera.position()[2] << " " <<
        camera.viewing_direction()[0] << " " <<
        camera.viewing_direction()[1] << " " <<
        camera.viewing_direction()[2] << " " <<
        camera.up()[0] << " " <<
        camera.up()[1] << " " <<
        camera.up()[2] << " " <<
        camera.frustum().left << " " <<
        camera.frustum().right << " " <<
        camera.frustum().bottom << " " <<
        camera.frustum().top << " " <<
        camera.frustum().near << " " <<
        camera.frustum().far;

  ofile.close();
}

}

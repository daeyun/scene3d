//
// Created by daeyun on 12/20/17.
//

#include "camera.h"

namespace scene3d {

void SaveCamera(const string &txt_filename, const scene3d::Camera &camera) {
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

void SaveCameras(const string &txt_filename, const vector<scene3d::Camera *> &cameras) {
  int precision = 12;
  std::ofstream ofile;
  ofile.open(txt_filename, std::ios::out);

  for (int i = 0; i < cameras.size(); ++i) {
    const auto &camera = *cameras[i];
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
          camera.frustum().far << std::endl;
  }

  ofile.close();
}

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

FrustumParams ForceFixedAspectRatio(double hw_ratio, const FrustumParams &frustum) {
  double left = frustum.left;
  double right = frustum.right;
  double top = frustum.top;
  double bottom = frustum.bottom;

  double lr = std::abs(right - left);
  double bt = std::abs(top - bottom);
  double box_hw_ratio = bt / lr;

  if (box_hw_ratio < hw_ratio) {
    // image is squeezed horizontally.
    double padding = (hw_ratio * lr - bt) * 0.5;
    top += padding;
    bottom -= padding;
  } else {
    double padding = (bt - hw_ratio * lr) / hw_ratio * 0.5;
    right += padding;
    left -= padding;
  }

  FrustumParams ret = frustum;
  ret.left = left;
  ret.right = right;
  ret.top = top;
  ret.bottom = bottom;

  return ret;
}
}

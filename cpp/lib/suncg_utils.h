//
// Created by daeyun on 8/28/18.
//

#pragma once

#include <fstream>

#include "common.h"
#include "camera.h"

namespace scene3d {
namespace suncg {

struct CameraParams {
  Vec3 cam_eye;
  Vec3 cam_view_dir;
  Vec3 cam_up;
  double x_fov = 0;
  double y_fov = 0;
  double score = 0;  // scene coverage score. not used at the moment.
};
}  // end of namespace suncg

PerspectiveCamera MakeCamera(const suncg::CameraParams &params, double near, double far) {
  double hw_ratio = std::tan(params.y_fov) / std::tan(params.x_fov);
  FrustumParams frustum = MakePerspectiveFrustumParams(hw_ratio, params.x_fov, near, far);
  return {params.cam_eye, params.cam_eye + params.cam_view_dir, params.cam_up, frustum};
}

namespace suncg {
void ReadCameraFile(const string &filename, vector<suncg::CameraParams> *suncg_params) {
  LOGGER->info("Reading file {}", filename);

  std::ifstream source;
  source.open(filename, std::ios_base::in);
  if (!source) {
    throw std::runtime_error("Can't open file.");
  }

  for (std::string line; std::getline(source, line);) {
    if (line.empty()) {
      continue;
    }

    std::istringstream in(line);
    CameraParams cam;

    in >> cam.cam_eye[0] >> cam.cam_eye[1] >> cam.cam_eye[2];
    in >> cam.cam_view_dir[0] >> cam.cam_view_dir[1] >> cam.cam_view_dir[2];
    in >> cam.cam_up[0] >> cam.cam_up[1] >> cam.cam_up[2];
    in >> cam.x_fov >> cam.y_fov >> cam.score;

    LOGGER->info("camera {}, eye {}, {}, {}, fov {}, {}", suncg_params->size(), cam.cam_eye[0], cam.cam_eye[1], cam.cam_eye[2], cam.x_fov, cam.y_fov);

    cam.cam_view_dir.normalize();

    suncg_params->push_back(cam);
  }
}

// Write in the same format as SunCG.
void WriteCameraFile(const string &filename, const vector<PerspectiveCamera>& cameras) {
  int precision = 13;
  std::ofstream ofile;
  ofile.open(filename, std::ios::out);

  for (const auto &camera : cameras) {
    double fx, fy;
    camera.fov(&fx, &fy);
    ofile << std::setprecision(precision) <<
          camera.position()[0] << " " <<
          camera.position()[1] << " " <<
          camera.position()[2] << " " <<
          camera.viewing_direction()[0] << " " <<
          camera.viewing_direction()[1] << " " <<
          camera.viewing_direction()[2] << " " <<
          camera.up()[0] << " " <<
          camera.up()[1] << " " <<
          camera.up()[2] << " " <<
          fx << " " <<
          fy << " " <<
          0 << std::endl;
  }

  ofile.close();
}

void ReadCameraFile(const string &filename, vector<PerspectiveCamera> *cameras) {
  vector<suncg::CameraParams> params;
  ReadCameraFile(filename, &params);

  for (const auto &param : params) {
    cameras->push_back(MakeCamera(param, 0.01, 100));  // near, far

  }
}
}  // end of namespace suncg
}

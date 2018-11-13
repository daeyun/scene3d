#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>

#include "cxxopts.hpp"

#include "lib/file_io.h"
#include "lib/benchmark.h"
#include "lib/multi_layer_depth_renderer.h"
#include "lib/depth.h"
#include "lib/depth_render_utils.h"
#include "lib/common.h"
#include "lib/string_utils.h"

using namespace scene3d;

int main(int argc, const char **argv) {
  cxxopts::Options options("render_suncg", "Render multi-layer depth images");

  options.add_options()
      ("camera_filename", "File containing camera parameters, one per line.", cxxopts::value<string>())
      ("camera_index", "Which line to use, in the camera file. 0-indexed.", cxxopts::value<int>())
      ("mesh", "Path to source mesh file.", cxxopts::value<string>())
      ("out", "Path to output mesh file.", cxxopts::value<string>())
      ("help", "Display help.", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
  auto flags = options.parse(argc, argv);

  if (flags["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  // Initialize logging.
  spdlog::stdout_color_mt("console");

  vector<string> required_flags = {"camera_filename", "camera_index", "mesh", "out"};
  for (const string &name: required_flags) {
    if (!flags.count(name)) {
      LOGGER->error("No argument specified for required option --{}. See --help.", name);
      throw std::runtime_error("");
    }
  }

  const string camera_filename = flags["camera_filename"].as<string>();
  const int camera_index = flags["camera_index"].as<int>();
  const string mesh_filename = flags["mesh"].as<string>();
  const string out_filename = flags["out"].as<string>();

  Expects(Exists(camera_filename));
  Expects(Exists(mesh_filename));
  Expects(camera_index >= 0);
  Expects(EndsWith(out_filename, ".ply") or EndsWith(out_filename, ".obj"));

  double start_time = scene3d::TimeSinceEpoch<std::milli>();

  vector<unique_ptr<scene3d::Camera>> cameras;
  ReadCameras(camera_filename, &cameras);
  Expects(camera_index < cameras.size());

  scene3d::Camera *camera = cameras[camera_index].get();

  TriMesh mesh;

  ReadTriangles(mesh_filename,
                [&](const array<array<float, 3>, 3> triangle) {
                  mesh.AddTriangle(triangle[0], triangle[1], triangle[2]);
                });

  array<Plane, 6> planes;
  camera->WorldFrustumPlanes(&planes);

  // TODO: This should be refactored.
  TriMesh mesh0;
  mesh.TruncateBelow(planes[0], &mesh0);
  TriMesh mesh1;
  mesh0.TruncateBelow(planes[1], &mesh1);
  TriMesh mesh2;
  mesh1.TruncateBelow(planes[2], &mesh2);
  TriMesh mesh3;
  mesh2.TruncateBelow(planes[3], &mesh3);
  TriMesh out_mesh;
  mesh3.TruncateBelow(planes[4], &out_mesh);

  if (scene3d::EndsWith(out_filename, ".ply")) {
    WritePly(out_filename, out_mesh.faces, out_mesh.vertices, true);
  } else if (scene3d::EndsWith(out_filename, ".obj")) {
    WriteObj(out_filename, out_mesh.faces, out_mesh.vertices);
  } else {
    throw std::runtime_error("Invalid output file format.");
  }

  LOGGER->info("Elapsed (scene.Build): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);
  std::cout << "Output file: " << out_filename << std::endl;
}

#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>

#include "cxxopts.hpp"
#include "csv.h"

#include "lib/file_io.h"
#include "lib/benchmark.h"
#include "lib/suncg_utils.h"
#include "lib/multi_layer_depth_renderer.h"
#include "lib/depth.h"
#include "lib/depth_render_utils.h"
#include "lib/common.h"
#include "lib/meshdist.h"

using namespace scene3d;

int main(int argc, const char **argv) {
  cxxopts::Options options("render_suncg", "Render multi-layer depth images");

  options.add_options()
      ("camera_filename", "File containing camera parameters, one per line.", cxxopts::value<string>())
      ("out_dir", "Path to save ply files.", cxxopts::value<string>())
      ("obj", "Path to obj mesh file.", cxxopts::value<string>())
      ("json", "Path to house json file.", cxxopts::value<string>())
      ("dd_factor", "Depth discontinuity factor", cxxopts::value<float>()->default_value("10.0"))
      ("resample_height", "Rendered image height for background extraction", cxxopts::value<unsigned int>()->default_value("480"))
      ("resample_width", "Rendered image width for background extraction", cxxopts::value<unsigned int>()->default_value("640"))
      ("category", "Path to category mapping file. e.g. ModelCategoryMapping.csv", cxxopts::value<string>())
      ("save_objects", "Save the objects in the scene as a ply file.", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
      ("save_background", "Save background as a ply file.", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
      ("save_both", "Save a ply file containing both objects and background.", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
      ("help", "Display help.", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
  auto flags = options.parse(argc, argv);

  if (flags["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  // Initialize logging.
  spdlog::stdout_color_mt("console");

  vector<string> required_flags = {"obj", "json", "camera_filename", "category", "out_dir"};
  for (const string &name: required_flags) {
    if (!flags.count(name)) {
      LOGGER->error("No argument specified for required option --{}. See --help.", name);
      throw std::runtime_error("");
    }
  }

  const string obj_filename = flags["obj"].as<string>();
  const string out_dir = flags["out_dir"].as<string>();
  const string json_filename = flags["json"].as<string>();
  const string category_filename = flags["category"].as<string>();
  const string camera_filename = flags["camera_filename"].as<string>();
  const float dd_factor = flags["dd_factor"].as<float>();
  const unsigned int resample_height = flags["resample_height"].as<unsigned int>();
  const unsigned int resample_width = flags["resample_width"].as<unsigned int>();
  Expects(Exists(obj_filename));
  Expects(Exists(json_filename));
  Expects(Exists(category_filename));
  Expects(Exists(camera_filename));

  const bool save_objects = flags["save_objects"].as<bool>();
  const bool save_background = flags["save_background"].as<bool>();
  const bool save_both = flags["save_both"].as<bool>();

  if (!(save_objects | save_both | save_background)) {
    LOGGER->error("One of --save_objects, --save_background, --save_both must be specified.");
    throw std::runtime_error("");
  }

  double start_time;

  start_time = scene3d::TimeSinceEpoch<std::milli>();
  // Read the obj, json, and category mappings.
  auto scene = make_unique<suncg::Scene>(json_filename, obj_filename, category_filename);
  scene->Build();
  LOGGER->info("Elapsed (scene.Build): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);

  vector<unique_ptr<scene3d::Camera>> cameras;
  ReadCameras(camera_filename, &cameras);

  for (int camera_i = 0; camera_i < cameras.size(); ++camera_i) {
    scene3d::Camera *camera = cameras[camera_i].get();
    Expects(camera->is_perspective());

    start_time = scene3d::TimeSinceEpoch<std::milli>();
    TriMesh gt_mesh_background_only;
    TriMesh gt_mesh_object_only;

    ExtractFrustumMesh(scene.get(), *camera, resample_height, resample_width, dd_factor, &gt_mesh_background_only, &gt_mesh_object_only);
    LOGGER->info("Elapsed (ExtractFrustumMesh total): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);

    // Function that generates file names in `out_dir`.
    auto generate_filename = [&](int index, const string &suffix) -> string {
      Ensures(suffix.size() < 128);
      char buff[2048];
      snprintf(buff, sizeof(buff), "%06d_%s", index, suffix.c_str());
      auto ret = JoinPath(out_dir, string(buff));
      return ret;
    };

    auto save_mesh = [&](const string &suffix, const TriMesh &mesh) {
      const string out_filename = generate_filename(camera_i, suffix);
      WritePly(out_filename, mesh.faces, mesh.vertices, true);
      // NOTE: This line is important. The python script parses this line to determine which files were generated. Must start with "Output file: "
      std::cout << "Output file: " << out_filename << std::endl;
    };

    start_time = scene3d::TimeSinceEpoch<std::milli>();
    PrepareDir(out_dir);
    if (save_background) {
      save_mesh("bg.ply", gt_mesh_background_only);
    }
    if (save_objects) {
      save_mesh("objects.ply", gt_mesh_object_only);
    }
    if (save_both) {
      TriMesh both;
      both.AddMesh(gt_mesh_object_only);
      both.AddMesh(gt_mesh_background_only);
      save_mesh("all.ply", both);
    }
    LOGGER->info("Elapsed (Save): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);
  }
}

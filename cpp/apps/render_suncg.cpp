#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>

#include "cxxopts.hpp"
#include "csv.h"

#include "lib/file_io.h"
#include "lib/suncg_utils.h"
#include "lib/multi_layer_depth_renderer.h"
#include "lib/depth.h"
#include "lib/depth_render_utils.h"

using namespace scene3d;

int main(int argc, const char **argv) {
  cxxopts::Options options("render_suncg", "Render multi-layer depth images");

  options.add_options()
      ("h,height", "Rendered image height.", cxxopts::value<int>()->default_value("240"))
      ("w,width", "Rendered image width.", cxxopts::value<int>()->default_value("320"))
      ("cameras", "Path to txt file containing camera parameters.", cxxopts::value<string>())
      ("out_dir", "Path to output directory.", cxxopts::value<string>())
      ("obj", "Path to obj mesh file.", cxxopts::value<string>())
      ("json", "Path to obj mesh file.", cxxopts::value<string>())
      ("category", "Path to category mapping file. e.g. ModelCategoryMapping.csv", cxxopts::value<string>())
      ("help", "Display help.", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
  auto flags = options.parse(argc, argv);

  if (flags["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  // Initialize logging.
  spdlog::stdout_color_mt("console");

  // Check required flags.
  vector<string> required_flags = {"obj", "json", "cameras", "category", "out_dir"};
  for (const string &name: required_flags) {
    if (!flags.count(name)) {
      LOGGER->error("No argument specified for required option --{}. See --help.", name);
      throw std::runtime_error("");
    }
  }
  const string obj_filename = flags["obj"].as<string>();
  const string json_filename = flags["json"].as<string>();
  const string camera_filename = flags["cameras"].as<string>();
  const string category_filename = flags["category"].as<string>();
  const string out_dir = flags["out_dir"].as<string>();
  const unsigned int width = static_cast<const unsigned int>(flags["width"].as<int>());
  const unsigned int height = static_cast<const unsigned int>(flags["height"].as<int>());

  Ensures(width < 1e5);
  Ensures(height < 1e5);

  // Read the obj, json, and category mappings.
  auto scene = make_unique<suncg::Scene>(json_filename, obj_filename, category_filename);
  scene->Build();

  // Read the camera file.
  std::vector<PerspectiveCamera> cameras;
  suncg::ReadCameraFile(camera_filename, &cameras);
  LOGGER->info("{} cameras in {}", cameras.size(), camera_filename);

  scene3d::RayTracer ray_tracer(scene->faces, scene->vertices);
  ray_tracer.PrintStats();

  for (int camera_i = 0; camera_i < cameras.size(); ++camera_i) {
    PerspectiveCamera camera = cameras[camera_i];
    LOGGER->info("Rendering camera {}", camera_i);

    auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
        &ray_tracer,
        &camera,
        width,
        height,
        scene.get()
    );

    auto ml_depth = MultiLayerImage<float>(height, width);
    auto ml_prim_ids = MultiLayerImage<uint32_t>(height, width);
    RenderMultiLayerDepthImage(&renderer, &ml_depth, &ml_prim_ids);

  }

  PrepareDir(out_dir);

  LOGGER->info("OK");
  return 0;
}

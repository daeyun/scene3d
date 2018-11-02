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

using namespace scene3d;

int main(int argc, const char **argv) {
  cxxopts::Options options("render_suncg", "Render multi-layer depth images");

  options.add_options()
      ("cameras", "Path to txt file containing camera parameters.", cxxopts::value<string>())
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
  vector<string> required_flags = {"obj", "json", "cameras", "category"};
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

  // Read the obj, json, and category mappings.
  auto scene = make_unique<suncg::Scene>(json_filename, obj_filename, category_filename);
  scene->Build();

  // Read the camera file.
  vector<unique_ptr<scene3d::Camera>> cameras;
  ReadCameras(camera_filename, &cameras);

  // All of them should be orthographic.
  Expects(!cameras[0]->is_perspective());  // Sanity check. This is not a strict requirement.

  scene3d::RayTracer ray_tracer(scene->faces, scene->vertices);
  ray_tracer.PrintStats();

  for (int camera_i = 0; camera_i < cameras.size(); ++camera_i) {
    Camera *overhead_cam = cameras[camera_i].get();
    LOGGER->info("Rendering camera {}", camera_i);

    {
      // 2. Overhead camera rendering
      // -------------------------------------------------------
      const unsigned int kOverheadHeight = 300;
      const unsigned int kOverheadWidth = 300;
      auto overhead_renderer = scene3d::SunCgMultiLayerDepthRenderer(
          &ray_tracer,
          overhead_cam,
          kOverheadWidth,
          kOverheadHeight,
          scene.get()
      );
      overhead_renderer.set_overhead_rendering_mode(true);
      MultiLayerImage<float> ml_depth_overhead;
      MultiLayerImage<uint32_t> ml_prim_ids_overhead;
      RenderMultiLayerDepthImage(&overhead_renderer, &ml_depth_overhead, &ml_prim_ids_overhead);
      MultiLayerImage<float> out_ml_depth_overhead;
      MultiLayerImage<uint16_t> out_ml_model_indices_overhead;
      MultiLayerImage<uint32_t> out_ml_prim_ids_overhead;
      GenerateMultiDepthExample(scene.get(), ml_depth_overhead, ml_prim_ids_overhead, &out_ml_depth_overhead, &out_ml_model_indices_overhead, &out_ml_prim_ids_overhead);

      // Detect main floor.
      vector<float> overhead_depth0_values;
      out_ml_depth_overhead.ExtractLayer(0, &overhead_depth0_values);
      vector<uint16_t> overhead_model0_values;
      out_ml_model_indices_overhead.ExtractLayer(0, &overhead_model0_values);
      vector<float> overhead_floor_values;
      for (int i = 0; i < overhead_depth0_values.size(); ++i) {
        if (std::isfinite(overhead_depth0_values[i]) && overhead_model0_values[i] == 3) {  // model_id 3 is floor.
          overhead_floor_values.push_back(overhead_depth0_values[i]);
        }
      }
      float floor_depth = [&](vector<float> &v) -> float {
        if (v.empty()) {
          // There is no floor pixel in this image. Assume floor is at 0.
          // This is very rare, but happens at least once in 16cf44a3f11809f4098d7a306eadcba4/000007
          return static_cast<float>(overhead_cam->position()[1]);
        }
        size_t n = v.size() / 2;
        nth_element(v.begin(), v.begin() + n, v.end());
        return v[n];
      }(overhead_floor_values);  // Find median. Mutates `overhead_floor_values`.

      auto floor_height = static_cast<float>(overhead_cam->position()[1] - floor_depth);
      LOGGER->info("Floor height: {}", floor_height);
    }
  }

  LOGGER->info("OK");
  return 0;
}

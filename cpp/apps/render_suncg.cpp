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

  auto timer0 = SimpleTimer("reading and parsing");
  auto timer4 = SimpleTimer("ray tracer initialization");
  auto timer1 = SimpleTimer("ray tracing");
  auto timer2 = SimpleTimer("LDI labeling");
  auto timer5 = SimpleTimer("overhead camera");
  auto timer3 = SimpleTimer("saving");

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

  timer0.Tic();
  // Read the obj, json, and category mappings.
  auto scene = make_unique<suncg::Scene>(json_filename, obj_filename, category_filename);
  scene->Build();
  timer0.Toc();

  timer4.Tic();
  // Read the camera file.
  std::vector<PerspectiveCamera> cameras;
  suncg::ReadCameraFile(camera_filename, &cameras);
  LOGGER->info("{} cameras in {}", cameras.size(), camera_filename);

  scene3d::RayTracer ray_tracer(scene->faces, scene->vertices);
  ray_tracer.PrintStats();
  timer4.Toc();

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

    timer1.Tic();
    MultiLayerImage<float> ml_depth;
    MultiLayerImage<uint32_t> ml_prim_ids;
    RenderMultiLayerDepthImage(&renderer, &ml_depth, &ml_prim_ids);
    timer1.Toc();

    timer2.Tic();
    MultiLayerImage<float> out_ml_depth;
    MultiLayerImage<uint16_t> out_ml_model_indices;
    MultiLayerImage<uint32_t> out_ml_prim_ids;
    GenerateMultiDepthExample(scene.get(), ml_depth, ml_prim_ids, &out_ml_depth, &out_ml_model_indices, &out_ml_prim_ids);
    timer2.Toc();

    // Overhead camera and rendering
    timer5.Tic();
    AABB average_bounding_box;
    vector<AABB> candidate_boxes;
    PerspectiveCamera aligner(camera);
    unsigned int overhead_height = 300;
    unsigned int overhead_width = 300;
    double overhead_hw_ratio = static_cast<double>(overhead_height) / overhead_width;
    unique_ptr<OrthographicCamera>
        overhead_cam = ComputeOverheadCamera(out_ml_depth, out_ml_model_indices, out_ml_prim_ids, camera, scene.get(), overhead_hw_ratio, &average_bounding_box, &candidate_boxes, &aligner);

    auto overhead_renderer = scene3d::SunCgMultiLayerDepthRenderer(
        &ray_tracer,
        overhead_cam.get(),
        overhead_width,
        overhead_height,
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
      size_t n = v.size() / 2;
      nth_element(v.begin(), v.begin() + n, v.end());
      return v[n];
    }(overhead_floor_values);  // Find median.
    LOGGER->info("floor depth: {}", floor_depth);

    out_ml_depth_overhead.Transform([floor_depth](float d) -> float {
      float floor_height = floor_depth - d;
      if (floor_height < 0) {
        floor_height = 0;
      }
      return floor_height;
    });

    timer5.Toc();

    timer3.Tic();
    auto generate_filename = [&](int index, const string &suffix, const string &extension) -> string {
      Ensures(suffix.size() < 128);
      Ensures(extension.size() < 128);
      Ensures(extension[0] != '.');
      char buff[2048];
      snprintf(buff, sizeof(buff), "%06d_%s.%s", index, suffix.c_str(), extension.c_str());
      auto ret = JoinPath(out_dir, string(buff));

      // NOTE: This line is important. The python script parses this line to determine which files were generated. Must start with "Output file: "
      std::cout << "Output file: " << ret << std::endl;
      return ret;
    };

    const unsigned int kNumLayers = 4;
    out_ml_depth.Save(generate_filename(camera_i, "ldi", "bin"), kNumLayers);
    out_ml_model_indices.Save(generate_filename(camera_i, "model", "bin"), kNumLayers);
    out_ml_depth_overhead.Save(generate_filename(camera_i, "ldi-o", "bin"), kNumLayers);
    out_ml_model_indices_overhead.Save(generate_filename(camera_i, "model-o", "bin"), kNumLayers);

    SaveCameras(generate_filename(camera_i, "cam", "txt"), vector<Camera *>{&camera, overhead_cam.get()});
    SaveAABB(generate_filename(camera_i, "aabb", "txt"), vector<AABB>{average_bounding_box, candidate_boxes[0], candidate_boxes[1], candidate_boxes[2]});

    timer3.Toc();
  }

  LOGGER->info("Elapsed time ({}) : {:.1f}", timer0.name(), timer0.Duration<std::milli>());
  LOGGER->info("Elapsed time ({}) : {:.1f}", timer4.name(), timer4.Duration<std::milli>());
  LOGGER->info("Elapsed time ({}) : {:.1f}", timer1.name(), timer1.Duration<std::milli>());
  LOGGER->info("Elapsed time ({}) : {:.1f}", timer2.name(), timer2.Duration<std::milli>());
  LOGGER->info("Elapsed time ({}) : {:.1f}", timer5.name(), timer5.Duration<std::milli>());
  LOGGER->info("Elapsed time ({}) : {:.1f}", timer3.name(), timer3.Duration<std::milli>());
  LOGGER->info("OK");
  return 0;
}

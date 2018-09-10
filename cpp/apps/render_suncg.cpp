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
  auto timer6 = SimpleTimer("object-centered");
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

  // Function that generates file names in `out_dir`.
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
  const unsigned int kNumOutputLayers = 4;

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

    {
      // 1. Input camera rendering
      // -------------------------------------------------------
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

      // 2. Overhead camera rendering
      // -------------------------------------------------------
      timer5.Tic();
      AABB average_bounding_box;
      vector<AABB> candidate_boxes;
      PerspectiveCamera aligner(camera);
      const unsigned int kOverheadHeight = 300;
      const unsigned int kOverheadWidth = 300;
      double overhead_hw_ratio = static_cast<double>(kOverheadHeight) / kOverheadWidth;
      unique_ptr<OrthographicCamera>
          overhead_cam = ComputeOverheadCamera(out_ml_depth, out_ml_model_indices, out_ml_prim_ids, camera, scene.get(), overhead_hw_ratio, &average_bounding_box, &candidate_boxes, &aligner);

      auto overhead_renderer = scene3d::SunCgMultiLayerDepthRenderer(
          &ray_tracer,
          overhead_cam.get(),
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
        size_t n = v.size() / 2;
        nth_element(v.begin(), v.begin() + n, v.end());
        return v[n];
      }(overhead_floor_values);  // Find median.
      LOGGER->info("floor depth: {}", floor_depth);

      out_ml_depth_overhead.Transform([floor_depth](size_t index, size_t l, float d) -> float {
        return std::max(floor_depth - d, 0.0f);
      });

      timer5.Toc();

      timer3.Tic();
      // Save as compressed binary files.
      out_ml_depth.Save(generate_filename(camera_i, "ldi", "bin"), kNumOutputLayers);
      out_ml_model_indices.Save(generate_filename(camera_i, "model", "bin"), kNumOutputLayers);
      out_ml_depth_overhead.Save(generate_filename(camera_i, "ldi-o", "bin"), kNumOutputLayers);
      out_ml_model_indices_overhead.Save(generate_filename(camera_i, "model-o", "bin"), kNumOutputLayers);
      // Save cameras.
      SaveCameras(generate_filename(camera_i, "cam", "txt"), vector<Camera *>{&camera, overhead_cam.get()});
      SaveAABB(generate_filename(camera_i, "aabb", "txt"), vector<AABB>{average_bounding_box, candidate_boxes[0], candidate_boxes[1], candidate_boxes[2]});
      timer3.Toc();
    }

    {
      // 3. Object-centered thickness
      // -------------------------------------------------------
      timer6.Tic();
      MultiLayerImage<float> ml_depth_objcentered;
      MultiLayerImage<uint32_t> ml_prim_ids_objcentered;
      RenderObjectCenteredMultiLayerDepthImage(&renderer, &ml_depth_objcentered, &ml_prim_ids_objcentered);

      MultiLayerImage<float> out_ml_depth_objcentered;
      MultiLayerImage<uint16_t> out_ml_model_indices_objcentered;
      MultiLayerImage<uint32_t> out_ml_prim_ids_objcentered;
      GenerateMultiDepthExample(scene.get(), ml_depth_objcentered, ml_prim_ids_objcentered, &out_ml_depth_objcentered, &out_ml_model_indices_objcentered, &out_ml_prim_ids_objcentered);

      // TODO: This part can be refactored/simplified.
      out_ml_depth_objcentered.Transform([&](size_t index, size_t l, float d) -> float {
        if (l == 1) {
          if (std::isfinite(d)) {
            float ret = d - out_ml_depth_objcentered.at(index, 0);
            Ensures(ret >= -1e-5);  // in case of numerical error.
            return std::max(ret, 0.0f);
          }
        }
        return d;
      });
      Image<float> object_centered_instance_thickness;
      out_ml_depth_objcentered.ExtractLayer(1, &object_centered_instance_thickness);

      // Surface normals.
      Image<uint32_t> first_layer_prim_id;
      MultiLayerImage<float> surface_normals(object_centered_instance_thickness.height(), object_centered_instance_thickness.width(), NAN);
      ml_prim_ids_objcentered.ExtractLayer(0, &first_layer_prim_id);
      for (int y = 0; y < object_centered_instance_thickness.height(); ++y) {
        for (int x = 0; x < object_centered_instance_thickness.width(); ++x) {
          if (std::isfinite(object_centered_instance_thickness.at(y, x))) {
            const array<float, 3> &normal = scene->PrimNormal(first_layer_prim_id.at(y, x));
            Vec3 cam_normal;
            camera.WorldToCamNormal(Vec3{normal[0], normal[1], normal[2]}, &cam_normal);
            Ensures(std::isfinite(cam_normal[0]));
            Ensures(std::isfinite(cam_normal[1]));
            Ensures(std::isfinite(cam_normal[2]));
            cam_normal.normalize();
            if (cam_normal[2] < 0) {
              cam_normal *= -1;
            }
            surface_normals.values(y, x)->push_back(cam_normal[0]);
            surface_normals.values(y, x)->push_back(cam_normal[1]);
            surface_normals.values(y, x)->push_back(cam_normal[2]);
          }
        }
      }
      timer6.Toc();

      timer3.Tic();
      // Save as compressed binary files.
      object_centered_instance_thickness.Save(generate_filename(camera_i, "oit", "bin"));
      surface_normals.Save(generate_filename(camera_i, "n", "bin"), 3);
      timer3.Toc();
    }
  }

  LOGGER->info("Elapsed time ({}) : {:.1f}", timer0.name(), timer0.TotalElapsed<std::milli>() / cameras.size());
  LOGGER->info("Elapsed time ({}) : {:.1f}", timer4.name(), timer4.TotalElapsed<std::milli>() / cameras.size());
  LOGGER->info("Elapsed time ({}) : {:.1f}", timer1.name(), timer1.TotalElapsed<std::milli>() / cameras.size());
  LOGGER->info("Elapsed time ({}) : {:.1f}", timer2.name(), timer2.TotalElapsed<std::milli>() / cameras.size());
  LOGGER->info("Elapsed time ({}) : {:.1f}", timer5.name(), timer5.TotalElapsed<std::milli>() / cameras.size());
  LOGGER->info("Elapsed time ({}) : {:.1f}", timer6.name(), timer6.TotalElapsed<std::milli>() / cameras.size());
  LOGGER->info("Elapsed time ({}) : {:.1f}", timer3.name(), timer3.TotalElapsed<std::milli>() / cameras.size());
  LOGGER->info("OK");
  return 0;
}

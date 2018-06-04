#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>
#include <lib/multi_layer_depth_renderer.h>

#include "cxxopts.hpp"

#include "lib/file_io.h"

struct CameraParams {
  Vec3 cam_eye;
  Vec3 cam_view_dir;
  Vec3 cam_up;
  double x_fov;
  double y_fov;
  double score;  // scene coverage score. not used at the moment.
};

int main(int argc, const char **argv) {
  cxxopts::Options options("render", "Render multi-layer depth images");
  options.add_options()
      ("h,height", "Rendered image height.", cxxopts::value<int>()->default_value("480"))
      ("w,width", "Rendered image width.", cxxopts::value<int>()->default_value("640"))
      ("m,max_hits", "Maximum number of lay hits. Set to 0 for unlimited.", cxxopts::value<int>()->default_value("0"))
      ("obj", "Path to obj mesh file.", cxxopts::value<string>())
      ("cameras", "Path to txt file containing camera parameters.", cxxopts::value<string>())
      ("out_dir", "Path to output directory.", cxxopts::value<string>())
      ("help", "Display help.", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
  auto flags = options.parse(argc, argv);

  if (flags["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  vector<string> required_flags = {"obj", "cameras", "out_dir"};
  for (const string &name: required_flags) {
    if (!flags.count(name)) {
      throw std::runtime_error("No argument specified for required option --" + name + ". See --help.");
    }
  }

  spdlog::stdout_color_mt("console");

  std::string obj_filename = flags["obj"].as<std::string>();
  std::string camera_filename = flags["cameras"].as<std::string>();
  std::string out_dir = flags["out_dir"].as<std::string>();

  if (out_dir.back() == '/') {
    out_dir.pop_back();
  }

  std::vector<std::array<unsigned int, 3>> faces;
  std::vector<std::array<float, 3>> vertices;
  std::vector<int> prim_id_to_node_id;
  std::vector<std::string> prim_id_to_node_name;

  LOGGER->info("Reading file {}", camera_filename);

  std::ifstream source;
  source.open(camera_filename, std::ios_base::in);
  if (!source) {
    throw std::runtime_error("Can't open file.");
  }

  std::vector<CameraParams> suncg_cameras;
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

    LOGGER->info("camera {}, eye {}, {}, {}, fov {}, {}", suncg_cameras.size(), cam.cam_eye[0], cam.cam_eye[1], cam.cam_eye[2], cam.x_fov, cam.y_fov);

    cam.cam_view_dir.normalize();

    suncg_cameras.push_back(cam);
  }

  LOGGER->info("Reading file {}", obj_filename);

  bool ok = scene3d::ReadFacesAndVertices(obj_filename, &faces, &vertices, &prim_id_to_node_id, &prim_id_to_node_name);

  // Sanity check.
  Ensures(faces.size() == prim_id_to_node_id.size());
  Ensures(faces.size() == prim_id_to_node_name.size());

  LOGGER->info("{} faces, {} vertices", faces.size(), vertices.size());

  scene3d::RayTracer ray_tracer(faces, vertices);
  ray_tracer.PrintStats();

  const size_t width = static_cast<size_t>(flags["width"].as<int>());
  const size_t height = static_cast<size_t>(flags["height"].as<int>());
  const size_t max_hits = static_cast<size_t>(flags["max_hits"].as<int>());

  for (int camera_i = 0; camera_i < suncg_cameras.size(); ++camera_i) {
    CameraParams suncg_cam = suncg_cameras[camera_i];
    LOGGER->info("Rendering camera {}", camera_i);

    auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
        &ray_tracer,
        suncg_cam.cam_eye,
        suncg_cam.cam_view_dir,
        suncg_cam.cam_up,
        suncg_cam.x_fov,
        suncg_cam.y_fov,
        width,
        height,
        max_hits,
        prim_id_to_node_name
    );

    vector<vector<unique_ptr<vector<float>>>> grid_depth_values(height);
    vector<float> background_values;
    for (int y = 0; y < height; y++) {
      grid_depth_values[y].resize(width);
    }

    size_t num_layers = 0;
    bool found_at_least_one_backgrond_value = false;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        auto depth_values = make_unique<vector<float>>();
        int depth_value_index = renderer.depth_values(x, y, depth_values.get());
        bool found_background = depth_value_index >= 0;

        num_layers = std::max(num_layers, depth_values->size());
        if (found_background) {
          background_values.push_back(depth_values->at(static_cast<size_t>(depth_value_index)));
          found_at_least_one_backgrond_value = true;
        } else {
          background_values.push_back(NAN);
        }
        grid_depth_values[y][x] = move(depth_values);
      }
    }

    LOGGER->info("Num layers found: {}", num_layers);

    vector<vector<float>> depth_images;
    // `max_hits` comes from command line argument. If it was set to 0 (unlimited), the number of depth images will be `num_layers`.
    const size_t n = (max_hits == 0) ? num_layers : std::min(max_hits, num_layers);

    if (n == 0) {
      LOGGER->warn("Zero ray hits. No image saved for camera {}", camera_i);
      continue;
    }

    depth_images.resize(n);

    for (int i = 0; i < n; ++i) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          // Populate the depth image buffer in row-major order.
          if (i < grid_depth_values[y][x]->size()) {
            depth_images[i].push_back(grid_depth_values[y][x]->at(static_cast<size_t>(i)));
          } else {
            depth_images[i].push_back(NAN);
          }
        }
      }
    }

    // Buffer of data to save. Contains (N, H, W) tensor data.
    vector<float> all_depth_values;
    all_depth_values.reserve(height * width * n);
    for (const auto &depth_image : depth_images) {
      all_depth_values.insert(all_depth_values.end(), depth_image.begin(), depth_image.end());
    }

    char buff[2048];
    snprintf(buff, sizeof(buff), "%s/%06d", out_dir.c_str(), camera_i);
    string out_filename = std::string(buff) + ".bin";

    const vector<int> shape{static_cast<int>(n), static_cast<int>(height), static_cast<int>(width)};
    scene3d::SerializeTensor<float>(out_filename, all_depth_values.data(), shape);

    // NOTE: This line is important. The python script parses this line to determine which files were generated. Must start with "Output file: "
    std::cout << "Output file: " << out_filename << std::endl;

    if (found_at_least_one_backgrond_value) {
      snprintf(buff, sizeof(buff), "%s/%06d", out_dir.c_str(), camera_i);
      string out_filename_bg = std::string(buff) + "_bg.bin";
      scene3d::SerializeTensor<float>(out_filename_bg, background_values.data(), {height, width});
      std::cout << "Output file: " << out_filename_bg << std::endl;
    }
  }

  LOGGER->info("OK");
  return 0;
}

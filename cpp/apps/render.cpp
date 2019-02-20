#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>

#include "spdlog/spdlog.h"
#include "cxxopts.hpp"
#include "gsl/gsl_assert"

#include "lib/file_io.h"
#include "lib/common.h"
#include "lib/camera.h"
#include "lib/ray_mesh_intersection.h"

struct CameraParams {
  Vec3 cam_eye;
  Vec3 cam_view_dir;
  Vec3 cam_up;
  Mat33 K;
};

const std::vector<string> background_name_substrings{"Floor", "Wall", "Ceiling", "Room", "Level", "floor", "background"};
const std::set<std::string> suncg_doors_and_windows
    {"122", "126", "133", "209", "210", "211", "212", "213", "214", "246", "247", "326", "327", "331", "361", "73", "752", "753", "754", "755", "756", "757", "758", "759", "760", "761", "762", "763",
     "764", "765", "766", "767", "768", "769", "770", "771", "s__1276", "s__1762", "s__1763", "s__1764", "s__1765", "s__1766", "s__1767", "s__1768", "s__1769", "s__1770", "s__1771", "s__1772",
     "s__1773", "s__2010", "s__2011", "s__2012", "s__2013", "s__2014", "s__2015", "s__2016", "s__2017", "s__2019"};

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
  std::vector<int> node_id_for_each_face;
  std::vector<std::string> node_name_for_each_face;

  LOGGER->info("Reading file {}", camera_filename);

  std::ifstream source;
  source.open(camera_filename, std::ios_base::in);
  if (!source) {
    throw std::runtime_error("Can't open file.");
  }

  std::vector<CameraParams> cam_params;
  for (std::string line; std::getline(source, line);) {
    if (line.empty()) {
      continue;
    }

    std::istringstream in(line);
    CameraParams cam;

    in >> cam.cam_eye[0] >> cam.cam_eye[1] >> cam.cam_eye[2];
    in >> cam.cam_view_dir[0] >> cam.cam_view_dir[1] >> cam.cam_view_dir[2];
    in >> cam.cam_up[0] >> cam.cam_up[1] >> cam.cam_up[2];

    double fx, fy, cx, cy;
    in >> fx >> fy >> cx >> cy;

    cam.K.setIdentity();
    cam.K(0, 0) = fx;
    cam.K(1, 1) = fy;
    cam.K(0, 2) = cx;
    cam.K(1, 2) = cy;

    std::stringstream Ks;
    Ks << cam.K;

    LOGGER->info("camera {}, eye {}, {}, {}, K:\n{}", cam_params.size(), cam.cam_eye[0], cam.cam_eye[1], cam.cam_eye[2], Ks.str());

    cam.cam_view_dir.normalize();

    cam_params.push_back(cam);
  }

  LOGGER->info("Reading file {}", obj_filename);

  bool ok = scenecompletion::ReadFacesAndVertices(obj_filename, &faces, &vertices, &node_id_for_each_face, &node_name_for_each_face);

  // Sanity check.
  Ensures(faces.size() == node_id_for_each_face.size());
  Ensures(faces.size() == node_name_for_each_face.size());

  LOGGER->info("{} faces, {} vertices", faces.size(), vertices.size());

  scenecompletion::RayTracer ray_tracer(faces, vertices);

  ray_tracer.PrintStats();

  // Routine to determine if a triangle belongs to background.
  auto is_background = [&](int face_index) -> bool {
    Expects(face_index < node_name_for_each_face.size());
    auto node_name = node_name_for_each_face[face_index];
    bool ret = false;
    for (const string &substr: background_name_substrings) {
      if (node_name.find(substr) != std::string::npos) {
        ret = true;
        break;
      }
    }
    if (!ret and node_name.find("Model#") != std::string::npos) {
      string model_id = node_name.substr(node_name.find('#') + 1);
      if (suncg_doors_and_windows.find(model_id) != suncg_doors_and_windows.end()) {
        // If the model id matches a window or a door, it is the background.
        ret = true;
      }
    }
    return ret;
  };

  for (int camera_i = 0; camera_i < cam_params.size(); ++camera_i) {
    CameraParams cam_params_i = cam_params[camera_i];
    LOGGER->info("Rendering camera {}", camera_i);

    Vec3 cam_eye = cam_params_i.cam_eye;
    Vec3 cam_view_dir = cam_params_i.cam_view_dir;
    Vec3 cam_up = cam_params_i.cam_up;
    Mat33 K_inv = cam_params_i.K.inverse();

    scenecompletion::FrustumParams frustum;
    frustum.far = 10000;
    frustum.near = 0.001;

    auto camera = scenecompletion::PerspectiveCamera(cam_eye, cam_eye + cam_view_dir, cam_up, frustum);

    const size_t image_width = static_cast<size_t>(flags["width"].as<int>());
    const size_t image_height = static_cast<size_t>(flags["height"].as<int>());

    const Vec3 cam_ray_origin{0, 0, 0};

    Vec3 ray_origin;
    camera.CamToWorld(cam_ray_origin, &ray_origin);

    const size_t max_hits = static_cast<size_t>(flags["max_hits"].as<int>());

    vector<vector<unique_ptr<vector<float>>>> grid_depth_values(image_height);
    vector<float> background_values;
    for (int y = 0; y < image_height; y++) {
      grid_depth_values[y].resize(image_width);
    }

    size_t num_layers = 0;
    bool found_at_least_one_backgrond_value = false;

    for (int y = 0; y < image_height; y++) {
      for (int x = 0; x < image_width; x++) {
        Vec3 cam_ray_direction = K_inv * Vec3(x + 0.5, y + 0.5, 1);
        cam_ray_direction.normalize();

        Vec3 ray_direction;
        camera.CamToWorldNormal(cam_ray_direction, &ray_direction);

        // Stack of depth values. e.g.  [FG, O1, O2, ... ,BG]. Can be empty if the ray hits nothing.
        auto depth_values = make_unique<vector<float>>();
        bool found_background = false;

        // Depth values are collected in the callback function, in the order traversed.
        ray_tracer.Traverse(ray_origin, ray_direction, [&](float t, float u, float v, unsigned int prim_id) -> bool {
          // Convert ray displacement to depth.
          float d = t * cam_view_dir.dot(ray_direction);
          depth_values->push_back(d);

          // Stop traversal if the triangle ID corresponds to a background.
          found_background = is_background(prim_id);
          return !found_background;
        });

        num_layers = std::max(num_layers, depth_values->size());
        if (found_background) {
          background_values.push_back(depth_values->at(depth_values->size() - 1));
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
      for (int y = 0; y < image_height; y++) {
        for (int x = 0; x < image_width; x++) {
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
    all_depth_values.reserve(image_height * image_width * n);
    for (const auto &depth_image : depth_images) {
      all_depth_values.insert(all_depth_values.end(), depth_image.begin(), depth_image.end());
    }

    char buff[2048];
    snprintf(buff, sizeof(buff), "%s/%06d", out_dir.c_str(), camera_i);
    string out_filename = std::string(buff) + ".bin";

    const vector<int> shape{static_cast<int>(n), static_cast<int>(image_height), static_cast<int>(image_width)};
    scenecompletion::SerializeTensor<float>(out_filename, all_depth_values.data(), shape);

    // NOTE: This line is important. The python script parses this line to determine which files were generated. Must start with "Output file: "
    std::cout << "Output file: " << out_filename << std::endl;

    if (found_at_least_one_backgrond_value) {
      snprintf(buff, sizeof(buff), "%s/%06d", out_dir.c_str(), camera_i);
      string out_filename_bg = std::string(buff) + "_bg.bin";
      scenecompletion::SerializeTensor<float>(out_filename_bg, background_values.data(), {image_height, image_width});
      std::cout << "Output file: " << out_filename_bg << std::endl;
    }
  }

  LOGGER->info("OK");
  return 0;
}

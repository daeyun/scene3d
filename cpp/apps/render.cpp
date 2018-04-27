#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>

#include "spdlog/spdlog.h"
#include "nanort.h"
#include "cxxopts.hpp"
#include "gsl/gsl_assert"

#include "lib/file_io.h"
#include "lib/common.h"
#include "lib/camera.h"

struct SuncgCamera {
  Vec3 cam_eye;
  Vec3 cam_view_dir;
  Vec3 cam_up;
  double x_fov;
  double y_fov;
  double score;  // scene coverage score
};

const std::vector<string> background_name_substrings{"Floor", "Wall", "Ceiling", "Room", "Level"};
const std::set<std::string> suncg_doors_and_windows
    {"122", "126", "133", "209", "210", "211", "212", "213", "214", "246", "247", "326", "327", "331", "361", "73", "752", "753", "754", "755", "756", "757", "758", "759", "760", "761", "762", "763",
     "764", "765", "766", "767", "768", "769", "770", "771", "s__1276", "s__1762", "s__1763", "s__1764", "s__1765", "s__1766", "s__1767", "s__1768", "s__1769", "s__1770", "s__1771", "s__1772",
     "s__1773", "s__2010", "s__2011", "s__2012", "s__2013", "s__2014", "s__2015", "s__2016", "s__2017", "s__2019"};

int main(int argc, const char **argv) {
  cxxopts::Options options("render", "Render multi-layer depth images");
  options.add_options()
      ("h,height", "Rendered image height.", cxxopts::value<int>()->default_value("480"))
      ("w,width", "Rendered image width.", cxxopts::value<int>()->default_value("640"))
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

  std::vector<SuncgCamera> suncg_cameras;
  for (std::string line; std::getline(source, line);) {
    if (line.empty()) {
      continue;
    }

    std::istringstream in(line);
    SuncgCamera cam;

    in >> cam.cam_eye[0] >> cam.cam_eye[1] >> cam.cam_eye[2];
    in >> cam.cam_view_dir[0] >> cam.cam_view_dir[1] >> cam.cam_view_dir[2];
    in >> cam.cam_up[0] >> cam.cam_up[1] >> cam.cam_up[2];
    in >> cam.x_fov >> cam.y_fov >> cam.score;

    LOGGER->info("camera {}, eye {}, {}, {}, fov {}, {}", suncg_cameras.size(), cam.cam_eye[0], cam.cam_eye[1], cam.cam_eye[2], cam.x_fov, cam.y_fov);

    cam.cam_view_dir.normalize();

    suncg_cameras.push_back(cam);
  }

  LOGGER->info("Reading file {}", obj_filename);

  bool ok = scenecompletion::ReadFacesAndVertices(obj_filename, &faces, &vertices, &node_id_for_each_face, &node_name_for_each_face);

  // Sanity check.
  Ensures(faces.size() == node_id_for_each_face.size());
  Ensures(faces.size() == node_name_for_each_face.size());

  LOGGER->info("{} faces, {} vertices", faces.size(), vertices.size());

  nanort::TriangleMesh<float> triangle_mesh(vertices.data()->data(), faces.data()->data(), sizeof(float) * 3);
  nanort::TriangleSAHPred<float> triangle_pred(vertices.data()->data(), faces.data()->data(), sizeof(float) * 3);

  nanort::BVHBuildOptions<float> build_options;
//  build_options.cache_bbox = false;

  nanort::BVHAccel<float> accel;

  LOGGER->info("Building BVH tree");
  bool build_ok = accel.Build(static_cast<const unsigned int>(faces.size()), triangle_mesh, triangle_pred, build_options);
  Ensures(build_ok);

  nanort::BVHBuildStatistics stats = accel.GetStatistics();

  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
  printf("  Max tree depth     : %d\n", stats.max_tree_depth);
  float bmin[3], bmax[3];
  accel.BoundingBox(bmin, bmax);
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

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

  for (int camera_i = 0; camera_i < suncg_cameras.size(); ++camera_i) {
    SuncgCamera suncg_cam = suncg_cameras[camera_i];
    LOGGER->info("Rendering camera {}", camera_i);

    Vec3 cam_eye = suncg_cam.cam_eye;
    Vec3 cam_view_dir = suncg_cam.cam_view_dir;
    Vec3 cam_up = suncg_cam.cam_up;

    double xf = suncg_cam.x_fov;
    double yf = suncg_cam.y_fov;

    scenecompletion::FrustumParams frustum;
    frustum.far = 10000;
    frustum.near = 0.001;

    auto camera = scenecompletion::PerspectiveCamera(cam_eye, cam_eye + cam_view_dir, cam_up, frustum);

/* Old blender setting
  // mm
  double sensor_width = 4.54;
  //  double sensor_height = 3.42;
  double focal_length = 4.10;

  // pixels
  int image_width = 960;
  int image_height = 540;

  // Distance to the image plane
  double image_focal_length = static_cast<double>(image_width) * focal_length / sensor_width;
*/

    const int image_width = flags["width"].as<int>();
    const int image_height = flags["height"].as<int>();

    // Distance to the image plane according to the x fov.
    double xl = 0.5 * image_width / std::tan(xf);
    // Distance to the image plane according to the y fov.
    double yl = 0.5 * image_height / std::tan(yf);

    // For now, we assume the aspect ratio is always 1.0. So the distance to image plane should end up being the same according to both x and y.
    // Otherwise the image size or focal length is wrong. This can also happen because of precision error.
    // 0.1 is an arbitrary threshold.
    if (std::abs(xl - yl) > 0.1) {
      LOGGER->warn("xf: {}, yf: {}, width: {}, height: {}, xl: {}, yl: {}", xf, yf, image_width, image_height, xl, yl);
      throw std::runtime_error("Inconsistent distance to image plane.");
    }

    // Compute the average of the two distances. There are probably other, better ways to do this.
    double image_focal_length = (xl + yl) * 0.5;

    Vec3 image_optical_center{image_width * 0.5, image_height * 0.5, 0};

    const Vec3 cam_ray_origin{0, 0, 0};

    Vec3 ray_origin;
    camera.CamToWorld(cam_ray_origin, &ray_origin);

    vector<vector<float>> depth_images(2);

    nanort::TriangleIntersector<> triangle_intersector(vertices.data()->data(), faces.data()->data(), sizeof(float) * 3);

    for (int y = 0; y < image_height; y++) {
      for (int x = 0; x < image_width; x++) {
        Vec3 image_plane_coord{static_cast<double>(x) + 0.5, image_height - (static_cast<double>(y) + 0.5), -image_focal_length};
        Vec3 cam_ray_direction = (image_plane_coord - image_optical_center).normalized();

        nanort::Ray<float> ray;
        ray.org[0] = static_cast<float>(ray_origin[0]);
        ray.org[1] = static_cast<float>(ray_origin[1]);
        ray.org[2] = static_cast<float>(ray_origin[2]);

        Vec3 ray_direction;
        camera.CamToWorldNormal(cam_ray_direction, &ray_direction);

        ray.dir[0] = static_cast<float>(ray_direction[0]);
        ray.dir[1] = static_cast<float>(ray_direction[1]);
        ray.dir[2] = static_cast<float>(ray_direction[2]);

        int depth_i = 0;
        nanort::TriangleIntersection<> isect{};
        bool hit = accel.Traverse(ray, triangle_intersector, &isect);
        if (hit) {
          auto displacement = isect.t;
          // Depth value is the distance perpendicular to the image plane.
          auto depth_value = static_cast<float>(-cam_ray_direction[2] * displacement);
          depth_images[depth_i].push_back(depth_value);
        } else {
          depth_images[depth_i].push_back(NAN);
        }

        depth_i = 1;
        while (true) {
          nanort::TriangleIntersection<> isect{};
          bool hit = accel.Traverse(ray, triangle_intersector, &isect);
          if (hit) {
            auto displacement = isect.t;
            // Depth value is the distance perpendicular to the image plane.
            auto depth_value = static_cast<float>(-cam_ray_direction[2] * displacement);

            if (is_background(isect.prim_id)) {
              depth_images[depth_i].push_back(depth_value);
              break;
            } else {
              ray.min_t = static_cast<float>(displacement + 0.0001);
            }

          } else {
            depth_images[depth_i].push_back(NAN);
            break;
          }
        }
      }
    }

    for (int j = 0; j < depth_images.size(); ++j) {
      char buff[2048];
      snprintf(buff, sizeof(buff), "%s/%06d_%02d", out_dir.c_str(), camera_i, j);
      string out_filename = std::string(buff) + ".bin";

      scenecompletion::SerializeTensor<float>(out_filename, depth_images[j].data(), {image_height, image_width});

      // NOTE: This line is important. The python script parses this line to determine which files were generated. Must start with "Output file: "
      std::cout << "Output file: " << out_filename << std::endl;
    }

  }

  LOGGER->info("OK");
  return 0;
}
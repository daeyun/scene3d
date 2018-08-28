#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>
#include <lib/multi_layer_depth_renderer.h>

#include "cxxopts.hpp"
#include "csv.h"

#include "lib/file_io.h"

struct CameraParams {
  Vec3 cam_eye;
  Vec3 cam_view_dir;
  Vec3 cam_up;
  double x_fov;
  double y_fov;
  double score;  // scene coverage score. not used at the moment.

  bool is_orthographic = false;
  double left = 0;
  double right = 0;
  double top = 0;
  double bottom = 0;
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
      ("categories", "Path to category mapping file. Segmentation image is not generated if not specified.", cxxopts::value<string>()->default_value(""))
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


  ///
  std::string categories_filename = flags["categories"].as<std::string>();
  bool render_segmentation = scene3d::Exists(categories_filename);

  std::map<std::string, std::string> model_id_to_nyuv2_40class;
  std::set<std::string> nyuv2_40class_set;
  std::map<std::string, int> nyuv2_40class_to_index;

  if (render_segmentation) {
    io::CSVReader<2> csv_reader(categories_filename);
    csv_reader.read_header(io::ignore_extra_column, "model_id", "nyuv2_40class");
    std::string model_id, nyuv2_40class;
    while (csv_reader.read_row(model_id, nyuv2_40class)) {
      if (model_id_to_nyuv2_40class.find(model_id) != model_id_to_nyuv2_40class.end()) {
        LOGGER->error("Duplicate model_id in {}", categories_filename);
        throw std::runtime_error("Duplicate model_id.");
      }
      model_id_to_nyuv2_40class[model_id] = nyuv2_40class;
      if (nyuv2_40class != "void") {
        nyuv2_40class_set.insert(nyuv2_40class);
      }
    }

    LOGGER->info("{} nyu40 classes found in {} (excluding void)", nyuv2_40class_set.size(), categories_filename);

    // Assign numerical values to category names;
    // TODO(daeyun): This is temporary. Need to find a way to make it consistent with other datasets.
    nyuv2_40class_to_index["void"] = 255;
    std::cout << "void" << ": " << 255 << std::endl;
    int nyuv2_40class_i = 0;
    for (const std::string &item: nyuv2_40class_set) {
      nyuv2_40class_to_index[item] = nyuv2_40class_i;
      std::cout << item << ": " << nyuv2_40class_i << std::endl;
      nyuv2_40class_i++;
    }
  }



  ///


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
        prim_id_to_node_name,
        false, 0, 0, 0, 0
    );

    vector<vector<unique_ptr<vector<float>>>> grid_depth_values(height);
    vector<vector<unique_ptr<vector<uint8_t>>>> grid_categories(height);
    vector<vector<unique_ptr<vector<int>>>> grid_node_ids(height);  // TODO: probably temporary
    vector<float> background_values;
    vector<float> foreground_prim_ids;  // those are integer values, but stored as float. because of NaN values.
    vector<uint8_t> background_category_values;
    for (int y = 0; y < height; y++) {
      grid_depth_values[y].resize(width);
      grid_categories[y].resize(width);
      grid_node_ids[y].resize(width);
    }

    size_t num_layers = 0;
    bool found_at_least_one_backgrond_value = false;

    // Depth values along camera ray direction.
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        auto depth_values = make_unique<vector<float>>();
        vector<string> model_ids;
        vector<unsigned int> prim_ids;
        int depth_value_index = renderer.DepthValues(x, y, depth_values.get(), &model_ids, &prim_ids);
        bool found_background = depth_value_index >= 0;

        auto node_ids = make_unique<vector<int>>();  // used for excluding insignificant objects from overhead bounding box computation.
        for (const auto &prim_id : prim_ids) {
          node_ids->push_back(prim_id_to_node_id[prim_id]);
        }

        // depth_values and model_ids must have same size.

        auto nyu40_classes = make_unique<vector<uint8_t>>();
        for (const auto &model_id : model_ids) {
          if (model_id_to_nyuv2_40class.find(model_id) == model_id_to_nyuv2_40class.end()) {
            LOGGER->error("model_id is not in nyuv2_40class. model_id: {}", model_id);
          }
          nyu40_classes->push_back(static_cast<uint8_t>(nyuv2_40class_to_index[model_id_to_nyuv2_40class[model_id]]));
        }

        num_layers = std::max(num_layers, depth_values->size());
        if (found_background) {
          background_values.push_back(depth_values->at(static_cast<size_t>(depth_value_index)));
          background_category_values.push_back(nyu40_classes->at(static_cast<size_t>(depth_value_index)));
          found_at_least_one_backgrond_value = true;
          foreground_prim_ids.push_back(static_cast<float>(prim_ids[0]));
        } else {
          background_values.push_back(NAN);
          background_category_values.push_back(255);
          foreground_prim_ids.push_back(NAN);
        }
        grid_depth_values[y][x] = move(depth_values);
        grid_categories[y][x] = move(nyu40_classes);
        grid_node_ids[y][x] = move(node_ids);
      }
    }

    LOGGER->info("Num layers found: {}", num_layers);

    vector<vector<float>> depth_images;
    vector<vector<uint8_t>> category_images;
    // `max_hits` comes from command line argument. If it was set to 0 (unlimited), the number of depth images will be `num_layers`.
    const size_t n = (max_hits == 0) ? num_layers : std::min(max_hits, num_layers);

    if (n == 0) {
      LOGGER->warn("Zero ray hits. No image saved for camera {}", camera_i);
      continue;
    }

    depth_images.resize(n);
    category_images.resize(n);

    for (int i = 0; i < n; ++i) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          // Populate the depth image buffer in row-major order.
          if (i < grid_depth_values[y][x]->size()) {
            depth_images[i].push_back(grid_depth_values[y][x]->at(static_cast<size_t>(i)));
            category_images[i].push_back(static_cast<uint8_t>(grid_categories[y][x]->at(static_cast<size_t>(i))));  // assuming `grid_categories` has the same size as `grid_depth_values`.
          } else {
            depth_images[i].push_back(NAN);
            category_images[i].push_back(255);
          }
        }
      }
    }


    // Top-down view.
    // Need to figure out which node ids to ignore.
    map<int, int> node_id_counts;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (!grid_node_ids[y][x]->empty()) {
          int node_id = grid_node_ids[y][x]->at(0);
          node_id_counts[node_id]++;
        }
      }
    }
    set<int> node_ids_to_ignore;
    for (const auto &kv: node_id_counts) {
      if (kv.second < height * width * 0.005) {  // this is the number of pixel threshold.
        node_ids_to_ignore.insert(kv.first);
      }
    }

    // Need to find the centroid of background first.
    int i = 0;
    Vec3 centroid{0, 0, 0};
    vector<Vec3> pcl;
    vector<Vec3> pcl_bg;
    int finite_count = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float t = background_values[i];
        if (!std::isnan(t)) {
          {
            Vec3 dir = renderer.RayDirection(x, y);
            Vec3 p = suncg_cam.cam_eye + dir * t;
            centroid += p;
            pcl_bg.push_back(p);
          }

          if (!grid_depth_values[y][x]->empty()) {
            float t2 = grid_depth_values[y][x]->at(0);
            uint8_t nyu40_category = grid_categories[y][x]->at(0);
            bool add_to_boundary_computation = true;

            if (nyu40_category == nyuv2_40class_to_index["door"] ||
                nyu40_category == nyuv2_40class_to_index["wall"] ||
                nyu40_category == nyuv2_40class_to_index["ceiling"] ||
                nyu40_category == nyuv2_40class_to_index["floor"] ||
                nyu40_category == nyuv2_40class_to_index["floor_mat"] ||
                nyu40_category == nyuv2_40class_to_index["curtain"] ||
                nyu40_category == nyuv2_40class_to_index["window"]) {
              add_to_boundary_computation = false;
            }

            if (add_to_boundary_computation) {
              int node_id = grid_node_ids[y][x]->at(0);
              if (node_ids_to_ignore.find(node_id) != node_ids_to_ignore.end()) {
                add_to_boundary_computation = false;
              }
            }

            if (add_to_boundary_computation) {
              if (t2 > 7 || t2 < 1.3) {
                add_to_boundary_computation = false;
              }
            }

            if (add_to_boundary_computation) {
              Vec3 dir = renderer.RayDirection(x, y);
              Vec3 p = suncg_cam.cam_eye + dir * t2;
              pcl.push_back(p);
            }
          }

          ++finite_count;
        }
        ++i;
      }
    }
    centroid /= finite_count;

    double floor_y = kInfinity;
    for (const auto &item : pcl_bg) {
      if (item[1] < floor_y) {
        floor_y = item[1];
      }
    }

    Vec3 center_of_rotation = (centroid + suncg_cam.cam_eye) * 0.5;

    CameraParams topdown_cam;
    topdown_cam.cam_eye = center_of_rotation + ((suncg_cam.cam_eye - center_of_rotation).norm() * Vec3{0, 1, 0}) * 0.5;
    topdown_cam.cam_view_dir = {0, -1, 0};
    Vec3 topdown_right = (topdown_cam.cam_view_dir.cross(suncg_cam.cam_up)).normalized();
    topdown_cam.cam_up = topdown_right.cross(topdown_cam.cam_view_dir);
    topdown_cam.cam_up.normalize();
    topdown_cam.x_fov = suncg_cam.x_fov;
    topdown_cam.y_fov = suncg_cam.y_fov;

    scene3d::FrustumParams frustum;
    frustum.far = 10000;
    frustum.near = 0.01;
    auto cam = std::make_unique<scene3d::PerspectiveCamera>(topdown_cam.cam_eye, topdown_cam.cam_eye + topdown_cam.cam_view_dir, topdown_cam.cam_up, frustum);

    Vec3 bbox_min{kInfinity, kInfinity, kInfinity};
    Vec3 bbox_max{-kInfinity, -kInfinity, -kInfinity};
    for (const auto &p: pcl) {
      Vec3 p_cam;
      cam->WorldToCam(p, &p_cam);
      for (int j = 0; j < 3; ++j) {
        if (p_cam[j] < bbox_min[j]) {
          bbox_min[j] = p_cam[j];
        }
        if (p_cam[j] > bbox_max[j]) {
          bbox_max[j] = p_cam[j];
        }
      }
    }
    LOGGER->info("bbox_min {}, {}, {}", bbox_min[0], bbox_min[1], bbox_min[2]);
    LOGGER->info("bbox_max {}, {}, {}", bbox_max[0], bbox_max[1], bbox_max[2]);
    Vec3 bbox_max_world;
    cam->CamToWorld(bbox_max, &bbox_max_world);
    LOGGER->info("bbox_max_world {}, {}, {}", bbox_max_world[0], bbox_max_world[1], bbox_max_world[2]);
    Vec3 bbox_center = (bbox_max + bbox_min) * 0.5;
    Vec3 bbox_center_world;
    cam->CamToWorld(bbox_center, &bbox_center_world);
    topdown_cam.cam_eye[0] = bbox_center_world[0];
    topdown_cam.cam_eye[2] = bbox_center_world[2];

    bbox_min = {kInfinity, kInfinity, kInfinity};
    bbox_max = {-kInfinity, -kInfinity, -kInfinity};
    for (const auto &p: pcl) {
      Vec3 p_cam;
      cam->WorldToCam(p, &p_cam);
      for (int j = 0; j < 3; ++j) {
        if (p_cam[j] < bbox_min[j]) {
          bbox_min[j] = p_cam[j];
        }
        if (p_cam[j] > bbox_max[j]) {
          bbox_max[j] = p_cam[j];
        }
      }
    }

    {
      char buff[2048];
      snprintf(buff, sizeof(buff), "/tmp/scene3d/%06d.bin", camera_i);
      scene3d::WritePclTensor(std::string(buff), pcl);
    }

    double left = -(bbox_max[0] - bbox_min[0]) * 0.5;
    double right = -left;
    double top = (bbox_max[1] - bbox_min[1]) * 0.5;
    double bottom = -top;

    double hw_ratio = static_cast<double>(height) / static_cast<double>(width);
    double lr = std::abs(right - left);
    double bt = std::abs(top - bottom);
    double box_hw_ratio = bt / lr;

    if (box_hw_ratio < hw_ratio) {
      // image is squeezed horizontally.
      double padding = (hw_ratio * lr - bt) * 0.5;
      top += padding;
      bottom -= padding;
    } else {
      double padding = (bt - hw_ratio * lr) / hw_ratio * 0.5;
      right += padding;
      left -= padding;
    }

    double edge_padding_ratio = 1.1; // this is actually 5% of the width or height.
    top *= edge_padding_ratio;
    bottom *= edge_padding_ratio;
    left *= edge_padding_ratio;
    right *= edge_padding_ratio;

    LOGGER->info("center of rotation {}, {}, {}", center_of_rotation[0], center_of_rotation[1], center_of_rotation[2]);

    auto top_down_renderer = scene3d::SunCgMultiLayerDepthRenderer(
        &ray_tracer,
        topdown_cam.cam_eye,
        topdown_cam.cam_view_dir,
        topdown_cam.cam_up,
        topdown_cam.x_fov,
        topdown_cam.y_fov,
        width,
        height,
        max_hits,
        prim_id_to_node_name,
        true,
        left,
        right,
        top,
        bottom // TODO
    );
    top_down_renderer.set_do_not_render_background_except_floor(true);
    vector<float> top_down_foreground_values;
    vector<float> top_down_foreground_prim_ids;  // those are integer values, but stored as float. because of NaN values.
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        auto depth_values = make_unique<vector<float>>();
        vector<string> model_ids;
        vector<unsigned int> prim_ids;
        int depth_value_index = top_down_renderer.DepthValues(x, y, depth_values.get(), &model_ids, &prim_ids);
        bool found_background = depth_value_index >= 0;

        if (found_background) {
          float depth_value = depth_values->at(0);
          top_down_foreground_values.push_back((topdown_cam.cam_eye[1] - floor_y) - depth_value);
          top_down_foreground_prim_ids.push_back(static_cast<float>(prim_ids[0]));
        } else {
          top_down_foreground_values.push_back(NAN);
          top_down_foreground_prim_ids.push_back(NAN);
        }
      }
    }


    // Thickness
    vector<float> thickness_values;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float thickness = renderer.ObjectCenteredVolume(x, y);
        if (thickness > 0) {
          thickness_values.push_back(thickness);
        } else {
          thickness_values.push_back(NAN);
        }
      }
    }


    // Save buffer. Contains (N, H, W) tensor data.
    vector<float> all_depth_values;
    all_depth_values.reserve(height * width * n);
    for (const auto &depth_image : depth_images) {
      all_depth_values.insert(all_depth_values.end(), depth_image.begin(), depth_image.end());
    }

    vector<uint8_t> all_category_values;
    all_category_values.reserve(height * width * n);
    for (const auto &category_image : category_images) {
      all_category_values.insert(all_category_values.end(), category_image.begin(), category_image.end());
    }

    char buff[2048];
    snprintf(buff, sizeof(buff), "%s/%06d", out_dir.c_str(), camera_i);
    {
      string out_filename = std::string(buff) + ".bin";
      const vector<int> shape{static_cast<int>(n), static_cast<int>(height), static_cast<int>(width)};
      scene3d::SerializeTensor<float>(out_filename, all_depth_values.data(), shape);
      // NOTE: This line is important. The python script parses this line to determine which files were generated. Must start with "Output file: "
      std::cout << "Output file: " << out_filename << std::endl;
    }

    {
      string out_filename = std::string(buff) + "_c.bin";
      const vector<int> shape{static_cast<int>(n), static_cast<int>(height), static_cast<int>(width)};
      scene3d::SerializeTensor<uint8_t>(out_filename, all_category_values.data(), shape);
      // NOTE: This line is important. The python script parses this line to determine which files were generated. Must start with "Output file: "
      std::cout << "Output file: " << out_filename << std::endl;
    }

    if (found_at_least_one_backgrond_value) {
      {
        string out_filename_bg = std::string(buff) + "_bg.bin";
        scene3d::SerializeTensor<float>(out_filename_bg, background_values.data(), {height, width});
        std::cout << "Output file: " << out_filename_bg << std::endl;
      }

      {
        string out_filename_bg = std::string(buff) + "_bg_c.bin";
        scene3d::SerializeTensor<uint8_t>(out_filename_bg, background_category_values.data(), {height, width});
        std::cout << "Output file: " << out_filename_bg << std::endl;
      }

      // Save top-down view
      {
        string out_filename_bg = std::string(buff) + "_td.bin";
        scene3d::SerializeTensor<float>(out_filename_bg, top_down_foreground_values.data(), {height, width});
        std::cout << "Output file: " << out_filename_bg << std::endl;
      }

      {
        string out_filename_bg = std::string(buff) + "_td_prim.bin";
        scene3d::SerializeTensor<float>(out_filename_bg, top_down_foreground_prim_ids.data(), {height, width});
        std::cout << "Output file: " << out_filename_bg << std::endl;
      }

      // Also save foreground prim ids of the original camera.
      {
        string out_filename_bg = std::string(buff) + "_prim.bin";
        scene3d::SerializeTensor<float>(out_filename_bg, foreground_prim_ids.data(), {height, width});
        std::cout << "Output file: " << out_filename_bg << std::endl;
      }
    }

#if 1
    string out_filename_thickness = std::string(buff) + "_ot.bin";
    scene3d::SerializeTensor<float>(out_filename_thickness, thickness_values.data(), {height, width});
    std::cout << "Output file: " << out_filename_thickness << std::endl;
#endif

#if 1
    {
      string out_filename_overhead_cam = std::string(buff) + "_td_cam.txt";
      scene3d::WriteFloatsTxt<double>(out_filename_overhead_cam, 11, vector<double>{
          topdown_cam.cam_eye[0],
          topdown_cam.cam_eye[1],
          topdown_cam.cam_eye[2],
          topdown_cam.cam_view_dir[0],
          topdown_cam.cam_view_dir[1],
          topdown_cam.cam_view_dir[2],
          topdown_cam.cam_up[0],
          topdown_cam.cam_up[1],
          topdown_cam.cam_up[2],
          left, right, top, bottom});
      std::cout << "Output file: " << out_filename_overhead_cam << std::endl;
    }

    {
      string out_filename_overhead_cam = std::string(buff) + "_cam.txt";
      scene3d::WriteFloatsTxt<double>(out_filename_overhead_cam, 11, vector<double>{
          suncg_cam.cam_eye[0],
          suncg_cam.cam_eye[1],
          suncg_cam.cam_eye[2],
          suncg_cam.cam_view_dir[0],
          suncg_cam.cam_view_dir[1],
          suncg_cam.cam_view_dir[2],
          suncg_cam.cam_up[0],
          suncg_cam.cam_up[1],
          suncg_cam.cam_up[2],
          suncg_cam.x_fov,
          suncg_cam.y_fov});
      std::cout << "Output file: " << out_filename_overhead_cam << std::endl;
    }

#endif

  }

  LOGGER->info("OK");
  return 0;
}

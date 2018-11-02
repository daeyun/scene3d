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

void ReadLayerMeshes(const string &filename, const scene3d::Camera &camera, float floor_height, array<TriMesh, 4> *meshes) {
  vector<int> shape;
  vector<float> ldi_data;
  ReadTensorData(filename, &shape, &ldi_data);

  Ensures(shape.size() == 3);
  Ensures(shape[0] == 4);

  const auto height = static_cast<unsigned int>(shape[1]);
  const auto width = static_cast<unsigned int>(shape[2]);

  // TODO: Need to detect floor height.
  if (!camera.is_perspective()) {
    std::transform(ldi_data.begin(), ldi_data.end(), ldi_data.begin(),
                   [&](float value) -> float { return static_cast<float>(camera.position()[1] - value - floor_height); });
  }

  float *layer0_data = ldi_data.data();

  {
    Image<float> layer(layer0_data, height, width, NAN);

    Ensures(layer.height() == height);
    Ensures(layer.width() == width);

    float dd_factor = 5.0;
    TriangulateDepth(layer, camera, dd_factor, &meshes->at(0).faces, &meshes->at(0).vertices);
  }

  float *layer1_data = ldi_data.data() + shape[1] * shape[2];

  {
    Image<float> layer(layer1_data, height, width, NAN);

    Ensures(layer.height() == height);
    Ensures(layer.width() == width);

    vector<array<unsigned int, 3>> faces;
    vector<array<float, 3>> vertices;

    float dd_factor = 5.0;
    TriangulateDepth(layer, camera, dd_factor, &meshes->at(1).faces, &meshes->at(1).vertices);
  }

  float *layer2_data = ldi_data.data() + shape[1] * shape[2] * 2;

  {
    Image<float> layer(layer2_data, height, width, NAN);

    Ensures(layer.height() == height);
    Ensures(layer.width() == width);

    vector<array<unsigned int, 3>> faces;
    vector<array<float, 3>> vertices;

    float dd_factor = 5.0;
    TriangulateDepth(layer, camera, dd_factor, &meshes->at(2).faces, &meshes->at(2).vertices);
  }

  float *layer3_data = ldi_data.data() + shape[1] * shape[2] * 3;

  {
    Image<float> layer(layer3_data, height, width, NAN);

    Ensures(layer.height() == height);
    Ensures(layer.width() == width);

    vector<array<unsigned int, 3>> faces;
    vector<array<float, 3>> vertices;

    float dd_factor = 5.0;
    TriangulateDepth(layer, camera, dd_factor, &meshes->at(3).faces, &meshes->at(3).vertices);
  }
}

int main(int argc, const char **argv) {
  cxxopts::Options options("render_suncg", "Render multi-layer depth images");

  options.add_options()
      ("example_name", "Example name.", cxxopts::value<string>())
      ("obj", "Path to obj mesh file.", cxxopts::value<string>())
      ("json", "Path to house json file.", cxxopts::value<string>())
      ("category", "Path to category mapping file. e.g. ModelCategoryMapping.csv", cxxopts::value<string>())
      ("help", "Display help.", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
  auto flags = options.parse(argc, argv);

  if (flags["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  // Initialize logging.
  spdlog::stdout_color_mt("console");

  vector<string> required_flags = {"obj", "json", "example_name", "category"};
  for (const string &name: required_flags) {
    if (!flags.count(name)) {
      LOGGER->error("No argument specified for required option --{}. See --help.", name);
      throw std::runtime_error("");
    }
  }

  const string obj_filename = flags["obj"].as<string>();
  const string json_filename = flags["json"].as<string>();
  const string category_filename = flags["category"].as<string>();
  const string example_name = flags["example_name"].as<string>();
  Expects(Exists(obj_filename));
  Expects(Exists(json_filename));
  Expects(Exists(category_filename));

  const string kBaseDir = "/data2/scene3d/v8/renderings/";
  const string camera_filename = kBaseDir + example_name + "_cam.txt";
  Expects(Exists(camera_filename));

  TriMesh gt_mesh;
  TriMesh gt_mesh_background_only;
  TriMesh gt_mesh_object_only;
  // Read the obj, json, and category mappings.
  auto scene = make_unique<suncg::Scene>(json_filename, obj_filename, category_filename);
  scene->Build();

  vector<unique_ptr<scene3d::Camera>> cameras;
  ReadCameras(camera_filename, &cameras);

  // TODO: this should be a command line argument.
  bool write_ply = true;

  float floor_height;
  {
    scene3d::Camera *camera = cameras[0].get();

    const unsigned int resample_height = 240 * 2;
    const unsigned int resample_width = 320 * 2;

    floor_height = ExtractFrustumMesh(scene.get(), *camera, resample_height, resample_width, 10, &gt_mesh_background_only, &gt_mesh_object_only);

    // TODO
    gt_mesh.AddMesh(gt_mesh_object_only);
    gt_mesh.AddMesh(gt_mesh_background_only);

    LOGGER->info("Floor height: {}", floor_height);

    if (write_ply) {
      WritePly("/tmp/scene3d_test/gt.ply", gt_mesh.faces, gt_mesh.vertices, false);
      WritePly("/tmp/scene3d_test/gt_object_only.ply", gt_mesh_object_only.faces, gt_mesh_object_only.vertices, false);
    }
  }

//  const string ldi_filename = kBaseDir + example_name + "_ldi.bin";
  const string ldi_filename = "/mnt/ramdisk/scene3d/target.bin";
  array<TriMesh, 4> frontal_meshes;
  ReadLayerMeshes(ldi_filename, *cameras[0], floor_height, &frontal_meshes);

  if (write_ply) {
    WritePly("/tmp/scene3d_test/frontal_layer0.ply", frontal_meshes.at(0).faces, frontal_meshes.at(0).vertices, false);
    WritePly("/tmp/scene3d_test/frontal_layer1.ply", frontal_meshes.at(1).faces, frontal_meshes.at(1).vertices, false);
    WritePly("/tmp/scene3d_test/frontal_layer2.ply", frontal_meshes.at(2).faces, frontal_meshes.at(2).vertices, false);
    WritePly("/tmp/scene3d_test/frontal_layer3.ply", frontal_meshes.at(3).faces, frontal_meshes.at(3).vertices, false);
  }

  {

//  const string ldi_filename = kBaseDir + example_name + "_ldi.bin";
    const string ldi_filename = "/mnt/ramdisk/scene3d/pred.bin";
    array<TriMesh, 4> frontal_meshes;
    ReadLayerMeshes(ldi_filename, *cameras[0], floor_height, &frontal_meshes);

    if (write_ply) {
      WritePly("/tmp/scene3d_test/pred_frontal_layer0.ply", frontal_meshes.at(0).faces, frontal_meshes.at(0).vertices, false);
      WritePly("/tmp/scene3d_test/pred_frontal_layer1.ply", frontal_meshes.at(1).faces, frontal_meshes.at(1).vertices, false);
      WritePly("/tmp/scene3d_test/pred_frontal_layer2.ply", frontal_meshes.at(2).faces, frontal_meshes.at(2).vertices, false);
      WritePly("/tmp/scene3d_test/pred_frontal_layer3.ply", frontal_meshes.at(3).faces, frontal_meshes.at(3).vertices, false);
    }

    frontal_meshes[0].AddMesh(frontal_meshes[1]);
    frontal_meshes[0].AddMesh(frontal_meshes[2]);
    frontal_meshes[0].AddMesh(frontal_meshes[3]);
    WritePly("/tmp/scene3d_test/pred_frontal.ply", frontal_meshes.at(0).faces, frontal_meshes.at(0).vertices, false);
  }

//  const string overhead_filename = kBaseDir + example_name + "_ldi-o.bin";
  const string overhead_filename = "/mnt/ramdisk/scene3d/overhead.bin";
  array<TriMesh, 4> overhead_meshes;
  ReadLayerMeshes(overhead_filename, *cameras[1], floor_height, &overhead_meshes);

  if (write_ply) {
    WritePly("/tmp/scene3d_test/overhead_layer0.ply", overhead_meshes.at(0).faces, overhead_meshes.at(0).vertices, false);
//    WritePly("/tmp/scene3d_test/overhead_layer1.ply", overhead_meshes.at(1).faces, overhead_meshes.at(1).vertices, false);
//    WritePly("/tmp/scene3d_test/overhead_layer2.ply", overhead_meshes.at(2).faces, overhead_meshes.at(2).vertices, false);
//    WritePly("/tmp/scene3d_test/overhead_layer3.ply", overhead_meshes.at(3).faces, overhead_meshes.at(3).vertices, false);
  }

  return 0;


  using meshdist_cgal::Triangle;

  std::vector<Triangle> gt_mesh_triangles;
  TrianglesFromTriMesh(gt_mesh, &gt_mesh_triangles);
  std::vector<Triangle> gt_mesh_object_only_triangles;
  TrianglesFromTriMesh(gt_mesh_object_only, &gt_mesh_object_only_triangles);

  array<TriMesh, 4> frontal_meshes_cumulative;
  array<TriMesh, 4> overhead_meshes_cumulative;
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < j + 1; ++i) {
      // TODO: refactor
      frontal_meshes_cumulative[j].AddMesh(frontal_meshes[i]);
      overhead_meshes_cumulative[j].AddMesh(overhead_meshes[i]);
    }
  }
  array<TriMesh, 4> combined_meshes_cumulative;
  for (int j = 0; j < 4; ++j) {
    combined_meshes_cumulative[j].AddMesh(frontal_meshes_cumulative[3]);
    for (int i = 0; i < j + 1; ++i) {
      combined_meshes_cumulative[j].AddMesh(overhead_meshes[i]);
    }
  }

  array<std::vector<Triangle>, 4> frontal_triangles;
  array<std::vector<Triangle>, 4> frontal_triangles_cumulative;
  array<std::vector<Triangle>, 4> overhead_triangles;
  array<std::vector<Triangle>, 4> overhead_triangles_cumulative;
  array<std::vector<Triangle>, 4> combined_triangles_cumulative;
  for (int j = 0; j < 4; ++j) {
    TrianglesFromTriMesh(frontal_meshes[j], &frontal_triangles[j]);
    TrianglesFromTriMesh(frontal_meshes_cumulative[j], &frontal_triangles_cumulative[j]);
    TrianglesFromTriMesh(overhead_meshes[j], &overhead_triangles[j]);
    TrianglesFromTriMesh(overhead_meshes_cumulative[j], &overhead_triangles_cumulative[j]);
    TrianglesFromTriMesh(combined_meshes_cumulative[j], &combined_triangles_cumulative[j]);
  }

  const float kSamplingDensity = 3000;
  const float kInlierThreshold = 0.005;

  for (int l = 0; l < 4; ++l) {
    std::vector<float> distances;
    float rms = MeshToMeshDistanceOneDirection(gt_mesh_triangles, frontal_triangles[l], kSamplingDensity, &distances);
    LOGGER->info("rms {}", rms);
    LOGGER->info("{} distances", distances.size());

    int inlier_count = 0;
    for (int k = 0; k < distances.size(); ++k) {
      if (distances[k] < kInlierThreshold) {
        inlier_count++;
      }
    }
    LOGGER->info("{} inliers", inlier_count);
    LOGGER->info("Frontal coverage: {}", static_cast<double>(inlier_count) / static_cast<double>(distances.size()));
  }

  for (int l = 0; l < 4; ++l) {
    std::vector<float> distances;
    float rms = MeshToMeshDistanceOneDirection(gt_mesh_triangles, frontal_triangles_cumulative[l], kSamplingDensity, &distances);
    LOGGER->info("rms {}", rms);
    LOGGER->info("{} distances", distances.size());

    int inlier_count = 0;
    for (int k = 0; k < distances.size(); ++k) {
      if (distances[k] < kInlierThreshold) {
        inlier_count++;
      }
    }
    LOGGER->info("{} inliers", inlier_count);
    LOGGER->info("Frontal coverage (cumulative): {}", static_cast<double>(inlier_count) / static_cast<double>(distances.size()));
  }

  for (int l = 0; l < 4; ++l) {
    std::vector<float> distances;
    float rms = MeshToMeshDistanceOneDirection(gt_mesh_triangles, overhead_triangles[l], kSamplingDensity, &distances);
    LOGGER->info("rms {}", rms);
    LOGGER->info("{} distances", distances.size());

    int inlier_count = 0;
    for (int k = 0; k < distances.size(); ++k) {
      if (distances[k] < kInlierThreshold) {
        inlier_count++;
      }
    }
    LOGGER->info("{} inliers", inlier_count);
    LOGGER->info("Overhead coverage: {}", static_cast<double>(inlier_count) / static_cast<double>(distances.size()));
  }

  for (int l = 0; l < 4; ++l) {
    std::vector<float> distances;
    float rms = MeshToMeshDistanceOneDirection(gt_mesh_triangles, overhead_triangles_cumulative[l], kSamplingDensity, &distances);
    LOGGER->info("rms {}", rms);
    LOGGER->info("{} distances", distances.size());

    int inlier_count = 0;
    for (int k = 0; k < distances.size(); ++k) {
      if (distances[k] < kInlierThreshold) {
        inlier_count++;
      }
    }
    LOGGER->info("{} inliers", inlier_count);
    LOGGER->info("Overhead coverage (cumulative): {}", static_cast<double>(inlier_count) / static_cast<double>(distances.size()));
  }

  for (int l = 0; l < 4; ++l) {
    std::vector<float> distances;
    float rms = MeshToMeshDistanceOneDirection(gt_mesh_triangles, combined_triangles_cumulative[l], kSamplingDensity, &distances);
    LOGGER->info("rms {}", rms);
    LOGGER->info("{} distances", distances.size());

    int inlier_count = 0;
    for (int k = 0; k < distances.size(); ++k) {
      if (distances[k] < kInlierThreshold) {
        inlier_count++;
      }
    }
    LOGGER->info("{} inliers", inlier_count);
    LOGGER->info("Combined coverage (cumulative): {}", static_cast<double>(inlier_count) / static_cast<double>(distances.size()));
  }


  // Object only

  for (int l = 0; l < 4; ++l) {
    std::vector<float> distances;
    float rms = MeshToMeshDistanceOneDirection(gt_mesh_object_only_triangles, frontal_triangles[l], kSamplingDensity, &distances);
    LOGGER->info("rms {}", rms);
    LOGGER->info("{} distances", distances.size());

    int inlier_count = 0;
    for (int k = 0; k < distances.size(); ++k) {
      if (distances[k] < kInlierThreshold) {
        inlier_count++;
      }
    }
    LOGGER->info("{} inliers", inlier_count);
    LOGGER->info("Frontal coverage, obj only: {}", static_cast<double>(inlier_count) / static_cast<double>(distances.size()));
  }

  for (int l = 0; l < 4; ++l) {
    std::vector<float> distances;
    float rms = MeshToMeshDistanceOneDirection(gt_mesh_object_only_triangles, frontal_triangles_cumulative[l], kSamplingDensity, &distances);
    LOGGER->info("rms {}", rms);
    LOGGER->info("{} distances", distances.size());

    int inlier_count = 0;
    for (int k = 0; k < distances.size(); ++k) {
      if (distances[k] < kInlierThreshold) {
        inlier_count++;
      }
    }
    LOGGER->info("{} inliers", inlier_count);
    LOGGER->info("Frontal coverage, obj only (cumulative): {}", static_cast<double>(inlier_count) / static_cast<double>(distances.size()));
  }

  for (int l = 0; l < 4; ++l) {
    std::vector<float> distances;
    float rms = MeshToMeshDistanceOneDirection(gt_mesh_object_only_triangles, overhead_triangles[l], kSamplingDensity, &distances);
    LOGGER->info("rms {}", rms);
    LOGGER->info("{} distances", distances.size());

    int inlier_count = 0;
    for (int k = 0; k < distances.size(); ++k) {
      if (distances[k] < kInlierThreshold) {
        inlier_count++;
      }
    }
    LOGGER->info("{} inliers", inlier_count);
    LOGGER->info("Overhead coverage, obj only: {}", static_cast<double>(inlier_count) / static_cast<double>(distances.size()));
  }

  for (int l = 0; l < 4; ++l) {
    std::vector<float> distances;
    float rms = MeshToMeshDistanceOneDirection(gt_mesh_object_only_triangles, overhead_triangles_cumulative[l], kSamplingDensity, &distances);
    LOGGER->info("rms {}", rms);
    LOGGER->info("{} distances", distances.size());

    int inlier_count = 0;
    for (int k = 0; k < distances.size(); ++k) {
      if (distances[k] < kInlierThreshold) {
        inlier_count++;
      }
    }
    LOGGER->info("{} inliers", inlier_count);
    LOGGER->info("Overhead coverage, obj only (cumulative): {}", static_cast<double>(inlier_count) / static_cast<double>(distances.size()));
  }

  for (int l = 0; l < 4; ++l) {
    std::vector<float> distances;
    float rms = MeshToMeshDistanceOneDirection(gt_mesh_object_only_triangles, combined_triangles_cumulative[l], kSamplingDensity, &distances);
    LOGGER->info("rms {}", rms);
    LOGGER->info("{} distances", distances.size());

    int inlier_count = 0;
    for (int k = 0; k < distances.size(); ++k) {
      if (distances[k] < kInlierThreshold) {
        inlier_count++;
      }
    }
    LOGGER->info("{} inliers", inlier_count);
    LOGGER->info("Combined coverage, obj only (cumulative): {}", static_cast<double>(inlier_count) / static_cast<double>(distances.size()));
  }

}

#include "suncg_utils.h"

#include <fstream>
#include <stdexcept>

#include "string_utils.h"
#include "file_io.h"

#include "nlohmann/json.hpp"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "assimp/DefaultLogger.hpp"
#include "assimp/LogStream.hpp"
#include "boost/filesystem.hpp"
#include "csv.h"

namespace scene3d {
namespace suncg {

void ParseObjectRoomHierarchy(const string &house_json_filename, map<string, Instance> *node_id_to_node) {
  Expects(Exists(house_json_filename));

  using nlohmann::json;

  std::ifstream i(house_json_filename);
  json j;
  i >> j;

  // Sanity check.
  Expects("[0,1,0]"_json == j["up"]);
  Expects("[0,0,1]"_json == j["front"]);
  Expects(1.0 == j["scaleToMeters"].get<float>());

  int num_levels = j["levels"].size();

  for (int level = 0; level < num_levels; ++level) {
    int num_nodes = j["levels"][level]["nodes"].size();
    for (int node_i = 0; node_i < num_nodes; ++node_i) {
      const auto &node = j["levels"][level]["nodes"][node_i];

      bool is_valid = node["valid"].get<uint8_t>();
      string node_id = node["id"].get<string>();

      if (node["type"] == "Room") {
        Expects(is_valid);  // Assumes there is no invalid room.

        if (node.find("nodeIndices") == node.end()) {
          LOGGER->warn("Room {} has no objects.", node_id);
        } else {
          int num_objects_in_room = node["nodeIndices"].size();
          for (int k = 0; k < num_objects_in_room; ++k) {
            int object_node_index = node["nodeIndices"][k].get<int>();
            const auto &object_node = j["levels"][level]["nodes"][object_node_index];
            if (!object_node["valid"].get<uint8_t>()) {
              continue;
            }
            string object_node_id = object_node["id"];

            if (node_id_to_node->find(object_node_id) != node_id_to_node->end()) {
              LOGGER->error("Duplicate object found. {} already belongs to {}. Current room node id: {}", object_node_id, (*node_id_to_node)[object_node_id].room_id, node_id);
            }

            Instance node_object;
            node_object.id = object_node_id;
            node_object.room_id = node_id;

            (*node_id_to_node)[object_node_id] = node_object;
          }
        }
      }
    }
  }

  // Model ids and objects with no room ids are handled in this second pass.
  // Node types include Object, Ground, Box, etc.
  for (int level = 0; level < num_levels; ++level) {
    int num_nodes = j["levels"][level]["nodes"].size();
    for (int node_i = 0; node_i < num_nodes; ++node_i) {
      const auto &node = j["levels"][level]["nodes"][node_i];

      bool is_valid = node["valid"].get<uint8_t>();
      string node_id = node["id"].get<string>();

      if (!is_valid) {
        continue;
      }

      if (node["type"] != "Room") {
        auto &node_object = (*node_id_to_node)[node_id];
        if (!node_object.id.empty()) {
          Ensures(node_object.id == node_id);  // Sanity check.
        } else {
          node_object.id = node_id;
        }
        if (node.find("modelId") != node.end()) {  // Box might not have model id.
          node_object.model_id = node["modelId"].get<string>();
        }
      }
    }
  }

  // Node type will be assigned in `Scene::Build`.
}

void ReadCameraFile(const string &filename, vector<suncg::CameraParams> *suncg_params) {
  LOGGER->info("Reading file {}", filename);

  std::ifstream source;
  source.open(filename, std::ios_base::in);
  if (!source) {
    throw std::runtime_error("Can't open file.");
  }

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

    LOGGER->info("camera {}, eye {}, {}, {}, fov {}, {}", suncg_params->size(), cam.cam_eye[0], cam.cam_eye[1], cam.cam_eye[2], cam.x_fov, cam.y_fov);

    cam.cam_view_dir.normalize();

    suncg_params->push_back(cam);
  }
}
void WriteCameraFile(const string &filename, const vector<PerspectiveCamera> &cameras) {
  int precision = 13;
  std::ofstream ofile;
  ofile.open(filename, std::ios::out);

  for (const auto &camera : cameras) {
    double fx, fy;
    camera.fov(&fx, &fy);
    ofile << std::setprecision(precision) <<
          camera.position()[0] << " " <<
          camera.position()[1] << " " <<
          camera.position()[2] << " " <<
          camera.viewing_direction()[0] << " " <<
          camera.viewing_direction()[1] << " " <<
          camera.viewing_direction()[2] << " " <<
          camera.up()[0] << " " <<
          camera.up()[1] << " " <<
          camera.up()[2] << " " <<
          fx << " " <<
          fy << " " <<
          0 << std::endl;
  }

  ofile.close();
}

void ReadCameraFile(const string &filename, vector<PerspectiveCamera> *cameras) {
  vector<suncg::CameraParams> params;
  ReadCameraFile(filename, &params);

  for (const auto &param : params) {
    cameras->push_back(MakeCamera(param, 0.01, 100));  // near, far

  }
}

void Scene::Build() {
  if (!vertices.empty() || !faces.empty()) {
    LOGGER->error("Build() should run only once.");
    throw std::runtime_error("");
  }

  ParseObjectRoomHierarchy(source_json_filename, &this->instance_id_to_node);

  std::vector<string> face_group_names;

  Ensures(ReadFacesAndVertices(source_obj_filename, &this->faces, &this->vertices, &face_group_names));
  Ensures(this->faces.size() == face_group_names.size());
  Ensures(!this->vertices.empty());

  has_normal.resize(this->faces.size());

  for (const auto &group_name : face_group_names) {
    string instance_id;
    InstanceType node_type;
    string room_id;  // for room components only.

    // Room envelope component names will be room_id#Typename.
    bool is_room_component = false;
    if (StartsWith(group_name, "WallInside#")) {
      room_id = group_name.substr(sizeof("WallInside#") - 1);  // This will be {room_id}_{wall_id}
      instance_id = room_id + "#WallInside";
      node_type = InstanceType::WallInside;
      is_room_component = true;
    } else if (StartsWith(group_name, "WallOutside#")) {
      room_id = group_name.substr(sizeof("WallOutside#") - 1);  // This will be {room_id}_{wall_id}
      instance_id = room_id + "#WallOutside";
      node_type = InstanceType::WallOutside;
      is_room_component = true;
    } else if (StartsWith(group_name, "Object#")) {
      instance_id = group_name.substr(sizeof("Object#") - 1);
      node_type = InstanceType::Object;
    } else if (StartsWith(group_name, "Box#")) {
      instance_id = group_name.substr(sizeof("Box#") - 1);
      node_type = InstanceType::Box;
    } else if (StartsWith(group_name, "Ground#")) {
      instance_id = group_name.substr(sizeof("Ground#") - 1);
      node_type = InstanceType::Ground;
    } else if (StartsWith(group_name, "Floor#")) {
      room_id = group_name.substr(sizeof("Floor#") - 1);
      instance_id = room_id + "#Floor";
      node_type = InstanceType::Floor;
      is_room_component = true;
    } else if (StartsWith(group_name, "Ceiling#")) {
      room_id = group_name.substr(sizeof("Ceiling#") - 1);
      instance_id = room_id + "#Ceiling";
      node_type = InstanceType::Ceiling;
      is_room_component = true;
    } else {
      LOGGER->error("Unexpected group name: {}", group_name);
      throw std::runtime_error("");
    }

    if (is_room_component) {
      Instance node_object;
      node_object.id = instance_id;
      node_object.type = node_type;
      // No room or model id.
      this->instance_id_to_node[instance_id] = node_object;
    } else {
      auto it = this->instance_id_to_node.find(instance_id);
      if (it == this->instance_id_to_node.end()) {
        LOGGER->error("node_id {} is not found in node_id_to_node. Group name: {}", instance_id, group_name);
      }
      Ensures(it != this->instance_id_to_node.end());

      if (it->second.type != node_type) {
        it->second.type = node_type;
      }
    }

    this->face_instance_ids.push_back(instance_id);
  }

  ParseCategoryMapping(source_category_mapping_csv_filename, &this->model_id_to_category);
}

Scene::Scene(const string &house_json_filename, const string &house_obj_filename, const string &category_mapping_csv_filename)
    : source_obj_filename(house_obj_filename),
      source_json_filename(house_json_filename),
      source_category_mapping_csv_filename(category_mapping_csv_filename) {
  Ensures(EndsWith(source_json_filename, ".json"));
  Ensures(EndsWith(source_obj_filename, ".obj"));
  Ensures(EndsWith(source_category_mapping_csv_filename, ".csv"));
}

bool ReadFacesAndVertices(const std::string &filename,
                          std::vector<std::array<unsigned int, 3>> *faces,
                          std::vector<std::array<float, 3>> *vertices,
                          std::vector<string> *face_group_names) {
  LOGGER->info("Importing {}", filename);
  if (!boost::filesystem::exists(filename)) {
    LOGGER->error("{} does not exist", filename);
    throw std::runtime_error("file not found");
  }

  Assimp::Importer importer;

  // List of post-processing flags can be found here:
  // http://sir-kimmi.de/assimp/lib_html/postprocess_8h.html#a64795260b95f5a4b3f3dc1be4f52e410
  const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

  if (!scene) {
    LOGGER->error("ERROR in {}: {}", filename, importer.GetErrorString());
    return false;
  }

  // TODO: there seems to be a problem reading binary ply files. this is a hack to detect parsing error.
  const float kMaxVertexValue = 1e7;

  int triangle_count = 0;
  int face_offset = 0;

  // Recursively collect nodes in DFS order.
  std::vector<aiNode *> nodes;
  std::function<void(aiNode *)> recursive_node_collector;
  recursive_node_collector = [&](aiNode *root) {
    nodes.push_back(root);
    for (int j = 0; j < root->mNumChildren; ++j) {
      recursive_node_collector(root->mChildren[j]);
    }
  };
  recursive_node_collector(scene->mRootNode);

  for (int node_index = 0; node_index < nodes.size(); ++node_index) {
    aiNode *node = nodes[node_index];
    for (int i = 0; i < node->mNumMeshes; ++i) {
      const int mesh_index = node->mMeshes[i];
      const aiMesh *mesh = scene->mMeshes[mesh_index];
      std::string group_name(mesh->mName.data);

      for (int j = 0; j < mesh->mNumVertices; ++j) {
        const auto &vertex = mesh->mVertices[j];
        if (std::abs(vertex.x) > kMaxVertexValue || std::abs(vertex.y) > kMaxVertexValue || std::abs(vertex.z) > kMaxVertexValue) {
          LOGGER->error("vertex value above threshold: {}, {}, {}", vertex.x, vertex.y, vertex.z);
          throw std::runtime_error("");
        }
        vertices->push_back({vertex.x, vertex.y, vertex.z});
      }
      for (int j = 0; j < mesh->mNumFaces; ++j) {
        auto face = mesh->mFaces[j];
        Expects(face.mNumIndices == 3);
        for (int k = 0; k < 3; ++k) {
          if (face.mIndices[k] >= mesh->mNumVertices) {
            LOGGER->warn("Invalid vertex index found. Skipping.");
            continue;
          }
        }

        faces->push_back({face_offset + face.mIndices[0],
                          face_offset + face.mIndices[1],
                          face_offset + face.mIndices[2]});

        face_group_names->push_back(group_name);

        ++triangle_count;
      }
      face_offset += mesh->mNumVertices;
      Ensures(face_offset == vertices->size());
    }
  }

  for (int i = 0; i < scene->mNumMeshes; ++i) {
  }

  if (triangle_count <= 0) {
    LOGGER->error("No triangles found in mesh file.");
  }

  return true;
}

void ParseCategoryMapping(const string &csv_filename, map<string, CategoryMappingEntry> *model_id_to_category) {
  io::CSVReader<8> csv_reader(csv_filename);
  csv_reader.read_header(io::ignore_extra_column, "index", "model_id", "fine_grained_class", "coarse_grained_class", "empty_struct_obj", "nyuv2_40class", "wnsynsetid", "wnsynsetkey");
  CategoryMappingEntry entry;
  string model_id;
  while (csv_reader.read_row(entry.index, model_id, entry.fine_grained_class, entry.coarse_grained_class, entry.empty_struct_obj, entry.nyuv2_40class, entry.wnsynsetid, entry.wnsynsetkey)) {
    (*model_id_to_category)[model_id] = entry;
  }
}

}  // end of namespace suncg

PerspectiveCamera MakeCamera(const suncg::CameraParams &params, double near, double far) {
  double hw_ratio = std::tan(params.y_fov) / std::tan(params.x_fov);
  FrustumParams frustum = MakePerspectiveFrustumParams(hw_ratio, params.x_fov, near, far);
  return {params.cam_eye, params.cam_eye + params.cam_view_dir, params.cam_up, frustum};
}
}


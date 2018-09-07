//
// Created by daeyun on 8/28/18.
//

#pragma once

#include <fstream>

#include "common.h"
#include "camera.h"

#include "third_party/repos/lrucache11/LRUCache11.hpp"

namespace scene3d {
namespace suncg {

struct CameraParams {
  Vec3 cam_eye;
  Vec3 cam_view_dir;
  Vec3 cam_up;
  double x_fov = 0;
  double y_fov = 0;
  double score = 0;  // scene coverage score. not used at the moment.
};

enum class InstanceType {
  WallInside, WallOutside, Object, Box, Ground, Floor, Ceiling
};

struct Instance {
  string id;  // Same as instance id.
  string model_id;  // Used to determine object category. "Box" for box type. Can be empty for floor, wall, etc.
  string room_id;  // Can be empty. Room envelope will not have room ids, for now.
  InstanceType type;
};

struct CategoryMappingEntry {
  uint16_t index;
  string fine_grained_class;
  string coarse_grained_class;
  string empty_struct_obj;
  string nyuv2_40class;
  string wnsynsetid;
  string wnsynsetkey;
};

// Run `Build()` before use.
class Scene {
 public:
  Scene(const string &house_json_filename, const string &house_obj_filename, const string &category_mapping_csv_filename);

  void Build();

  inline const Instance &PrimIdToInstance(unsigned int prim_id) const {
    return instance_id_to_node.at(face_instance_ids[prim_id]);
  }

  const CategoryMappingEntry &PrimIdToCategory(unsigned int prim_id) const {
    const CategoryMappingEntry *cached;
    if (prim_to_cat_cache_.tryGet(prim_id, cached)) {
      return *cached;
    }

    const Instance &instance = PrimIdToInstance(prim_id);

    switch (instance.type) {
      case InstanceType::WallInside:
      case InstanceType::WallOutside: return model_id_to_category.at("Wall");
      case InstanceType::Floor:
      case InstanceType::Ground: return model_id_to_category.at("Floor");
      case InstanceType::Ceiling: return model_id_to_category.at("Ceiling");
      case InstanceType::Box: return model_id_to_category.at("Box");
      default: break;
    }

    // Object
    const auto &it = model_id_to_category.find(instance.model_id);
    if (it == model_id_to_category.end()) {
      LOGGER->error("prim_id {} has uncategoried model_id {}", prim_id, instance.model_id);
    }

    prim_to_cat_cache_.insert(prim_id, &it->second);
    return it->second;
  }

  const array<float, 3> &PrimNormal(unsigned int prim_id) {
    if (!has_normal[prim_id]) {
      const auto &face = this->faces[prim_id];
      const auto &a = this->vertices[face[0]];
      const auto &b = this->vertices[face[1]];
      const auto &c = this->vertices[face[2]];
      const Vec3 va = {a[0], a[1], a[2]};
      const Vec3 vb = {b[0], b[1], b[2]};
      const Vec3 vc = {c[0], c[1], c[2]};
      const Vec3 normal = (vb - va).cross(vc - va).normalized();
      face_normals[prim_id][0] = (float) normal[0];
      face_normals[prim_id][1] = (float) normal[1];
      face_normals[prim_id][2] = (float) normal[2];
      has_normal[prim_id] = true;
    }

    return face_normals[prim_id];
  }

  const bool IsPrimBackground(unsigned int prim_id) const {
    bool cached;
    if (is_prim_background_cache_.tryGet(prim_id, cached)) {
      return cached;
    }

    const auto &catetory = PrimIdToCategory(prim_id);
    const auto &nyu40_category = catetory.nyuv2_40class;
    const auto &coarse_grained = catetory.coarse_grained_class;

    // TV?
    bool ret = nyu40_category == "wall" ||
        nyu40_category == "floor" ||
        nyu40_category == "ceiling" ||
        nyu40_category == "door" ||
        nyu40_category == "floor_mat" ||
        nyu40_category == "window" ||
        nyu40_category == "curtain" ||
        nyu40_category == "blinds" ||
        nyu40_category == "picture" ||
        nyu40_category == "mirror" ||
        nyu40_category == "fireplace" ||
        coarse_grained == "roof" ||
        nyu40_category == "whiteboard";

    is_prim_background_cache_.insert(prim_id, ret);
    return ret;
  }

  vector<array<unsigned int, 3>> faces;
  vector<array<float, 3>> vertices;
  vector<array<float, 3>> face_normals;
  vector<bool> has_normal;

  vector<string> face_instance_ids;
  map<string, Instance> instance_id_to_node;
  map<string, CategoryMappingEntry> model_id_to_category;

  string source_json_filename;
  string source_obj_filename;
  string source_category_mapping_csv_filename;

 private:
  mutable lru11::Cache<unsigned int, const CategoryMappingEntry *> prim_to_cat_cache_{16, 0};
  mutable lru11::Cache<unsigned int, bool> is_prim_background_cache_{16, 0};
};

}  // end of namespace suncg

PerspectiveCamera MakeCamera(const suncg::CameraParams &params, double near, double far);

namespace suncg {
void ReadCameraFile(const string &filename, vector<suncg::CameraParams> *suncg_params);

// Write in the same format as SunCG.
void WriteCameraFile(const string &filename, const vector<PerspectiveCamera> &cameras);

void ReadCameraFile(const string &filename, vector<PerspectiveCamera> *cameras);

void ParseObjectRoomHierarchy(const string &house_json_filename, map<string, Instance> *node_id_to_node);

bool ReadFacesAndVertices(const std::string &filename,
                          std::vector<std::array<unsigned int, 3>> *faces,
                          std::vector<std::array<float, 3>> *vertices,
                          std::vector<string> *face_group_names);

void ParseCategoryMapping(const string &csv_filename, map<string, CategoryMappingEntry> *model_id_to_category);

}  // end of namespace suncg
}

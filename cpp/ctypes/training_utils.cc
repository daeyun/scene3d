//
// Created by daeyun on 9/9/18.
//

#include <string>
#include <vector>
#include <omp.h>
#include "lib/common.h"
#include "lib/suncg_utils.h"
#include "lib/benchmark.h"

using scene3d::suncg::CategoryMappingEntry;
using std::string;
using std::vector;
using std::set;

using Selector = std::function<string(const CategoryMappingEntry &)>;

const uint16_t kVoid = 65535;  // Max value for uint16

//map<uint16_t, uint16_t> model_index_to_fine_grained_class_index;
//map<uint16_t, uint16_t> model_index_to_coarse_grained_class_index;
//map<uint16_t, uint16_t> model_index_to_empty_struct_obj_index;
map<uint16_t, uint16_t> model_index_to_nyuv2_40class_index;
//map<uint16_t, uint16_t> model_index_to_nyuv2_wnsynsetid_index;
//map<uint16_t, uint16_t> model_index_to_nyuv2_wnsynsetkey_index;
map<uint16_t, bool> is_model_index_background;
volatile bool is_initialized{false};

// Python naming convention.
extern "C" {
void model_index_to_category(const char *mapping_name,
                             uint16_t *data,
                             uint32_t num_items);

void initialize_category_mapping(const char *csv_file_name);

}

void MapNameToSortedIndex(const map<string, CategoryMappingEntry> &model_id_to_category_mapping, const Selector &selector, map<string, uint16_t> *out) {
  set<string> names;
  for (const auto &kv : model_id_to_category_mapping) {
    // Ignore "empty" category when assigning indices.
    if (kv.second.index != 0) {
      names.insert(selector(kv.second));
    }
  }
  vector<string> output(names.size());
  std::copy(names.begin(), names.end(), output.begin());
  std::sort(output.begin(), output.end());  // Set guarantees sorting. but just in case. Remove this later.
  Ensures(output.size() <= std::numeric_limits<uint16_t>::max());
  for (int i = 0; i < output.size(); ++i) {
    (*out)[output[i]] = static_cast<uint16_t>(i);
  }

  // Regardless of which mapping we're using, reserve those category names for void.
  (*out)["Empty"] = kVoid;
  (*out)["empty"] = kVoid;
  (*out)["void"] = kVoid;
  (*out)["empty.n.01"] = kVoid;
  (*out)["n03284308"] = kVoid;
}

void initialize_category_mapping(const char *csv_file_name) {
  if (spdlog::get("console") == nullptr) {
    spdlog::stdout_color_mt("console");
  }
  if (is_initialized) {
    LOGGER->error("Category mapping is already initialized.");
    return;
  }
  std::string csv_file_name_str(csv_file_name);
  map<string, CategoryMappingEntry> model_id_to_category_mapping;
  ParseCategoryMapping(csv_file_name_str, &model_id_to_category_mapping);

  map<string, uint16_t> nyu40_name_to_index;
  MapNameToSortedIndex(model_id_to_category_mapping, [](const CategoryMappingEntry &entry) -> string {
    return entry.nyuv2_40class;
  }, &nyu40_name_to_index);

  for (const auto &kv : model_id_to_category_mapping) {
    auto it = model_index_to_nyuv2_40class_index.find(kv.second.index);
    Ensures(it == model_index_to_nyuv2_40class_index.end());
    model_index_to_nyuv2_40class_index[kv.second.index] = nyu40_name_to_index[kv.second.nyuv2_40class];

    const string &nyu40_category = kv.second.nyuv2_40class;
    const string &coarse_grained = kv.second.coarse_grained_class;

    // This should be the same as suncg_utils.h
    bool is_background = nyu40_category == "wall" ||
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
    is_model_index_background[kv.second.index] = is_background;
  }

  is_initialized = true;
}

void model_index_to_category(const char *mapping_name,
                             uint16_t *data,
                             uint32_t num_items) {
  if (spdlog::get("console") == nullptr) {
    spdlog::stdout_color_mt("console");
  }
  if (!is_initialized) {
    LOGGER->error("Category mapping was not initialized.");
    throw std::runtime_error("");
  }

  // TODO(daeyun): not used at the moment. Assume this is NYU40.
  const std::string mapping_name_str(mapping_name);

  if (mapping_name_str == "nyuv2_40class_merged_background") {
    for (uint32_t i = 0; i < num_items; ++i) {
      uint16_t model_index = *(data + i);
      auto it = model_index_to_nyuv2_40class_index.find(model_index);
      Ensures(it != model_index_to_nyuv2_40class_index.end());
      if (is_model_index_background[model_index]) {
        *(data + i) = kVoid;  // Group background into "void" category.
      } else {
        *(data + i) = it->second;  // This includes void.
      }
    }
  } else if (mapping_name_str == "nyuv2_40class") {
    for (uint32_t i = 0; i < num_items; ++i) {
      uint16_t model_index = *(data + i);
      auto it = model_index_to_nyuv2_40class_index.find(model_index);
      Ensures(it != model_index_to_nyuv2_40class_index.end());
      *(data + i) = it->second;
    }
  } else {
    LOGGER->error("Unrecognized mapping name: {}", mapping_name_str);
    throw std::runtime_error("");
  }
}


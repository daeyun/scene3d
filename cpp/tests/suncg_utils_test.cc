//
// Created by daeyun on 8/27/18.
//

#include "catch.hpp"
#include "spdlog/spdlog.h"

#include "boost/filesystem.hpp"

#include "lib/common.h"
#include "lib/camera.h"
#include "lib/suncg_utils.h"
#include "lib/file_io.h"

using namespace scene3d;
namespace fs = boost::filesystem;

TEST_CASE("parse single house") {
  std::string csv_filename = "resources/ModelCategoryMapping.csv";
  SECTION("915a66f48ec925febf644f962375720d") {
    std::string obj_filename = "resources/house/915a66f48ec925febf644f962375720d/house.obj";
    std::string json_filename = "resources/house/915a66f48ec925febf644f962375720d/house_p.json";
    suncg::Scene scene(json_filename, obj_filename, csv_filename);
    scene.Build();

    auto num_faces = static_cast<unsigned int>(scene.faces.size());
    for (unsigned int i = 0; i < num_faces; ++i) {
      const auto &category = scene.PrimIdToCategory(i);
      REQUIRE(!category.nyuv2_40class.empty());
    }
  };
  SECTION("0004d52d1aeeb8ae6de39d6bd993e992") {
    std::string obj_filename = "resources/house/0004d52d1aeeb8ae6de39d6bd993e992/house.obj";
    std::string json_filename = "resources/house/0004d52d1aeeb8ae6de39d6bd993e992/house_p.json";
    suncg::Scene scene(json_filename, obj_filename, csv_filename);
    scene.Build();

    auto num_faces = static_cast<unsigned int>(scene.faces.size());
    for (unsigned int i = 0; i < num_faces; ++i) {
      const auto &category = scene.PrimIdToCategory(i);
      REQUIRE(!category.nyuv2_40class.empty());
    }
  };

}


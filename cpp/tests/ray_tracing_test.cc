#include "catch.hpp"
#include "spdlog/spdlog.h"

#include "lib/common.h"
#include "lib/ray_mesh_intersection.h"

TEST_CASE("Intersection") {
  const std::vector<std::array<float, 3>> vertices{
      array<float, 3>{0.0, 0.0, 0.0},
      array<float, 3>{1.0, 0.0, 0.0},
      array<float, 3>{0.0, 1.0, 0.0},
      array<float, 3>{0.1, 0.1, 0.0},
      array<float, 3>{0.9, 0.1, 0.0},
      array<float, 3>{0.1, 0.9, 0.0},
      array<float, 3>{0.0, 0.0, -0.5f},
      array<float, 3>{1.0, 0.0, -0.5f},
      array<float, 3>{0.0, 1.0, -0.5f},
  };

  const std::vector<std::array<unsigned int, 3>> faces{
      array<unsigned int, 3>{0, 1, 2},
      array<unsigned int, 3>{3, 4, 5},
      array<unsigned int, 3>{6, 7, 8},
  };

  scenecompletion::AABBTree tree(faces, vertices);

  SECTION("Intersect three triangles") {
    auto ray = scenecompletion::Ray{
        .o = {0.2, 0.2, 1.0},
        .d = {0, 0, -1.0},
    };

    std::vector<scenecompletion::RayIntersection> hits;
    tree.MultiHitTraverse(ray, &hits);

    REQUIRE(3 == hits.size());

    REQUIRE(Approx(1.0) == hits[0].t);
    REQUIRE(Approx(1.0) == hits[1].t);
    REQUIRE(Approx(1.5) == hits[2].t);

    REQUIRE(2 == hits[2].prim_id);
  }

  SECTION("Intersect three triangles in reverse order") {
    auto ray = scenecompletion::Ray{
        .o = {0.2, 0.2, -1.0},
        .d = {0, 0, 1.0},
    };

    std::vector<scenecompletion::RayIntersection> hits;
    tree.MultiHitTraverse(ray, &hits);

    REQUIRE(3 == hits.size());

    REQUIRE(Approx(0.5) == hits[0].t);
    REQUIRE(Approx(1.0) == hits[1].t);
    REQUIRE(Approx(1.0) == hits[2].t);

    REQUIRE(2 == hits[0].prim_id);
  }

  SECTION("Intersect two triangles") {
    auto ray = scenecompletion::Ray{
        .o = {0.2, 0.2, -0.25},
        .d = {0, 0, 1.0},
    };

    std::vector<scenecompletion::RayIntersection> hits;
    tree.MultiHitTraverse(ray, &hits);

    REQUIRE(2 == hits.size());

    REQUIRE(Approx(0.25) == hits[0].t);
    REQUIRE(Approx(0.25) == hits[1].t);
  }

  SECTION("Intersect two triangles 2") {
    auto ray = scenecompletion::Ray{
        .o = {0.05, 0.05, 1.0},
        .d = {0, 0, -1.0},
    };

    std::vector<scenecompletion::RayIntersection> hits;
    tree.MultiHitTraverse(ray, &hits);

    REQUIRE(2 == hits.size());

    REQUIRE(Approx(1.0) == hits[0].t);
    REQUIRE(Approx(1.5) == hits[1].t);

    REQUIRE(0 == hits[0].prim_id);
    REQUIRE(2 == hits[1].prim_id);
  }
}



#pragma once

#include <list>

#include "common.h"

namespace scenecompletion {

// Forward declation for encapsulation.
struct CGAL_Triangles;
struct CGAL_Tree;

struct Ray {
  Vec3 o;
  Vec3 d;
};

struct RayIntersection {
  float t;
  unsigned int prim_id;
};

class AABBTree {
 public:
  AABBTree(const std::vector<std::array<unsigned int, 3>> &faces, const std::vector<std::array<float, 3>> &vertices);
  ~AABBTree();

  size_t MultiHitTraverse(const Ray &ray, std::vector<RayIntersection> *hits);

 private:
  std::unique_ptr<CGAL_Triangles> triangles_;
  std::unique_ptr<CGAL_Tree> tree_;
};
}

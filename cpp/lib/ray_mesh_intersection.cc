#include "ray_mesh_intersection.h"

#include "common.h"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/squared_distance_3.h>

namespace scenecompletion {
typedef CGAL::Simple_cartesian<double> K;
typedef K::Triangle_3 CGAL_Triangle;
typedef std::list<CGAL_Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> CGAL_AABB_Tree;
typedef boost::optional<CGAL_AABB_Tree::Intersection_and_primitive_id<K::Ray_3>::Type> CGAL_Ray_intersection;

struct CGAL_Triangles {
  std::list<CGAL_Triangle> triangles;
};

struct CGAL_Tree {
  std::unique_ptr<CGAL_AABB_Tree> tree;
};

AABBTree::AABBTree(const std::vector<std::array<unsigned int, 3>> &faces, const std::vector<std::array<float, 3>> &vertices) :
    triangles_(std::make_unique<CGAL_Triangles>()), tree_(std::make_unique<CGAL_Tree>()) {
  for (const auto &face: faces) {
    const auto &a = vertices[face[0]];
    const auto &b = vertices[face[1]];
    const auto &c = vertices[face[2]];
    triangles_->triangles.push_back(CGAL_Triangle(K::Point_3{a[0], a[1], a[2]}, K::Point_3{b[0], b[1], b[2]}, K::Point_3{c[0], c[1], c[2]}));
  }

  tree_ = std::make_unique<CGAL_Tree>();
  tree_->tree = std::make_unique<CGAL_AABB_Tree>(triangles_->triangles.begin(), triangles_->triangles.end());
  tree_->tree->accelerate_distance_queries();
}

size_t AABBTree::MultiHitTraverse(const Ray &ray, std::vector<RayIntersection> *hits) {
  K::Point_3 o{ray.o[0], ray.o[1], ray.o[2]};
  K::Vector_3 d{ray.d[0], ray.d[1], ray.d[2]};
  K::Ray_3 ray_query(o, d);

  std::list<CGAL_Ray_intersection> intersections;
  tree_->tree->all_intersections(ray_query, std::back_inserter(intersections));

  for (const auto &intersection : intersections) {
    if (intersection) {
      const K::Point_3 p = *boost::get<K::Point_3>(&(intersection->first));
      CGAL_AABB_Tree::Primitive_id s = intersection->second;

      auto face_index = std::distance(triangles_->triangles.begin(), s);
      float dist = std::sqrt(CGAL::squared_distance(o, p));

      hits->push_back(RayIntersection{
          .t=dist,
          .prim_id=static_cast<unsigned int>(face_index)
      });
    }
  }

  std::sort(hits->begin(), hits->end(), [](const RayIntersection &a, const RayIntersection &b) {
    return a.t < b.t;
  });

  return hits->size();
}

AABBTree::~AABBTree() {}

}

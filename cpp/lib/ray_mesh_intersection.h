//
// Created by daeyun on 4/27/18.
//

#pragma once

#include <functional>

#include "lib/common.h"

#include "nanort.h"

namespace scene3d {
class RayTracer {
 public:
  RayTracer(const std::vector<std::array<unsigned int, 3>> &faces,
            const std::vector<std::array<float, 3>> &vertices)
      : faces_(faces), vertices_(vertices), triangle_intersector_(vertices_.data()->data(), faces_.data()->data(), sizeof(float) * 3) {
    nanort::TriangleMesh<float> triangle_mesh(vertices_.data()->data(), faces_.data()->data(), sizeof(float) * 3);
    nanort::TriangleSAHPred<float> triangle_pred(vertices_.data()->data(), faces_.data()->data(), sizeof(float) * 3);
    nanort::BVHBuildOptions<float> build_options;
    build_options.cache_bbox = true;  // ~1 second difference when loading suncg house models.

    bool build_ok = accel_.Build(static_cast<const unsigned int>(faces_.size()), triangle_mesh, triangle_pred, build_options);
    Ensures(build_ok);

  }

  void Traverse(const Vec3 &ray_origin, const Vec3 &ray_direction, std::function<bool(float t, float u, float v, unsigned int prim_id)> callback) const {
    nanort::Ray<float> ray;

    ray.org[0] = static_cast<float>(ray_origin[0]);
    ray.org[1] = static_cast<float>(ray_origin[1]);
    ray.org[2] = static_cast<float>(ray_origin[2]);

    ray.dir[0] = static_cast<float>(ray_direction[0]);
    ray.dir[1] = static_cast<float>(ray_direction[1]);
    ray.dir[2] = static_cast<float>(ray_direction[2]);

    std::set<unsigned int> ignored_prim_ids;
    nanort::BVHTraceOptions trace_options;
    trace_options.ignored_prim_ids = &ignored_prim_ids;

    while (true) {
      nanort::TriangleIntersection<> isect{};
      bool hit = accel_.Traverse(ray, triangle_intersector_, &isect, trace_options);

      if (hit) {
        bool keep_going = callback(isect.t, isect.u, isect.v, isect.prim_id);

        if (!keep_going) {
          break;
        }

        ray.min_t = static_cast<float>(isect.t - kEps);
        ignored_prim_ids.insert(isect.prim_id);

      } else {
        break;
      }
    }
  }

  void PrintStats() const {
    nanort::BVHBuildStatistics stats = accel_.GetStatistics();

    printf("  BVH statistics:\n");
    printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
    printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
    printf("  Max tree depth     : %d\n", stats.max_tree_depth);
    float bmin[3], bmax[3];
    accel_.BoundingBox(bmin, bmax);
    printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
    printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);
  }

  const Vec3 vertex(size_t i) {
    return Vec3{vertices_[i][0], vertices_[i][1], vertices_[i][2]};
  }

  const std::array<unsigned int, 3> face(size_t i) {
    return {faces_[i][0], faces_[i][1], faces_[i][2]};
  }

 private:
  std::vector<std::array<unsigned int, 3>> faces_;
  std::vector<std::array<float, 3>> vertices_;

  nanort::BVHAccel<float> accel_;
  nanort::TriangleIntersector<> triangle_intersector_;
  const float kEps = 0.0001;
};
}

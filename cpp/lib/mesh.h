#pragma once

#include "common.h"
#include "camera.h"

namespace scene3d {

class TriMesh {
 public:
  vector<array<unsigned int, 3>> faces;
  vector<array<float, 3>> vertices;

  void AddTriangle(const array<float, 3> &v0, const array<float, 3> &v1, const array<float, 3> &v2) {
    auto offset = static_cast<unsigned int>(vertices.size());
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    faces.push_back(array<unsigned int, 3>{offset, offset + 1, offset + 2});
  }

  void AddTriangle(const Vec3 &v0, const Vec3 &v1, const Vec3 &v2) {
    auto offset = static_cast<unsigned int>(vertices.size());
    vertices.push_back(array<float, 3>{(float) v0[0], (float) v0[1], (float) v0[2]});
    vertices.push_back(array<float, 3>{(float) v1[0], (float) v1[1], (float) v1[2]});
    vertices.push_back(array<float, 3>{(float) v2[0], (float) v2[1], (float) v2[2]});
    faces.push_back(array<unsigned int, 3>{offset, offset + 1, offset + 2});
  }

  void TruncateBelow(const Plane &plane, TriMesh *new_mesh) const {
    for (int i = 0; i < faces.size(); ++i) {
      const auto &face = faces[i];
      array<float, 3> v0 = vertices[face[0]];
      array<float, 3> v1 = vertices[face[1]];
      array<float, 3> v2 = vertices[face[2]];
      double d0 = plane.Displacement(v0);
      double d1 = plane.Displacement(v1);
      double d2 = plane.Displacement(v2);

      bool swap_02 = false;
      bool swap_01 = false;
      bool swap_12 = false;
      if (d0 < d2) {
        std::swap(d0, d2);
        std::swap(v0, v2);
        swap_02 = true;
      }
      if (d0 < d1) {
        std::swap(d0, d1);
        std::swap(v0, v1);
        swap_01 = true;
      }
      if (d1 < d2) {
        std::swap(d1, d2);
        std::swap(v1, v2);
        swap_12 = true;
      }

      const double kThreshold = 1e-4;

      if (d0 <= kThreshold) {
        continue;
      }

      // Undo swaps and save.
      auto add_triangle = [&](Vec3 v0, Vec3 v1, Vec3 v2) {
        if (swap_12) {
          std::swap(v1, v2);
        }
        if (swap_01) {
          std::swap(v0, v1);
        }
        if (swap_02) {
          std::swap(v0, v2);
        }
        new_mesh->AddTriangle(v0, v1, v2);
      };

      Vec3 vv0{v0[0], v0[1], v0[2]};
      Vec3 vv1{v1[0], v1[1], v1[2]};
      Vec3 vv2{v2[0], v2[1], v2[2]};

      if (d2 > -kThreshold) {
        add_triangle(vv0, vv1, vv2);
        continue;
      }

      if (d1 < 0) {
        double t01;
        Vec3 v01 = (vv1 - vv0).normalized();
        bool intersect = plane.IntersectRay(vv0, v01, &t01);
        Ensures(intersect);
        double t02;
        Vec3 v02 = (vv2 - vv0).normalized();
        Ensures(plane.IntersectRay(vv0, v02, &t02));
        add_triangle(vv0, vv0 + v01 * t01, vv0 + v02 * t02);
      } else {
        double t20;
        Vec3 v20 = (vv0 - vv2).normalized();
        Ensures(plane.IntersectRay(vv2, v20, &t20));
        double t21;
        Vec3 v21 = (vv1 - vv2).normalized();
        Ensures(plane.IntersectRay(vv2, v21, &t21));
        Vec3 new_v02 = vv2 + v20 * t20;
        Vec3 new_v12 = vv2 + v21 * t21;
        add_triangle(vv0, new_v12, new_v02);
        add_triangle(vv0, vv1, new_v12);
      }
    }
  }

  void RemoveFacesContainingVertices(const vector<bool> &is_vertex_invalid, TriMesh *new_mesh) const {
    Expects(is_vertex_invalid.size() == vertices.size());

    for (int i = 0; i < faces.size(); ++i) {
      bool remove_face = false;
      const auto &face = faces[i];
      for (int j = 0; j < 3; ++j) {
        if (is_vertex_invalid[face[j]]) {
          remove_face = true;
          break;
        }
      }

      if (remove_face) {
        continue;
      }

      new_mesh->AddTriangle(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
    }
  }

  void AddMesh(const TriMesh &mesh) {
    const auto offset = vertices.size();
    vertices.insert(vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
    for (int i = 0; i < mesh.faces.size(); ++i) {
      faces.push_back(array<unsigned int, 3>{
          static_cast<unsigned int>(mesh.faces[i][0] + offset),
          static_cast<unsigned int>(mesh.faces[i][1] + offset),
          static_cast<unsigned int>(mesh.faces[i][2] + offset),
      });
    }
  }

};

}


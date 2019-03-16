#pragma once

#include <limits>
#include <unordered_set>

#include "common.h"
#include "camera.h"
#include "benchmark.h"
#include "depth.h"
#include "multi_layer_depth_renderer.h"
#include "file_io.h"
#include "pcl.h"
#include "mesh.h"
#include "vectorization_utils.h"

namespace scene3d {

void RenderMultiLayerDepthImage(MultiLayerDepthRenderer *renderer,
                                MultiLayerImage<float> *ml_depth,
                                MultiLayerImage<uint32_t> *ml_prim_ids) {
  *ml_depth = MultiLayerImage<float>(static_cast<unsigned int>(renderer->height()),
                                     static_cast<unsigned int>(renderer->width()),
                                     NAN);
  *ml_prim_ids = MultiLayerImage<uint32_t>(static_cast<unsigned int>(renderer->height()),
                                           static_cast<unsigned int>(renderer->width()),
                                           std::numeric_limits<uint32_t>::max());

  for (unsigned int y = 0; y < renderer->height(); y++) {
    for (unsigned int x = 0; x < renderer->width(); x++) {
      vector<float> *depth_values = ml_depth->values(y, x);
      vector<uint32_t> *prim_ids = ml_prim_ids->values(y, x);
      renderer->DepthValues(x, y, depth_values, prim_ids);
    }
  }
}

void RenderObjectCenteredMultiLayerDepthImage(MultiLayerDepthRenderer *renderer,
                                              MultiLayerImage<float> *ml_depth,
                                              MultiLayerImage<uint32_t> *ml_prim_ids) {
  *ml_depth = MultiLayerImage<float>(static_cast<unsigned int>(renderer->height()),
                                     static_cast<unsigned int>(renderer->width()),
                                     NAN);
  *ml_prim_ids = MultiLayerImage<uint32_t>(static_cast<unsigned int>(renderer->height()),
                                           static_cast<unsigned int>(renderer->width()),
                                           std::numeric_limits<uint32_t>::max());

  for (unsigned int y = 0; y < renderer->height(); y++) {
    for (unsigned int x = 0; x < renderer->width(); x++) {
      vector<float> *depth_values = ml_depth->values(y, x);
      vector<uint32_t> *prim_ids = ml_prim_ids->values(y, x);
      renderer->ObjectCenteredRayDisplacement(x, y, depth_values, prim_ids);
    }
  }
}

AABB DepthCamBoundingBox(const Image<float> &depth_image, const Camera &camera) {
  Points2i xy;
  Points1d d;
  ValidPixelCoordinates(depth_image, &xy, &d);
  Points3d cam_out;
  camera.ImageToCam(xy, d, depth_image.height(), depth_image.width(), &cam_out);
  Vec3 bmax = cam_out.rowwise().maxCoeff();
  Vec3 bmin = cam_out.rowwise().minCoeff();

  AABB ret(bmin, bmax);
  return ret;
}

void GenerateMultiDepthExample(suncg::Scene *scene,
                               const MultiLayerImage<float> &full_ml_depth,
                               const MultiLayerImage<uint32_t> &ml_prim_ids,
                               MultiLayerImage<float> *out_ml_depth,
                               MultiLayerImage<uint16_t> *out_ml_model_indices,
                               MultiLayerImage<uint32_t> *out_ml_prim_ids) {
  Expects(full_ml_depth.width() == ml_prim_ids.width());
  Expects(full_ml_depth.height() == ml_prim_ids.height());

  unsigned int h = full_ml_depth.height();
  unsigned int w = full_ml_depth.width();

  *out_ml_depth = MultiLayerImage<float>(h, w, NAN);
  *out_ml_model_indices = MultiLayerImage<uint16_t>(h, w, std::numeric_limits<uint16_t>::max());
  *out_ml_prim_ids = MultiLayerImage<uint32_t>(h, w, std::numeric_limits<uint32_t>::max());

  // Populate instance ids of objects in the traditional depth.
  std::set<std::string> visible_instance_ids;
  for (unsigned int y = 0; y < full_ml_depth.height(); y++) {
    for (unsigned int x = 0; x < full_ml_depth.width(); x++) {
      vector<uint32_t> *prim_ids = ml_prim_ids.values(y, x);
      if (prim_ids->empty()) {
        continue;
      }
      const suncg::Instance &instance = scene->PrimIdToInstance(prim_ids->at(0));
      visible_instance_ids.insert(instance.id);
    }
  }

  auto should_triangle_be_included = [&](uint32_t prim_id) -> bool {
    if (scene->IsPrimBackground(prim_id)) {
      return true;
    }
    const suncg::Instance &instance_id = scene->PrimIdToInstance(prim_id);
    return visible_instance_ids.find(instance_id.id) != visible_instance_ids.end();
  };

  for (unsigned int y = 0; y < full_ml_depth.height(); y++) {
    for (unsigned int x = 0; x < full_ml_depth.width(); x++) {
      vector<float> *d = full_ml_depth.values(y, x);
      vector<uint32_t> *prim_ids = ml_prim_ids.values(y, x);
      vector<float> *out_d = out_ml_depth->values(y, x);
      vector<uint16_t> *out_m = out_ml_model_indices->values(y, x);
      vector<uint32_t> *out_p = out_ml_prim_ids->values(y, x);

      auto insert_layer = [&out_d, &out_m, &out_p, &d, &prim_ids, &scene](size_t ind) {
        Ensures(ind < d->size());
        out_d->push_back(d->at(ind));
        auto primid = prim_ids->at(ind);
        out_p->push_back(primid);
        out_m->push_back(scene->PrimIdToCategory(primid).index);
      };
      auto insert_nan = [&out_d, &out_m, &out_p, &d, &prim_ids, &scene]() {
        out_d->push_back(NAN);
        out_p->push_back(std::numeric_limits<uint32_t>::max());  // Should be treated as NaN.
        out_m->push_back(0);  // The "Empty" model/category is index 0. According to ModelCategoryMapping.csv
      };

      Expects(d->size() == prim_ids->size());

      // If this pixel is empty (ray did not hit an object or background), all five layers will be NaN.
      if (d->empty()) {
        insert_nan();
        insert_nan();
        insert_nan();
        insert_nan();
        insert_nan();
      } else {
        bool has_background = scene->IsPrimBackground(prim_ids->at(prim_ids->size() - 1));
        bool is_first_layer_background = scene->IsPrimBackground(prim_ids->at(0));
        if (is_first_layer_background) {
          // shortcut, for performance reasons.
          insert_nan();
          insert_nan();
          insert_nan();
          insert_nan();
          insert_layer(0);
        } else {
          // There must be at least one object here because first layer is not background.
          std::vector<size_t> object_instance_entry_indices;
          std::vector<string> object_instance_entry_ids;
          {
            for (size_t j = 0; j < prim_ids->size(); ++j) {
              const auto &prim_id = prim_ids->at(j);
              if (not should_triangle_be_included(prim_id)) {
                continue;
              }

              const suncg::Instance &instance_j_hit = scene->PrimIdToInstance(prim_id);
              if (std::find(object_instance_entry_ids.begin(), object_instance_entry_ids.end(), instance_j_hit.id) == object_instance_entry_ids.end()
                  and not scene->IsPrimBackground(prim_ids->at(j))) {
                object_instance_entry_ids.push_back(instance_j_hit.id);
                object_instance_entry_indices.push_back(j);
              }
              if (object_instance_entry_indices.size() == 2) {
                break;  // We only need the first two instances.
              }
            }
          }

          std::vector<size_t> object_instance_exit_indices(object_instance_entry_indices.size());
          int insert_count = 0;
          {
            std::set<string> object_instance_ids;
            for (int j = prim_ids->size() - 1; j >= 0; --j) {
              const auto &prim_id = prim_ids->at(j);
              if (not should_triangle_be_included(prim_id)) {
                continue;
              }

              const suncg::Instance &instance_j_hit = scene->PrimIdToInstance(prim_id);
              if (object_instance_ids.find(instance_j_hit.id) == object_instance_ids.end() and not scene->IsPrimBackground(prim_ids->at(j))) {
                // New object found.
                object_instance_ids.insert(instance_j_hit.id);
                // If it's any of the two found in the forward pass, save them.
                if (instance_j_hit.id == object_instance_entry_ids[0]) {
                  object_instance_exit_indices[0] = j;
                  insert_count++;
                } else if ((object_instance_entry_ids.size() > 1) and (instance_j_hit.id == object_instance_entry_ids[1])) {
                  object_instance_exit_indices[1] = j;
                  insert_count++;
                }
              }
              if (insert_count == object_instance_entry_indices.size()) {
                break;  // We only need the first one or two instances.
              }
            }
          }

          Ensures(object_instance_entry_indices.size() == object_instance_exit_indices.size());
          Ensures(!object_instance_entry_indices.empty());

          if (object_instance_entry_indices.size() == 1) {
            insert_layer(object_instance_entry_indices[0]);
            insert_layer(object_instance_exit_indices[0]);
            insert_nan();
            insert_nan();
          } else {
            // those two object depths aren't always ordered. e.g. interlocking objects like desk and chair.
            insert_layer(object_instance_entry_indices[0]);
            insert_layer(object_instance_exit_indices[0]);
            insert_layer(object_instance_entry_indices[1]);
            insert_layer(object_instance_exit_indices[1]);
          }
          if (has_background) {
            insert_layer(prim_ids->size() - 1);
          } else {
            insert_nan();
          }
        }
      }
    }
  }
}

void ExtractBackgroundFromFourLayerModel(const MultiLayerImage<float> &ml_depth, Image<float> *out) {
  Expects(ml_depth.NumLayers() == 4);

  const unsigned int height = ml_depth.height();
  const unsigned int width = ml_depth.width();

  out->Resize(height, width);

  for (unsigned int y = 0; y < height; ++y) {
    for (unsigned int x = 0; x < width; ++x) {
      if (std::isnan(ml_depth.at(y, x, 1))) {
        out->at(y, x) = ml_depth.at(y, x, 0);
      } else {
        out->at(y, x) = ml_depth.at(y, x, 3);
      }
    }
  }
}

// returns estimated floor height (TODO: this is deprecated).
// A reasonable dd_factor value is 10.
float ExtractFrustumMesh(suncg::Scene *scene,
                         const scene3d::Camera &camera,
                         unsigned int height,
                         unsigned int width,
                         float dd_factor,
                         TriMesh *out_mesh_background_only,
                         TriMesh *out_mesh_object_only) {
  scene3d::RayTracer ray_tracer(scene->faces, scene->vertices);
  ray_tracer.PrintStats();

  double start_time = scene3d::TimeSinceEpoch<std::milli>();
  auto renderer = scene3d::SunCgMultiLayerDepthRenderer(
      &ray_tracer,
      &camera,
      width,
      height,
      scene
  );

  auto ml_depth = MultiLayerImage<float>(height, width, NAN);
  auto ml_prim_ids = MultiLayerImage<uint32_t>(height, width, std::numeric_limits<uint32_t>::max());
  RenderMultiLayerDepthImage(&renderer, &ml_depth, &ml_prim_ids);

  MultiLayerImage<float> out_ml_depth;
  MultiLayerImage<uint16_t> out_ml_model_indices;
  MultiLayerImage<uint32_t> out_ml_prim_ids;
  GenerateMultiDepthExample(scene, ml_depth, ml_prim_ids, &out_ml_depth, &out_ml_model_indices, &out_ml_prim_ids);

  Image<float> background(out_ml_depth.height(), out_ml_depth.width(), NAN);

  // TODO: make sure this is ok
//  ExtractBackgroundFromFourLayerModel(out_ml_depth, &background);
  out_ml_depth.ExtractLayer(4, &background);

  LOGGER->info("Elapsed (ExtractFrustumMesh: depth rendering): {} ms",
               scene3d::TimeSinceEpoch<std::milli>() - start_time);

  start_time = scene3d::TimeSinceEpoch<std::milli>();
//  const float dd_factor = 10.0;
  TriangulateDepth(background,
                   camera,
                   dd_factor,
                   &out_mesh_background_only->faces,
                   &out_mesh_background_only->vertices);
  LOGGER->info("Elapsed (ExtractFrustumMesh: TriangulateDepth): {} ms",
               scene3d::TimeSinceEpoch<std::milli>() - start_time);


  // Floor detection
  // TODO: this may fail if the floor height varies, for some reason.
  float min_y = 1e10;
  for (const auto &v : out_mesh_background_only->vertices) {
    if (v[1] < min_y) {
      min_y = v[1];
    }
  }
  // end of floor height detection.


  start_time = scene3d::TimeSinceEpoch<std::milli>();
  std::unordered_set<uint16_t> visible_model_indices;
  out_ml_model_indices.UniqueValues(&visible_model_indices);

  std::unordered_set<uint32_t> unique_prim_ids;
  ml_prim_ids.UniqueValues(&unique_prim_ids);
  std::unordered_map<unsigned int, unsigned int> new_vertex_mapping;

  TriMesh visible_triangles_mesh;
  for (unsigned int prim_id = 0; prim_id < scene->faces.size(); ++prim_id) {
    if (scene->IsPrimBackground(prim_id)) {
      continue;
    }

    const auto &category = scene->PrimIdToCategory(prim_id);
    if (visible_model_indices.find(category.index) == visible_model_indices.end()) {
      // Face does not belong to a visible object.
      continue;
    }

    array<unsigned int, 3> &face = scene->faces[prim_id];
    for (int i = 0; i < 3; ++i) {
      if (new_vertex_mapping.find(face[i]) == new_vertex_mapping.end()) {
        new_vertex_mapping[face[i]] = static_cast<unsigned int>(visible_triangles_mesh.vertices.size());
        visible_triangles_mesh.vertices.push_back(scene->vertices[face[i]]);
      }
    }
    visible_triangles_mesh.faces.push_back(array<unsigned int, 3>{new_vertex_mapping[face[0]],
                                                                  new_vertex_mapping[face[1]],
                                                                  new_vertex_mapping[face[2]]});
  }
  LOGGER->info("Elapsed (ExtractFrustumMesh: visibility test): {} ms",
               scene3d::TimeSinceEpoch<std::milli>() - start_time);

  start_time = scene3d::TimeSinceEpoch<std::milli>();
  array<Plane, 6> planes;
  camera.WorldFrustumPlanes(&planes);

  // TODO: This should be refactored.
  TriMesh mesh0;
  visible_triangles_mesh.TruncateBelow(planes[0], &mesh0);
  TriMesh mesh1;
  mesh0.TruncateBelow(planes[1], &mesh1);
  TriMesh mesh2;
  mesh1.TruncateBelow(planes[2], &mesh2);
  TriMesh mesh3;
  mesh2.TruncateBelow(planes[3], &mesh3);
  TriMesh mesh4;
  mesh3.TruncateBelow(planes[4], &mesh4);

  Points3d vertex_pts;
  ToEigen(mesh4.vertices, &vertex_pts);

  Points2i image_xy;
  Points1d cam_depth_value;
  camera.WorldToImage(vertex_pts, background.height(), background.width(), &image_xy, &cam_depth_value);

  Ensures(image_xy.cols() == vertex_pts.cols());
  vector<bool> is_vertex_invalid;
  is_vertex_invalid.resize(vertex_pts.cols());

  const double kProjThreshold = 0.1;

  for (int j = 0; j < image_xy.cols(); ++j) {
    const Vec2i xy = image_xy.col(j);
    if (xy[0] < 0 || xy[0] >= background.width()) {
      continue;
    }
    if (xy[1] < 0 || xy[1] >= background.height()) {
      continue;
    }
    const auto projected_depth = (float) cam_depth_value[j];
    const auto image_depth = background.at((unsigned int) xy[1], (unsigned int) xy[0]);
    is_vertex_invalid[j] = std::isfinite(image_depth) && projected_depth > image_depth + kProjThreshold;
  }

  mesh4.RemoveFacesContainingVertices(is_vertex_invalid, out_mesh_object_only);
  LOGGER->info("Elapsed (ExtractFrustumMesh: clipping): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);

  return min_y;
}

}

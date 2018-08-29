#pragma once

#include "common.h"
#include "camera.h"
#include "depth.h"
#include "multi_layer_depth_renderer.h"
#include "file_io.h"

namespace scene3d {

static void RenderMultiLayerDepthImage(const std::string &obj_filename, const Camera &camera, unsigned int height, unsigned int width, MultiLayerImage<float> *ml_depth) {
  std::vector<std::array<unsigned int, 3>> faces;
  std::vector<std::array<float, 3>> vertices;
  std::vector<int> prim_id_to_node_id;
  std::vector<std::string> prim_id_to_node_name;

  bool ok = ReadFacesAndVertices(obj_filename, &faces, &vertices, &prim_id_to_node_id, &prim_id_to_node_name);

  Ensures(ok);

  // Sanity check.
  Ensures(faces.size() == prim_id_to_node_name.size());
  Ensures(faces.size() == prim_id_to_node_name.size());

  LOGGER->info("{} faces, {} vertices", faces.size(), vertices.size());

  RayTracer ray_tracer(faces, vertices);

  auto renderer = unique_ptr<SunCgMultiLayerDepthRenderer>();

  double x_fov, y_fov;
  bool is_perspective = camera.fov(&x_fov, &y_fov);

  if (is_perspective) {
    renderer = make_unique<SunCgMultiLayerDepthRenderer>(
        &ray_tracer,
        camera.position(),
        camera.viewing_direction(),
        camera.up(),
        x_fov,
        y_fov,
        width,
        height,
        0,
        prim_id_to_node_name,
        false,
        0, 0, 0, 0  // TODO
    );
  } else {
    renderer = make_unique<SunCgMultiLayerDepthRenderer>(
        &ray_tracer,
        camera.position(),
        camera.viewing_direction(),
        camera.up(),
        1, 1,  // TODO
        width,
        height,
        0,
        prim_id_to_node_name,
        true,
        camera.frustum().left, camera.frustum().right, camera.frustum().top, camera.frustum().bottom
    );
  }

  renderer->ray_tracer()->PrintStats();

  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      vector<float> *depth_values = ml_depth->values(y, x);
      vector<string> model_ids;
      vector<unsigned int> prim_ids;
      int depth_value_index = renderer->DepthValues(x, y, depth_values, &model_ids, &prim_ids);
    }
  }
}

}

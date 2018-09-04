#pragma once

#include "common.h"
#include "camera.h"
#include "depth.h"
#include "multi_layer_depth_renderer.h"
#include "file_io.h"
#include "pcl.h"

namespace scene3d {

void RenderMultiLayerDepthImage(MultiLayerDepthRenderer *renderer, MultiLayerImage<float> *ml_depth, MultiLayerImage<uint32_t> *ml_prim_ids) {
  for (unsigned int y = 0; y < renderer->height(); y++) {
    for (unsigned int x = 0; x < renderer->width(); x++) {
      vector<float> *depth_values = ml_depth->values(y, x);
      vector<uint32_t> *prim_ids = ml_prim_ids->values(y, x);
      renderer->DepthValues(x, y, depth_values, prim_ids);
    }
  }
}

BoundingBox DepthCamBoundingBox(const Image<float> &depth_image, const Camera &camera) {
  Points2i xy;
  Points1d d;
  ValidPixelCoordinates(depth_image, &xy, &d);
  Points3d cam_out;
  camera.ImageToCam(xy, d, depth_image.height(), depth_image.width(), &cam_out);
  Vec3 bmax = cam_out.rowwise().maxCoeff();
  Vec3 bmin = cam_out.rowwise().minCoeff();

  BoundingBox ret(bmin, bmax);
  return ret;
}

}

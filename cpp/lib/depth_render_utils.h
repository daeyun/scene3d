#pragma once

#include <limits>

#include "common.h"
#include "camera.h"
#include "depth.h"
#include "multi_layer_depth_renderer.h"
#include "file_io.h"
#include "pcl.h"

namespace scene3d {

void RenderMultiLayerDepthImage(MultiLayerDepthRenderer *renderer, MultiLayerImage<float> *ml_depth, MultiLayerImage<uint32_t> *ml_prim_ids) {
  *ml_depth = MultiLayerImage<float>(static_cast<unsigned int>(renderer->height()), static_cast<unsigned int>(renderer->width()), NAN);
  *ml_prim_ids = MultiLayerImage<uint32_t>(static_cast<unsigned int>(renderer->height()), static_cast<unsigned int>(renderer->width()), std::numeric_limits<uint32_t>::max());

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

void GenerateMultiDepthExample(suncg::Scene *scene, const MultiLayerImage<float> &full_ml_depth, const MultiLayerImage<uint32_t> &ml_prim_ids,
                               MultiLayerImage<float> *out_ml_depth, MultiLayerImage<uint16_t> *out_ml_model_indices) {
  Expects(full_ml_depth.width() == ml_prim_ids.width());
  Expects(full_ml_depth.height() == ml_prim_ids.height());

  unsigned int h = full_ml_depth.height();
  unsigned int w = full_ml_depth.width();

  *out_ml_depth = MultiLayerImage<float>(h, w, NAN);
  *out_ml_model_indices = MultiLayerImage<uint16_t>(h, w, std::numeric_limits<uint16_t>::max());

  for (unsigned int y = 0; y < full_ml_depth.height(); y++) {
    for (unsigned int x = 0; x < full_ml_depth.width(); x++) {
      vector<float> *d = full_ml_depth.values(y, x);
      vector<uint32_t> *prim_ids = ml_prim_ids.values(y, x);
      vector<float> *out_d = out_ml_depth->values(y, x);
      vector<uint16_t> *out_m = out_ml_model_indices->values(y, x);

      auto insert_layer = [&out_d, &out_m, &d, &prim_ids, &scene](size_t ind) {
        Ensures(ind < d->size());
        out_d->push_back(d->at(ind));
        out_m->push_back(scene->PrimIdToCategory(prim_ids->at(ind)).index);  // TODO(daeyun): This could be optimized.
      };
      auto insert_nan = [&out_d, &out_m, &d, &prim_ids, &scene]() {
        out_d->push_back(NAN);
        out_m->push_back(0);  // The "Empty" model/category is index 0. According to ModelCategoryMapping.csv
      };

      Expects(d->size() == prim_ids->size());

      // If this pixel is empty (ray did not hit an object or background), all four layers will be NaN.
      if (d->empty()) {
        insert_nan();
        insert_nan();
        insert_nan();
        insert_nan();
      } else {
        // Layer 0. Traditional depth image.
        // ===========================================
        insert_layer(0);

        // Layer 1. Instance exit rule.
        // ===========================================
        // This `Instance` struct contains instance id, model id (from which we can get category id), room id, and the instance type.  See suncg_utils.h
        const suncg::Instance &layer0_instance = scene->PrimIdToInstance(prim_ids->at(0));
        bool is_first_layer_background = scene->IsPrimBackground(prim_ids->at(0));
        if (is_first_layer_background) {
          // If the first layer was a background, second layer is NAN.
          insert_nan();
        } else {
          // Because the first layer was not empty or a background, there must be a second layer. Even if there is only one input layer.
          // Iterate in reverse order and find the first item whose instance label is the same as the first layer.
          size_t instance_exit_index = 0;
          for (size_t i = prim_ids->size() - 1; i > 0; --i) {
            if (layer0_instance.id == scene->PrimIdToInstance(prim_ids->at(i)).id) {
              instance_exit_index = i;
              break;
            }
          }

          insert_layer(instance_exit_index);
        }

        // Layer 2. Second to the last layer.
        // ===========================================
        // This layer represents the empty space in front of the background.
        // Case 1: If Layer 0 was a background or empty space; or if there is no background in this pixel, Layer 2 will be NaN, i.e. ignored at training time.
        // Case 2: If there is only one object between the camera and the background, this layer will be the same as Layer 1.
        // Case 3: If there is another object, the value of this layer will be greater than Layer 1.  i.e. last instance exit.

        // First we need to know if there is a background (if there is, it is the last layer).
        bool is_last_layer_background = scene->IsPrimBackground(prim_ids->at(prim_ids->size() - 1));
        // Now handle Case 1.
        if (is_first_layer_background || !is_last_layer_background) {
          // This means there is no invisible empty space in front of the background. Or it is undefined because there is no background.
          insert_nan();
        } else {
          // There must be at least two layers.
          // Get the second to the last layer. This covers cases 2 and 3.
          insert_layer(prim_ids->size() - 2);
        }

        // Layer 3. Background layer.
        // ===========================================
        // Only disoccluded background.
        if (is_first_layer_background || !is_last_layer_background) { // Same NaN condition as layer 2.
          insert_nan();
        } else {
          // Get the last layer.
          insert_layer(prim_ids->size() - 1);
        }
      }
    }
  }
}

}

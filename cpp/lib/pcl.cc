#include "pcl.h"

namespace scene3d {

unique_ptr<OrthographicCamera> ComputeOverheadCamera(const MultiLayerImage<float> &ml_depth,
                                                     const MultiLayerImage<uint16_t> &ml_model_indices,
                                                     const MultiLayerImage<uint32_t> &ml_prim_ids,
                                                     const PerspectiveCamera &camera,
                                                     suncg::Scene *scene,
                                                     double overhead_hw_ratio,
                                                     AABB *average_bounding_box,
                                                     vector<AABB> *candidate_boxes,
                                                     PerspectiveCamera *aligner) {
  // Assume gravity direction is -Y.
  const double kMinBoxAreaXZ = 0.5;  // Candidates smaller than this will be ignored.

  // Prepare point cloud.
  Image<float> depth_0;
  ml_depth.ExtractLayer(0, &depth_0);
  Image<float> depth_1;
  ml_depth.ExtractLayer(1, &depth_1);
  Image<float> depth_3;
  ml_depth.ExtractLayer(3, &depth_3);

  Points3d depth_0_points;
  PclFromDepthInWorldCoords(depth_0, camera, &depth_0_points);

  Points3d depth_1_points;
  PclFromDepthInWorldCoords(depth_1, camera, &depth_1_points);

  Points3d depth_3_points;
  PclFromDepthInWorldCoords(depth_3, camera, &depth_3_points);

  Points3d depth_3_points_cam;  // TODO: This can be optimized.
  PclFromDepthInCamCoords(depth_3, camera, &depth_3_points_cam);

  Vec3 vd = camera.viewing_direction();
  vd[1] = 0;
  // Aligns the viewing direction with the ground plane.
  *aligner = PerspectiveCamera(camera.position(), camera.position() + vd, Vec3{0, 1, 0}, camera.frustum());

  Points3d depth_0_points_a;
  aligner->WorldToCam(depth_0_points, &depth_0_points_a);
  Points3d depth_1_points_a;
  aligner->WorldToCam(depth_1_points, &depth_1_points_a);
  Points3d depth_3_points_a;
  aligner->WorldToCam(depth_3_points, &depth_3_points_a);
  Points3d depth_01_points_a(3, depth_0_points_a.cols() + depth_1_points_a.cols());
  depth_01_points_a << depth_0_points_a, depth_1_points_a;

  // Heuristic 1: Depth statistics.
  candidate_boxes->push_back([&]() -> AABB {
    const double kBoundingBoxStd = 1.5;
    const double kMaxDistanceFromCamera = 9;
    const double kMinRadius = 1;

    if (depth_01_points_a.cols() < 40) {
      return AABB();  // This heuristic is not used if there are too few valid points.
    }

    Vec3 mean;
    Vec3 stddev;
    MeanAndStd(depth_01_points_a, &mean, &stddev);

    // Make sure the radius is not too small.
    Vec3 radius = stddev * kBoundingBoxStd;
    if (radius.norm() < kMinRadius) {
      radius.normalize();
      radius *= kMinRadius;
    }

    // Make sure the mean is not too far from the camera.
    if (mean.norm() > kMaxDistanceFromCamera) {
      mean.normalize();
      mean *= kMaxDistanceFromCamera;
    }

    return AABB(mean - radius, mean + radius);
  }());

  // Heuristic 2: Background and frustum.
  candidate_boxes->push_back([&]() -> AABB {
    const double kMaxDistanceFromCamera = 7.0;
    const double kMinDistanceFromCamera = 1.0;
    const double kCentroidWeight = 0.8;  // vs. camera position.
    const double kFrustumZoom = 0.9;

    double mean_depth = 0;
    if (depth_3_points_a.cols() < 10) {
      // This heuristic is used even if there are too few background points.
      // Just need to set a constant mean depth value.
      mean_depth = 0.5 * (kMaxDistanceFromCamera + kMinDistanceFromCamera);
    } else {
      mean_depth = std::abs(depth_3_points_cam.row(2).mean());  // In original camera coordinates.
      mean_depth *= kCentroidWeight;
      mean_depth = std::max(std::min(mean_depth, kMaxDistanceFromCamera), kMinDistanceFromCamera);
    }

    Ensures(mean_depth > 0);

    Vec3 principal_background_center{0, 0, -mean_depth};
    Vec3 background_center_aligned;
    {
      Vec3 w;
      camera.CamToWorld(principal_background_center, &w);
      aligner->WorldToCam(w, &background_center_aligned);
    }

    double x_fov;
    camera.fov(&x_fov, nullptr);
    double size = std::tan(x_fov) * mean_depth * kFrustumZoom;

    // Square bias.
    return AABB(background_center_aligned.array() - size, background_center_aligned.array() + size);
  }());
  Ensures(candidate_boxes->at(candidate_boxes->size() - 1).XZArea() > kMinBoxAreaXZ);  // This bounding box is always used.

  // Heuristic 3: Objects-only. Include all objects.
  candidate_boxes->push_back([&]() -> AABB {
    const double kMinDistance = 1.5;
    const double kMaxDistance = 9;

    if (depth_01_points_a.cols() < 40) {
      return AABB();  // This heuristic is not used if there are too few valid points.
    }

    // Extract prim ids for points in depth 0 and 1.
    vector<uint32_t> valid_prim_ids;
    for (unsigned int y = 0; y < depth_0.height(); ++y) {
      for (unsigned int x = 0; x < depth_0.width(); ++x) {
        float value = depth_0.at(y, x);
        if (std::isfinite(value)) {
          auto pid = ml_prim_ids.at(y, x, 0);
          Ensures(pid != std::numeric_limits<uint32_t>::max());
          valid_prim_ids.push_back(pid);
        }
      }
    }
    for (unsigned int y = 0; y < depth_1.height(); ++y) {
      for (unsigned int x = 0; x < depth_1.width(); ++x) {
        float value = depth_1.at(y, x);
        if (std::isfinite(value)) {
          auto pid = ml_prim_ids.at(y, x, 1);
          Ensures(pid != std::numeric_limits<uint32_t>::max());
          valid_prim_ids.push_back(pid);
        }
      }
    }
    Expects(valid_prim_ids.size() == depth_01_points_a.cols());

    // Find the current room. Pixels vote on which room they belong to. Majority wins.
    map<string, int> room_counts;
    int pixel_of_interest_count = 0;
    for (int i = 0; i < valid_prim_ids.size(); ++i) {
      if (scene->IsPrimBackground(valid_prim_ids[i])) {
        continue;
      }
      const auto &instance = scene->PrimIdToInstance(valid_prim_ids[i]);
      if (!instance.room_id.empty()) {
        room_counts[instance.room_id]++;
        pixel_of_interest_count++;
      }
    }

    if (pixel_of_interest_count < 20) {
      return AABB();  // This heuristic is not used if there are too few points that belong to an object.
    }

    // Find room id.
    int highest_count = 0;
    string room_id;
    for (const auto &rc : room_counts) {
      if (rc.second > highest_count) {
        highest_count = rc.second;
        room_id = rc.first;
      }
    }
    Expects(highest_count > 0);

    AABB box;
    int final_included_point_count = 0;
    for (int i = 0; i < valid_prim_ids.size(); ++i) {
      double distance_from_camera = depth_01_points_a.col(i).norm();
      if (distance_from_camera > kMaxDistance || distance_from_camera < kMinDistance) {
        continue;  // Points are too far or too close to the camera are ignored.
      }

      if (scene->IsPrimBackground(valid_prim_ids[i])) {
        continue;  // Ignore background.
      }

      const auto &category = scene->PrimIdToCategory(valid_prim_ids[i]);
      const auto &instance = scene->PrimIdToInstance(valid_prim_ids[i]);
      if (room_id != instance.room_id) {
        continue;  // Objects not in the same room are ignored.
      }

      // In addition to the room envelope, also ignore those categories. Those cases are very rare and only matter if they are far away from other objects.
      if (category.nyuv2_40class == "void" ||
          category.nyuv2_40class == "person" ||
          category.fine_grained_class == "plant" ||
          category.fine_grained_class == "chandelier" ||
          category.fine_grained_class == "decoration" ||
          category.fine_grained_class == "surveillance_camera") {
        continue;
      }

      // Now we know this point belongs to an object of interest.

      box.Grow(depth_01_points_a.col(i));
      ++final_included_point_count;
    }

    if (final_included_point_count < 15) {
      return AABB();
    }

    box.Expand(1.05);  // Grow 5% to avoid having objects too close to the edge.
    return box;
  }());

  LOGGER->info("box area: {:.1f}, {:.1f}, {:.1f}", candidate_boxes->at(0).XZArea(), candidate_boxes->at(1).XZArea(), candidate_boxes->at(2).XZArea());

  Vec box_weights(3);
  box_weights[0] = 1;
  box_weights[1] = 1;
  box_weights[2] = 2;

  for (size_t i = 0; i < candidate_boxes->size(); ++i) {
    if (candidate_boxes->at(i).XZArea() < kMinBoxAreaXZ) {
      box_weights[i] = 0;
    }
  }

  Vec3 combined_bmin = (box_weights[0] * candidate_boxes->at(0).bmin + box_weights[1] * candidate_boxes->at(1).bmin + box_weights[2] * candidate_boxes->at(2).bmin) / box_weights.sum();
  Vec3 combined_bmax = (box_weights[0] * candidate_boxes->at(0).bmax + box_weights[1] * candidate_boxes->at(1).bmax + box_weights[2] * candidate_boxes->at(2).bmax) / box_weights.sum();

  // Determine the the height of the overhead camera will be.
  combined_bmax[1] = depth_01_points_a.row(1).maxCoeff() - 0.05;

  // 0 in world coordinates. Not really important.
  combined_bmin[1] = -aligner->position()[1];

  *average_bounding_box = AABB(combined_bmin, combined_bmax);

  Vec3 overhead_campos_cam = average_bounding_box->Center();
  overhead_campos_cam[1] = combined_bmax[1];
  Vec3 overhead_campos;
  aligner->CamToWorld(overhead_campos_cam, &overhead_campos);

  FrustumParams overhead_frustum;
  overhead_frustum.right = 0.5 * (average_bounding_box->bmax[0] - average_bounding_box->bmin[0]);
  overhead_frustum.left = -overhead_frustum.right;
  overhead_frustum.top = 0.5 * (average_bounding_box->bmax[2] - average_bounding_box->bmin[2]);
  overhead_frustum.bottom = -overhead_frustum.top;
  overhead_frustum.near = 0.005;
  overhead_frustum.far = 50;
  overhead_frustum = ForceFixedAspectRatio(overhead_hw_ratio, overhead_frustum);

  auto ret = make_unique<OrthographicCamera>(overhead_campos, overhead_campos + Vec3{0, -1, 0}, aligner->viewing_direction(), overhead_frustum);

#if 0  // Enable for debugging and visualization.
  {
    PointCloud pcl(depth_01_points_a);
    pcl.Save("/tmp/scene3d/pcl.bin");
    auto save_bounding_box = [&](string filename, AABB bb) {
      Points3d p(3, 2);
      p << bb.bmin, bb.bmax;
      PointCloud bbpcl(p);
      bbpcl.Save(filename);
    };
    auto save_cam = [&](string filename, const OrthographicCamera &cam) {
      Points3d p(3, 4);
      p << cam.position(), cam.position() + cam.viewing_direction(), cam.position() + 0.5 * cam.viewing_direction(), cam.position() + cam.up();
      Points3d p2;
      aligner->WorldToCam(p, &p2);
      Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> transposed_points = p2.transpose().cast<float>().eval();
      SerializeTensor<float>(filename, transposed_points.data(), {static_cast<int>(4), 3});
    };
    save_bounding_box("/tmp/scene3d/pcl_b0.bin", *average_bounding_box);
    save_bounding_box("/tmp/scene3d/pcl_b1.bin", candidate_boxes->at(0));
    save_bounding_box("/tmp/scene3d/pcl_b2.bin", candidate_boxes->at(1));
    save_bounding_box("/tmp/scene3d/pcl_b3.bin", candidate_boxes->at(2));
    save_cam("/tmp/scene3d/pcl_o.bin", *ret);

  }
#endif

  return move(ret);
}

void PclFromDepthInWorldCoords(const Image<float> &depth, const Camera &camera, PointCloud *out) {
  Points3d pts;
  PclFromDepthInWorldCoords(depth, camera, &pts);
  *out = PointCloud(pts);
}

void PclFromDepthInWorldCoords(const Image<float> &depth, const Camera &camera, Points3d *out) {
  Points3d cam_out;
  PclFromDepthInCamCoords(depth, camera, &cam_out);
  camera.CamToWorld(cam_out, out);
}

void PclFromDepthInWorldCoords(const Image<float> &depth, const Camera &camera, Points2i *xy, Points3d *out) {
  Points3d cam_out;
  PclFromDepthInCamCoords(depth, camera, xy, &cam_out);
  camera.CamToWorld(cam_out, out);
}

void PclFromDepthInCamCoords(const Image<float> &depth, const Camera &camera, Points3d *out) {
  Points2i xy;
  Points1d cam_d;
  ValidPixelCoordinates(depth, &xy, &cam_d);

  // Calculate coordinates in camera space.
  camera.ImageToCam(xy, cam_d, depth.height(), depth.width(), out);
}

void PclFromDepthInCamCoords(const Image<float> &depth, const Camera &camera, Points2i *xy, Points3d *out) {
  Points1d cam_d;
  ValidPixelCoordinates(depth, xy, &cam_d);

  // Calculate coordinates in camera space.
  camera.ImageToCam(*xy, cam_d, depth.height(), depth.width(), out);
}

void ValidPixelCoordinates(const Image<float> &depth, Points2i *out_xy, Points1d *out_values) {
  vector<Vec2i> xy;
  vector<double> d;
  for (unsigned int y = 0; y < depth.height(); ++y) {
    for (unsigned int x = 0; x < depth.width(); ++x) {
      float value = depth.at(y, x);
      if (std::isfinite(value)) {
        xy.emplace_back(x, y);
        d.push_back(value);
      }
    }
  }

  out_xy->resize(2, xy.size());
  out_values->resize(1, xy.size());
  for (int i = 0; i < xy.size(); ++i) {
    out_xy->col(i) = xy[i];
    (*out_values)[i] = d[i];
  }
}

void MeanAndStd(const Points3d &points, Vec3 *mean, Vec3 *stddev) {
  *mean = points.rowwise().mean();
  *stddev = ((points.colwise() - *mean).array().square().rowwise().sum().array() / points.cols()).sqrt();
}

void SaveAABB(const string &txt_filename, const vector<AABB> &boxes) {
  int precision = 12;
  std::ofstream ofile;
  ofile.open(txt_filename, std::ios::out);

  for (int i = 0; i < boxes.size(); ++i) {
    ofile <<
          std::setprecision(precision) <<
          boxes[i].bmin[0] << " " <<
          boxes[i].bmin[1] << " " <<
          boxes[i].bmin[2] << " " <<
          boxes[i].bmax[0] << " " <<
          boxes[i].bmax[1] << " " <<
          boxes[i].bmax[2] << std::endl;

  }

  ofile.close();
}

// Taken from MVE's depthmap.cc
// BSD 3-Clause License
// https://github.com/simonfuhrmann/mve/blob/master/LICENSE.txt
bool dm_is_depthdisc(float *widths, float *depths, float dd_factor, int i1, int i2) {
  // Find index that corresponds to smaller depth.
  int i_min = i1;
  int i_max = i2;
  if (depths[i2] < depths[i1]) {
    std::swap(i_min, i_max);
  }

  // Check if indices are a diagonal.
  if (i1 + i2 == 3) {
    dd_factor *= M_SQRT2;
  }

  // Check for depth discontinuity.
  return depths[i_max] - depths[i_min] > widths[i_min] * dd_factor;
}

void TriangulateDepth(const Image<float> &depth, const Camera &camera, float dd_factor, vector<array<unsigned int, 3>> *faces, vector<array<float, 3>> *vertices) {
  const unsigned int width = depth.width();
  const unsigned int height = depth.height();

  Points2i xy;
  Points3d pcl;
  PclFromDepthInWorldCoords(depth, camera, &xy, &pcl);

  Ensures(xy.cols() == pcl.cols());

  for (int k = 0; k < pcl.cols(); ++k) {
    vertices->push_back(array<float, 3>{
        static_cast<float>(pcl(0, k)),
        static_cast<float>(pcl(1, k)),
        static_cast<float>(pcl(2, k)),
    });
  }

  Image<int> xy_to_vertex_index(height, width, -1);
  for (int i = 0; i < xy.cols(); ++i) {
    xy_to_vertex_index.at(xy(1, i), xy(0, i)) = static_cast<int>(i);
  }

  auto xy_to_depth = [&](int x, int y) -> float {
    float value = depth.at(static_cast<unsigned int>(y), static_cast<unsigned int>(x));
    if (std::isfinite(value)) {
      return value;
    }
    return 0;  // 0 means invalid.
  };

  auto make_triangle = [&](int x, int y, int i1, int i2, int i3) {
    int v1 = xy_to_vertex_index.at(static_cast<unsigned int>(y + (i1 / 2)), static_cast<unsigned int>(x + (i1 % 2)));
    int v2 = xy_to_vertex_index.at(static_cast<unsigned int>(y + (i2 / 2)), static_cast<unsigned int>(x + (i2 % 2)));
    int v3 = xy_to_vertex_index.at(static_cast<unsigned int>(y + (i3 / 2)), static_cast<unsigned int>(x + (i3 % 2)));
    Expects(v1 >= 0);
    Expects(v2 >= 0);
    Expects(v3 >= 0);

    faces->push_back(array<unsigned int, 3>{static_cast<unsigned int>(v1), static_cast<unsigned int>(v2), static_cast<unsigned int>(v3)});
  };

  for (int y = 0; y < height - 1; ++y) {
    for (int x = 0; x < width - 1; ++x) {
      float depths[4]{
          xy_to_depth(x, y),
          xy_to_depth(x + 1, y),
          xy_to_depth(x, y + 1),
          xy_to_depth(x + 1, y + 1),
      };

      // Based on MVE's `depthmap_triangulate` function.  See their depthmap.cc
      // https://github.com/simonfuhrmann/mve/blob/master/LICENSE.txt
      int mask = 0;
      int num_valid_pixels = 0;
      for (int j = 0; j < 4; ++j) {
        if (depths[j] > 0.0f) {
          mask |= 1 << j;
          num_valid_pixels += 1;
        }
      }

      // At least three valid depth values are required.
      if (num_valid_pixels < 3) {
        continue;
      }

      // Possible triangles, vertex indices relative to 2x2 block.
      int tris[4][3] = {
          {0, 2, 1}, {0, 3, 1}, {0, 2, 3}, {1, 2, 3}
      };

      // Decide which triangles to issue. 1-indexed. 0 means unused.
      int tri[2] = {0, 0};

      switch (mask) {
        case 7: tri[0] = 1;
          break;
        case 11: tri[0] = 2;
          break;
        case 13: tri[0] = 3;
          break;
        case 14: tri[0] = 4;
          break;
        case 15: {
          // Choose the triangulation with smaller diagonal.
          float ddiff1 = std::abs(depths[0] - depths[3]);
          float ddiff2 = std::abs(depths[1] - depths[2]);
          if (ddiff1 < ddiff2) {
            tri[0] = 2;
            tri[1] = 3;
          } else {
            tri[0] = 1;
            tri[1] = 4;
          }
          break;
        }
        default: continue;
      }

      // Omit depth discontinuity detection if dd_factor is zero.
      if (dd_factor > 0.0f) {
        // Cache pixel footprints.
        float widths[4];
        for (int j = 0; j < 4; ++j) {
          if (depths[j] == 0.0f) {
            continue;
          }
          Points2i v_xy(2, 1);
          v_xy << x + (j % 2), y + (j / 2);
          Points1d d(1, 1);
          d << depths[j];

          Points1d footprint;
          camera.PixelFootprintX(v_xy, d, height, width, &footprint);
          Ensures(footprint.size() == 1);
          widths[j] = footprint[0];
        }

        // Check for depth discontinuities.
        for (int j = 0; j < 2 && tri[j] != 0; ++j) {
          int *tv = tris[tri[j] - 1];
          if (dm_is_depthdisc(widths, depths, dd_factor, tv[0], tv[1])) tri[j] = 0;
          if (dm_is_depthdisc(widths, depths, dd_factor, tv[1], tv[2])) tri[j] = 0;
          if (dm_is_depthdisc(widths, depths, dd_factor, tv[2], tv[0])) tri[j] = 0;
        }
      }

      // Build triangles.
      for (int j = 0; j < 2; ++j) {
        if (tri[j] == 0) {
          continue;
        }
        make_triangle(x, y, tris[tri[j] - 1][0], tris[tri[j] - 1][1], tris[tri[j] - 1][2]);
      }
    }
  }

}

}

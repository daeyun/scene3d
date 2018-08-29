//
// Created by daeyun on 8/28/18.
//

#pragma once

#include "common.h"

namespace scene3d {
void ToEigen(const std::vector<std::array<float, 3>> &values, Points3d *out) {
  out->resize(3, values.size());
  for (int i = 0; i < values.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      out->col(i)[j] = static_cast<double>(values[i][j]);
    }
  }
  // NOTE: Eigen is column major by default
}
}


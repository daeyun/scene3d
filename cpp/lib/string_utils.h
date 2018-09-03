#include <string>
#include <sstream>

#include "lib/common.h"

#pragma once

namespace scene3d {

inline bool EndsWith(std::string const &full_string, std::string const &ending) {
  // https://stackoverflow.com/a/874160
  if (full_string.length() >= ending.length()) {
    return (0 == full_string.compare(full_string.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

inline bool StartsWith(std::string const &full_string, std::string const &prefix) {
  if (full_string.length() >= prefix.length()) {
    return 0 == full_string.compare(0, prefix.length(), prefix);
  } else {
    return false;
  }
}

std::string toString(const Eigen::MatrixXd &mat) {
  std::stringstream ss;
  ss << mat;
  return ss.str();
}
}

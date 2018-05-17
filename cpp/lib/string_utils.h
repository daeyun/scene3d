#include "lib/common.h"

#pragma once

bool EndsWith(std::string const &fullString, std::string const &ending) {
  // https://stackoverflow.com/a/874160
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

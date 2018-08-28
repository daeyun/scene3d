#pragma once

#include "common.h"
#include "camera.h"

namespace scene3d {

class DepthImage {
 public:
  DepthImage(unsigned int height, unsigned int width) : height_(height), width_(width) {
    data_.resize(height_ * width_);
  }

  inline float &at(unsigned int index) {
    return data_[index];
  }

  inline float &at(unsigned int y, unsigned int x) {
    return data_[width_ * y + x];
  }

  inline float at(unsigned int index) const {
    return data_[index];
  }

  inline float at(unsigned int y, unsigned int x) const {
    return data_[width_ * y + x];
  }

  const float *data() const {
    return data_.data();
  }

  unsigned int size() const {
    return static_cast<unsigned int>(data_.size());
  }

  unsigned int height() const {
    return height_;
  }

  unsigned int width() const {
    return width_;
  }

 private:
  vector<float> data_;
  unsigned int height_, width_;
};

class MultiLayerDepthImage {
 public:
  MultiLayerDepthImage(unsigned int height, unsigned int width) : height_(height), width_(width) {
    data_.resize(height_ * width_);
    for (int i = 0; i < data_.size(); ++i) {
      data_[i] = std::make_unique<vector<float>>();
    }
  }

  inline vector<float> *values(unsigned int index) const {
    return data_[index].get();
  }

  inline vector<float> *values(unsigned int y, unsigned int x) const {
    return data_[width_ * y + x].get();
  }

  inline float at(unsigned int y, unsigned int x, unsigned int l) const {
    const auto *v = values(y, x);
    if (l < v->size()) {
      return v->at(l);
    }
    return NAN;
  }

  inline float at(unsigned int index, unsigned int l) const {
    const auto *v = values(index);
    if (l < v->size()) {
      return v->at(l);
    }
    return NAN;
  }

  void ExtractLayer(unsigned int l, DepthImage *out) {
    *out = DepthImage(height_, width_);
    for (unsigned int i = 0; i < data_.size(); ++i) {
      out->at(i) = this->at(i, l);
    }
  }

  unsigned int NumLayers() const {
    unsigned int ret = 0;
    for (const auto &item_ptr : data_) {
      if (ret < item_ptr->size()) {
        ret = static_cast<unsigned int>(item_ptr->size());
      }
    }
    return ret;
  }

 private:
  vector<unique_ptr<vector<float>>> data_;
  unsigned int height_, width_;
};

}

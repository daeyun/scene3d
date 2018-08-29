#pragma once

#include "lib/common.h"
#include "lib/camera.h"

namespace scene3d {

template<typename T=float>
class Image {
 public:
  Image(unsigned int height, unsigned int width) : height_(height), width_(width) {
    data_.resize(height_ * width_);
  }

  inline T &at(unsigned int index) {
    return data_[index];
  }

  inline T &at(unsigned int y, unsigned int x) {
    return data_[width_ * y + x];
  }

  inline T at(unsigned int index) const {
    return data_[index];
  }

  inline T at(unsigned int y, unsigned int x) const {
    return data_[width_ * y + x];
  }

  const T *data() const {
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
  vector<T> data_;
  unsigned int height_, width_;
};

template<typename T=float>
class MultiLayerImage {
 public:
  MultiLayerImage(unsigned int height, unsigned int width) : height_(height), width_(width) {
    data_.resize(height_ * width_);
    for (int i = 0; i < data_.size(); ++i) {
      data_[i] = std::make_unique<vector<T>>();
    }
  }

  inline vector<T> *values(unsigned int index) const {
    return data_[index].get();
  }

  inline vector<T> *values(unsigned int y, unsigned int x) const {
    return data_[width_ * y + x].get();
  }

  inline T at(unsigned int y, unsigned int x, unsigned int l) const {
    const auto *v = values(y, x);
    if (l < v->size()) {
      return v->at(l);
    }
    return NAN;
  }

  inline T at(unsigned int index, unsigned int l) const {
    const auto *v = values(index);
    if (l < v->size()) {
      return v->at(l);
    }
    return NAN;
  }

  void ExtractLayer(unsigned int l, Image<T> *out) {
    *out = Image<T>(height_, width_);
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
  vector<unique_ptr<vector<T>>> data_;
  unsigned int height_, width_;
};

}

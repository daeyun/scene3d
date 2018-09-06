#pragma once

#include "lib/common.h"
#include "lib/camera.h"
#include "lib/file_io.h"

namespace scene3d {

template<typename T=float>
class Image {
 public:
  Image(unsigned int height, unsigned int width, T null_value) : height_(height), width_(width), null_value_(null_value) {
    data_.resize(height_ * width_);
  }

  void Save(const string &filename) const {
    SerializeTensor<T>(filename, this->data(), {height_, width_});
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
  T null_value_;
};

template<typename T=float>
class MultiLayerImage {
 public:
  MultiLayerImage() : height_(0), width_(0) {}

  // `null_value` is usually NAN if T is float.
  MultiLayerImage(unsigned int height, unsigned int width, T null_value) : height_(height), width_(width), null_value_(null_value) {
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
    return null_value_;
  }

  inline T at(unsigned int index, unsigned int l) const {
    const auto *v = values(index);
    if (l < v->size()) {
      return v->at(l);
    }
    return null_value_;
  }

  void ExtractLayer(unsigned int l, Image<T> *out) const {
    *out = Image<T>(height_, width_, null_value_);
    for (unsigned int i = 0; i < data_.size(); ++i) {
      out->at(i) = this->at(i, l);
    }
  }

  // Appends to `out` without initializing.
  void ExtractLayer(unsigned int l, vector<T> *out) const {
    for (unsigned int i = 0; i < data_.size(); ++i) {
      out->push_back(this->at(i, l));
    }
  }

  // Get the values of first N layers.
  void ExtractContiguousLayers(unsigned int num_layers, vector<T> *out) const {
    for (int i = 0; i < num_layers; ++i) {
      ExtractLayer(i, out);
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

  void Save(const string &filename, unsigned int num_layers) const {
    // After compression, (N, H, W) ends up being smaller than (H, W, N).
    // 344 Kb vs. 468 Kb on House 0004d5 LDI.

    vector<T> flat;
    ExtractContiguousLayers(num_layers, &flat);
    Ensures(flat.size() == height_ * width_ * num_layers);
    SerializeTensor<T>(filename, flat.data(), {num_layers, height_, width_});
  }

  unsigned int height() const {
    return height_;
  }
  unsigned int width() const {
    return width_;
  }

 private:
  vector<unique_ptr<vector<T>>> data_;
  unsigned int height_, width_;
  T null_value_;
};

}

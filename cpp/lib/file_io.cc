//
// Created by daeyun on 12/18/17.
//

#include "file_io.h"

#include <fstream>
#include <chrono>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <set>
#include <iomanip>
#include <sys/stat.h>

#include "spdlog/spdlog.h"

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "assimp/DefaultLogger.hpp"
#include "assimp/LogStream.hpp"
#include "boost/filesystem.hpp"
#include "blosc.h"

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include "lib/common.h"
#include "lib/string_utils.h"
#include "lib/benchmark.h"

namespace scene3d {
namespace fs = boost::filesystem;

bool ReadTriangles(const std::string &filename,
                   const std::function<void(const std::array<std::array<float, 3>, 3> &)> &triangle_handler) {
  LOGGER->info("Importing {}", filename);

  if (!boost::filesystem::exists(filename)) {
    LOGGER->error("{} does not exist", filename);
    throw std::runtime_error("file not found");
  }

  Assimp::Importer importer;

  // List of post-processing flags can be found here:
  // http://sir-kimmi.de/assimp/lib_html/postprocess_8h.html#a64795260b95f5a4b3f3dc1be4f52e410
  importer.SetPropertyInteger(AI_CONFIG_PP_FD_REMOVE, 1);  // Remove degenerate triangles.

  double start_time = scene3d::TimeSinceEpoch<std::milli>();
  const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
  LOGGER->debug("Elapsed (importer.ReadFile): {} ms", scene3d::TimeSinceEpoch<std::milli>() - start_time);

  if (!scene) {
    LOGGER->error("ERROR in {}: {}", filename, importer.GetErrorString());
    return false;
  }

  // TODO: there seems to be a problem in reading binary ply files. this is a hack to detect parsing error.
  const float kMaxVertexValue = 1e7;

  int triangle_count = 0;
  for (int i = 0; i < scene->mNumMeshes; ++i) {
    const aiMesh *mesh = scene->mMeshes[i];
    for (int j = 0; j < mesh->mNumFaces; ++j) {
      auto face = mesh->mFaces[j];
      Expects(face.mNumIndices == 3);

      for (int k = 0; k < 3; ++k) {
        if (face.mIndices[k] >= mesh->mNumVertices) {
          LOGGER->warn("Invalid vertex index found. Skipping.");
          continue;
        }
      }

      auto a = mesh->mVertices[face.mIndices[0]];
      auto b = mesh->mVertices[face.mIndices[1]];
      auto c = mesh->mVertices[face.mIndices[2]];

      Vec3 a_{a.x, a.y, a.z};
      Vec3 b_{b.x, b.y, b.z};
      Vec3 c_{c.x, c.y, c.z};
      double area = (b_ - a_).cross(c_ - a_).norm() * 0.5f;
      if (area < 1e-9) {
        LOGGER->warn("Triangle area too small: {:.20g}. Removed", area);
        continue;
      }

      if (std::abs(a.x) > kMaxVertexValue || std::abs(a.y) > kMaxVertexValue || std::abs(a.z) > kMaxVertexValue) {
        LOGGER->error("vertex value above threshold: {}, {}, {}", a.x, a.y, a.z);
        throw std::runtime_error("");
      }
      triangle_handler({std::array<float, 3>{a.x, a.y, a.z},
                        std::array<float, 3>{b.x, b.y, b.z},
                        std::array<float, 3>{c.x, c.y, c.z}});
      ++triangle_count;
    }
  }

  if (triangle_count <= 0) {
    LOGGER->error("No triangles found in mesh file.");
  }

  return true;
}

bool ReadFacesAndVertices(const std::string &filename,
                          std::vector<std::array<unsigned int, 3>> *faces,
                          std::vector<std::array<float, 3>> *vertices,
                          std::vector<int> *node_id_for_each_face,
                          std::vector<std::string> *node_name_for_each_face) {
  LOGGER->info("Importing {}", filename);
  if (!boost::filesystem::exists(filename)) {
    LOGGER->error("{} does not exist", filename);
    throw std::runtime_error("file not found");
  }

  Assimp::Importer importer;

  // List of post-processing flags can be found here:
  // http://sir-kimmi.de/assimp/lib_html/postprocess_8h.html#a64795260b95f5a4b3f3dc1be4f52e410
  const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);

  if (!scene) {
    LOGGER->error("ERROR in {}: {}", filename, importer.GetErrorString());
    return false;
  }

  // TODO: there seems to be a problem reading binary ply files. this is a hack to detect parsing error.
  const float kMaxVertexValue = 1e7;

  int triangle_count = 0;
  int face_offset = 0;

  // Recursively collect nodes in DFS order.
  std::vector<aiNode *> nodes;
  std::function<void(aiNode *)> recursive_node_collector;
  recursive_node_collector = [&](aiNode *root) {
    nodes.push_back(root);
    for (int j = 0; j < root->mNumChildren; ++j) {
      recursive_node_collector(root->mChildren[j]);
    }
  };
  recursive_node_collector(scene->mRootNode);

  for (int node_index = 0; node_index < nodes.size(); ++node_index) {
    aiNode *node = nodes[node_index];
    for (int i = 0; i < node->mNumMeshes; ++i) {
      const int mesh_index = node->mMeshes[i];
      const aiMesh *mesh = scene->mMeshes[mesh_index];
      std::string meshname(mesh->mName.data);

      for (int j = 0; j < mesh->mNumVertices; ++j) {
        auto vertex = mesh->mVertices[j];
        if (std::abs(vertex.x) > kMaxVertexValue || std::abs(vertex.y) > kMaxVertexValue
            || std::abs(vertex.z) > kMaxVertexValue) {
          LOGGER->error("vertex value above threshold: {}, {}, {}", vertex.x, vertex.y, vertex.z);
          throw std::runtime_error("");
        }
        vertices->push_back({vertex.x, vertex.y, vertex.z});
      }
      for (int j = 0; j < mesh->mNumFaces; ++j) {
        auto face = mesh->mFaces[j];
        Expects(face.mNumIndices == 3);
        for (int k = 0; k < 3; ++k) {
          if (face.mIndices[k] >= mesh->mNumVertices) {
            LOGGER->warn("Invalid vertex index found. Skipping.");
            continue;
          }
        }

        const auto a = mesh->mVertices[face.mIndices[0]];
        const auto b = mesh->mVertices[face.mIndices[1]];
        const auto c = mesh->mVertices[face.mIndices[2]];

        Vec3 a_{a.x, a.y, a.z};
        Vec3 b_{b.x, b.y, b.z};
        Vec3 c_{c.x, c.y, c.z};
        double area = (b_ - a_).cross(c_ - a_).norm() * 0.5f;
        if (area < 1e-9) {
          LOGGER->warn("Triangle area too small: {:.20g}. Removed", area);
          continue;
        }

        faces->push_back({static_cast<unsigned int>(face_offset + face.mIndices[0]),
                          static_cast<unsigned int>(face_offset + face.mIndices[1]),
                          static_cast<unsigned int>(face_offset + face.mIndices[2])});

        if (node_id_for_each_face != nullptr) {
          node_id_for_each_face->push_back(node_index);
        }
        if (node_name_for_each_face != nullptr) {
          node_name_for_each_face->push_back(std::string(node->mName.data));
        }

        ++triangle_count;
      }
      face_offset += mesh->mNumVertices;
      Ensures(face_offset == vertices->size());
    }
  }

  for (int i = 0; i < scene->mNumMeshes; ++i) {
  }

  if (triangle_count <= 0) {
    LOGGER->error("No triangles found in mesh file.");
  }

  return true;
}

vector<array<array<float, 3>, 3>>
ReadTriangles(const std::string &filename) {
  vector<array<array<float, 3>, 3 >> triangles;
  ReadTriangles(filename,
                [&](const array<array<float, 3>, 3> triangle) {
                  triangles.push_back(triangle);
                });
  return triangles;
}

// Dumps raw bytes to a stream.
void WriteBytes(const void *src, size_t size_bytes, std::ostream *stream) {
  stream->write(reinterpret_cast<const char *>(src), size_bytes);
}

void WriteBytes(const std::string &src, std::ostream *stream) {
  WriteBytes(src.data(), src.size(), stream);
}

template<typename T>
void WriteBytes(const vector<T> &src, std::ostream *stream) {
  WriteBytes(src.data(), sizeof(T) * src.size(), stream);
}

template<typename T>
void WriteBytes(const T &src, std::ostream *stream) {
  static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
  WriteBytes(&src, sizeof(src), stream);
}

template void WriteBytes<uint8_t>(const vector<uint8_t> &src, std::ostream *stream);
template void WriteBytes<int32_t>(const vector<int32_t> &src, std::ostream *stream);
template void WriteBytes<float>(const vector<float> &src, std::ostream *stream);

void CompressBytes(const void *src, int size_bytes, const char *compressor, int level, size_t typesize,
                   std::string *out) {
  out->resize(std::max(static_cast<size_t>(size_bytes * 4 + BLOSC_MAX_OVERHEAD), static_cast<size_t>(512)));
  int compressed_size = blosc_compress_ctx(level, true, typesize, static_cast<size_t>(size_bytes),
                                           src, &(*out)[0], out->size(), compressor, 0, 1);
  out->resize(static_cast<size_t>(compressed_size));
  if (compressed_size <= 0) {
    LOGGER->error("Compression failed.");
    throw std::runtime_error("Compression failed.");
  }
}

void DecompressBytes(const void *src, std::string *out) {
  size_t nbytes, cbytes, blocksize;
  blosc_cbuffer_sizes(src, &nbytes, &cbytes, &blocksize);
  out->resize(nbytes);
  int decompressed_size = blosc_decompress_ctx(src, &(*out)[0], out->size(), 1);
  if (decompressed_size <= 0) {
    LOGGER->error("Decompression failed.");
    throw std::runtime_error("Decompression failed.");
  }
}

std::string ReadBytes(const std::string &path) {
  auto canonical_path = boost::filesystem::canonical(path).string();
  std::ifstream stream(canonical_path);
  std::string content((std::istreambuf_iterator<char>(stream)),
                      (std::istreambuf_iterator<char>()));
  return content;
}

template<typename T>
void ReadTensorData(const string &filename, vector<int> *shape, vector<T> *data) {
  const string compressed = ReadBytes(filename);
  string serialized;
  DecompressBytes(compressed.data(), &serialized);

  const auto *header = reinterpret_cast<const int32_t *>(serialized.data());
  const int32_t dims = *header;

  for (int i = 1; i <= dims; ++i) {
    shape->push_back(*(header + i));
  }

  size_t size = serialized.size() - sizeof(int32_t) * (dims + 1);

  size_t shape_prod = static_cast<size_t>(
      std::accumulate(shape->begin(), shape->end(), 1, std::multiplies<>()));
  size_t num_elements = size / sizeof(T);

  Ensures(shape_prod == num_elements);

  const auto *data_start = reinterpret_cast<const T *>(header + dims + 1);
  data->reserve(num_elements);
  data->assign(data_start, data_start + num_elements);
}

template void ReadTensorData<float>(const string &filename, vector<int> *shape, vector<float> *data);
template void ReadTensorData<int>(const string &filename, vector<int> *shape, vector<int> *data);

bool Exists(const std::string &filename) {
  if (filename.empty()) {
    return false;
  }

  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0);
}

string SystemTempDir() {
  return fs::temp_directory_path().string();
}

bool PrepareDir(const string &dirname) {
  auto path = fs::absolute(dirname);
  if (!fs::is_directory(path)) {
    Expects(!fs::is_regular_file(path));
    bool ok = fs::create_directories(path);
    if (ok) {
      LOGGER->debug("mkdir -p {}", path.string());
      return true;
    }
  }
  return false;
}

bool PrepareDirForFile(const string &filename) {
  auto parent_path = fs::absolute(filename).parent_path();
  return PrepareDir(parent_path.string());
}

template<typename T>
void SerializeTensor(const std::string &filename, const void *data, const std::vector<int> &shape) {
  const int num_bytes = sizeof(T) * std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());

  Expects(num_bytes > sizeof(T));

  std::ostringstream stream;
  WriteBytes<int32_t>(shape.size(), &stream);
  WriteBytes<int32_t>(shape, &stream);
  WriteBytes(data, num_bytes, &stream);

  string encoded = stream.str();

  std::string compressed;
  CompressBytes(encoded.data(), static_cast<int>(encoded.size()), "lz4hc", 9, sizeof(T), &compressed);
  void *out_ptr = &compressed[0];
  auto out_size = compressed.size();

  std::ofstream file;
  auto absolute_path = boost::filesystem::absolute(filename).string();
  PrepareDir(fs::path(absolute_path).parent_path().string());

  LOGGER->info("Saving {}", absolute_path);
  file.open(absolute_path, std::ios_base::out | std::ios_base::binary);

  WriteBytes(out_ptr, out_size, &file);
  Ensures(Exists(absolute_path));
}
void WritePclTensor(const std::string &filename, const vector<Vec3> &pcl) {
  vector<float> data;
  for (const auto &p : pcl) {
    for (int i = 0; i < 3; ++i) {
      data.push_back(static_cast<float>(p[i]));
    }
  }
  SerializeTensor<float>(filename, data.data(), {pcl.size(), 3});
}

template void SerializeTensor<float>(const std::string &filename, const void *data, const std::vector<int> &shape);
template void SerializeTensor<double>(const std::string &filename, const void *data, const std::vector<int> &shape);
template void SerializeTensor<char>(const std::string &filename, const void *data, const std::vector<int> &shape);
template void SerializeTensor<int>(const std::string &filename, const void *data, const std::vector<int> &shape);
template void SerializeTensor<uint8_t>(const std::string &filename, const void *data, const std::vector<int> &shape);
template void SerializeTensor<uint16_t>(const std::string &filename, const void *data, const std::vector<int> &shape);
template void SerializeTensor<uint32_t>(const std::string &filename, const void *data, const std::vector<int> &shape);

template<typename T>
void WriteFloatsTxt(const std::string &txt_filename, int precision, const std::vector<T> &data) {
  std::ofstream ofile;
  ofile.open(txt_filename, std::ios::out);
  for (const auto &item : data) {
    ofile << std::setprecision(precision) << item << " ";
  }
  ofile.close();
}

template void WriteFloatsTxt<float>(const std::string &txt_filename, int precision, const std::vector<float> &data);
template void WriteFloatsTxt<double>(const std::string &txt_filename, int precision, const std::vector<double> &data);

vector<string> DirectoriesInDirectory(const string &dir) {
  fs::path path(dir);
  vector<string> paths;
  if (fs::exists(path) && fs::is_directory(path)) {
    fs::directory_iterator end_iter;
    for (fs::directory_iterator dir_iter(path); dir_iter != end_iter; ++dir_iter) {
      if (fs::is_directory(dir_iter->status())) {
        paths.push_back(dir_iter->path().string());
      }
    }
  }
  std::sort(std::begin(paths), std::end(paths));

  return paths;
}

void ReadLines(const string &filename, vector<string> *lines) {
  Ensures(Exists(filename));
  std::ifstream f(filename);

  std::string line;
  while (std::getline(f, line)) {  // does not include linebreak.
    lines->push_back(line);
  }
}

string JoinPath(const string &a, const string &b) {
  return (fs::path(a) / fs::path(b)).string();
}

void RemoveDirIfExists(const string &path) {
  if (fs::is_directory(path)) {
    LOGGER->info("rm -rf {}", fs::absolute(path).string());
    fs::remove_all(path);
  }
}

void RemoveFileIfExists(const string &path) {
  if (fs::is_regular_file(path)) {
    LOGGER->info("rm {}", fs::absolute(path).string());
    fs::remove(path);
  }
}

void WritePly(const string &filename, const vector<array<unsigned int, 3>> &faces, const vector<array<float, 3>> &vertices, bool is_binary) {
  Ensures(EndsWith(filename, ".ply"));
  Ensures(sizeof(unsigned int) == 4);

  std::filebuf buf;
  if (is_binary) {
    buf.open(filename, std::ios::out | std::ios::binary);
  } else {
    buf.open(filename, std::ios::out);
  }
  std::ostream outstream(&buf);
  if (outstream.fail()) {
    throw std::runtime_error("failed to open " + filename);
  }

  using namespace tinyply;

  PlyFile ply_file;

  ply_file.add_properties_to_element("vertex", {"x", "y", "z"}, Type::FLOAT32, vertices.size(), const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&vertices[0][0])), Type::INVALID, 0);
  ply_file.add_properties_to_element("face", {"vertex_indices"}, Type::UINT32, faces.size(), const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&faces[0][0])), Type::UINT8, 3);

  LOGGER->info("Writing {}", filename);

  // Write a binary file
  ply_file.write(outstream, is_binary);
}

void WriteObj(const string &filename, const vector<array<unsigned int, 3>> &faces, const vector<array<float, 3>> &vertices, float red, float green, float blue) {
  Expects(EndsWith(filename, ".obj"));
  std::ofstream file;
  file.open(filename);

  bool write_color = false;
  if (red >= 0 && green >= 0 && blue >= 0 && red <= 1 && green <= 1 && blue <= 1) {
    write_color = true;
  }

  for (const auto &v : vertices) {
    file << "v ";
    file << v[0] << " ";
    file << v[1] << " ";
    file << v[2];

    if (write_color) {
      file << " " << red << " " << green << " " << blue;
    }

    file << "\n";
  }

  for (const auto &f : faces) {
    file << "f ";
    file << f[0] + 1 << " ";
    file << f[1] + 1 << " ";
    file << f[2] + 1 << "\n";
  }

  file.close();
}
}

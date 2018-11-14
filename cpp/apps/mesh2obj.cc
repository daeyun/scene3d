#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>

#include "cxxopts.hpp"
#include "lib/file_io.h"

using namespace scene3d;

int main(int argc, const char **argv) {
  cxxopts::Options options("render_suncg", "Render multi-layer depth images");

  options.add_options()
      ("mesh_file", "Path to mesh file.", cxxopts::value<string>())
      ("r", "Optional R value in RGB.", cxxopts::value<float>()->default_value("-1"))
      ("g", "Optional G value in RGB.", cxxopts::value<float>()->default_value("-1"))
      ("b", "Optional B value in RGB.", cxxopts::value<float>()->default_value("-1"))
      ("help", "Display help.", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
  auto flags = options.parse(argc, argv);

  if (flags["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  // Initialize logging.
  spdlog::stdout_color_mt("console");

  // Check required flags.
  vector<string> required_flags = {"mesh_file"};
  for (const string &name: required_flags) {
    if (!flags.count(name)) {
      LOGGER->error("No argument specified for required option --{}. See --help.", name);
      throw std::runtime_error("");
    }
  }
  const string mesh_filename = flags["mesh_file"].as<string>();
  const float red = flags["r"].as<float>();
  const float green = flags["g"].as<float>();
  const float blue = flags["b"].as<float>();

  if (red >= 0 || green >= 0 || blue >= 0) {
    if (red < 0 || green < 0 || blue < 0) {
      throw std::runtime_error("All of r, g, b must be specified.");
    }
    if (red > 1 || green > 1 || blue > 1) {
      throw std::runtime_error("r, g, b must be in [0, 1]");
    }
  }

  const auto extention_index = mesh_filename.find_last_of('.');
  if (extention_index >= mesh_filename.length()) {
    throw std::runtime_error("Filename must contain a '.'");
  }
  if (mesh_filename.length() - extention_index > 7) {
    throw std::runtime_error("Filename contains too many characters after '.'");
  }

  std::vector<std::array<unsigned int, 3>> faces;
  std::vector<std::array<float, 3>> vertices;

  ReadFacesAndVertices(mesh_filename, &faces, &vertices, nullptr, nullptr);

  const string out_filename = mesh_filename.substr(0, extention_index) + ".obj";

  WriteObj(out_filename, faces, vertices, red, green, blue);
}

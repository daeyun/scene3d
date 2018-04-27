#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "spdlog/spdlog.h"

int main(int argc, char *argv[]) {
  spdlog::stdout_color_mt("console");
  int result = Catch::Session().run(argc, argv);
  return result;
}

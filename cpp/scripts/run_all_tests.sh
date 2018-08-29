#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_ROOT_DIR=$DIR/../..


MODE="release"

while getopts ":d" opt; do
    case $opt in
        d)
            MODE="debug"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
    esac
done



set -ex

cd $PROJ_ROOT_DIR


./cpp/scripts/build_all.sh

./cpp/cmake-build-release/tests/depth_renderer_test
./cpp/cmake-build-release/tests/depth_render_utils_test
./cpp/cmake-build-release/tests/pcl_test
./cpp/cmake-build-release/tests/ray_tracing_test
./cpp/cmake-build-release/tests/camera_test

echo "OK"

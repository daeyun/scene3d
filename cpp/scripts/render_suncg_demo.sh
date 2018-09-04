#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_DIR=$DIR/../..

set -ex

cd ${PROJ_DIR}

./cpp/cmake-build-release/apps/render_suncg \
    --height=240 \
    --width=320 \
    --cameras=resources/house/0004d52d1aeeb8ae6de39d6bd993e992/camera.txt \
    --obj=resources/house/0004d52d1aeeb8ae6de39d6bd993e992/house.obj \
    --json=resources/house/0004d52d1aeeb8ae6de39d6bd993e992/house_p.json \
    --category=resources/ModelCategoryMapping.csv \
    --out_dir=/tmp/scene3d/render_suncg

echo "OK"

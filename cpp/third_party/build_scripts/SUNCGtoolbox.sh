#!/usr/bin/env bash

set -ex

NAME=SUNCGtoolbox

# ---
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR=${DIR}/../install/${NAME}
mkdir -p ${INSTALL_DIR}
cd ${DIR}/../repos/${NAME}
# ---

cd gaps
make clean
make

rm -rf ${DIR}/../install/${NAME}/bin

mkdir ${DIR}/../install/${NAME}/gaps
mv bin ${DIR}/../install/${NAME}/gaps/bin

#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_DIR=$DIR/..

set -ex

cd $PROJ_DIR
pwd


rsync -atvurWAH --info=progress2,name0 . aleph0.ics:~/git/scene3d


echo "OK"


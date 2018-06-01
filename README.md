## Installation

Clone the repository with the `--recurse-submodules` option to download third party dependencies.

```
git clone --recurse-submodules git@github.com:daeyun/scene3d.git
```

#### Setting up Python virtual environment

Download the Python 3.6 version of [Anaconda](https://www.anaconda.com/download/).

From the repository's root directory, run the following command to create a new Conda environment and install and dependencies. `scene3d` can be replaced with any name you want.

```
conda env create --file python/environment.yml --name scene3d
```

More info on managing Conda environments can be found [here](https://conda.io/docs/user-guide/tasks/manage-environments.html).

#### Build instructions for C++

```
./cpp/third_party/build_scripts/SUNCGtoolbox.sh
./cpp/third_party/build_scripts/assimp.sh
./cpp/third_party/build_scripts/c-blosc.sh
```

```
./cpp/scripts/build_all.sh
```

Compiled binaries can be found in `./cpp/cmake-build-release/apps`.

For example,

```
./cpp/cmake-build-release/apps/render --help
```

## Development

### Project structure

#### Python

- The project's library is in [python/scene3d](python/scene3d).
- Work-in-progress scripts are in [python/scratch](python/scratch).
- Web UI code: [python/web](python/web).

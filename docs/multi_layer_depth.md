## How to use the multi-layer depth renderer

Compile the `render` executable using the provided build script [cpp/scripts/build_all.sh](cpp/scripts/build_all.sh) or download a compiled binary from [github.com/daeyun/scene3d/releases](https://github.com/daeyun/scene3d/releases).

See the source code [cpp/apps/render.cpp](cpp/apps/render.cpp) or use the `./render --help` command to see the options.

Before rendering, prepare the `.obj` mesh and a `.txt` file containing camera parameters (see [resources/depth_render/cam.txt](resources/depth_render/cam.txt) for an example).

There should be one line for each camera (same format as (pbrs)[https://github.com/yindaz/pbrs]):

```
[camera position x] [camera position y] [camera position z] [view direction x] [view direction y] [view direction z] [up x] [up y] [up z] [x fov] [y fov] [score]
```

- `fov` is the field-of-view angle measured from the principal-axis, so it should be half of the end-to-end angle, in radians. 
- `score` is not used, so it can be any value.

Example command:

```
./render -h 300 -w 400 --obj resources/depth_render/mocap_liangjian01_scene0.obj --out_dir ./out_depth --cameras resources/depth_render/cam.txt
```

Note that the x and y field of views should match the height and width of the output image, otherwise the aspect ratio won't be 1 and the program will generate an error. You can use this formula: `y_fov = arctan(tan(x_fov) * image_h/image_w)`.

`./render` outputs two files per camera:

- `000000.bin`: (N, H, W) tensor of depth images in front-to-back order. N is the number of layers. The first image is the "traditional" depth image. `nan` value means no ray hit; those values will need to be replaced to something else, depending on your needs.
- `000000_bg.bin`: (H, W) tensor containing the background depth image only.


## How to read the `.bin` files

`.bin` is a compressed binary file containing floating point array values in the following order:

```
[number of dimensions] [size of dimension 1] [size of dimension 2] ... [size of dimension n] [32-bit floating point values in c-contiguous (row-major) order]
```

This is compressed in [LZ4](https://github.com/lz4/lz4) format.

#### Example code for reading as numpy arrays

Install [blosc](https://github.com/Blosc/c-blosc) with `pip install blosc`. Then

```python
from scene3d import io_utils

images = io_utils.read_array_compressed('out_depth/000000.bin', dtype=np.float32)
```

If you wanted a more self-contained version:

```python
import blosc
import struct
import numpy as np

def bytes_to_array(s: bytes, dtype=np.float32):
    dims = struct.unpack('i', s[:4])[0]
    assert 0 <= dims < 1000
    shape = struct.unpack('i' * dims, s[4:4 * dims + 4])
    for dim in shape:
        assert dim > 0
    ret = np.frombuffer(s[4 * dims + 4:], dtype=dtype)
    assert ret.size == np.prod(shape), (ret.size, shape)
    ret.shape = shape
    return ret.copy()

def read_array_compressed(filename, dtype=np.float32):
    with open(filename, mode='rb') as f:
        compressed = f.read()
    decompressed = blosc.decompress(compressed)
    return bytes_to_array(decompressed, dtype=dtype)
```

The directory structure of our `v8` dataset:

```
all.txt              A list of all house_id, camera_id pairs, e.g. "0004dd3cb11e50530676f77b55262d38/000000". There are 568743 examples.
train.txt            A list of examples in the training set, of size 540705.  ~95%
test.txt             A list of examples in the evaluation set, of size 28038.  We take subsets of this to make validation and test splits.
renderings/          A directory containing the dataset. There is a subdirectory for each house model.


renderings/0004dd3cb11e50530676f77b55262d38
├── 000000_aabb.txt          -   Axis-aligned bounding box parameters. First three are bounding boxes generated using each of our three heuristics.
                                 The last one is the weighted average of the three.
├── 000000_cam.txt           -   Camera parameters of the frontal and overhead depth images. Same format as PBRS.
├── 000000_ldi.bin           -   Frontal multi-layer depth images. Size (4, 240, 320) and type float32. In the order of
                                 1. front  2. instance-exit  3. last-exit  4. room envelope.
├── 000000_ldi-o.bin         -   Overhead multi-layer depth images. Size (4, 300, 300) and type float32.
├── 000000_model.bin         -   Frontal pixel-wise model indices. Size (4, 240, 320) and type uint16. We have code that converts this to semantic segmentation,
                                 using the category mappings from SUNCG:  https://github.com/shurans/SUNCGtoolbox/blob/master/metadata/ModelCategoryMapping.csv   
├── 000000_model-o.bin       -   Overhead pixel-wise model indices.  Size (4, 300, 300) and type uint16.
├── 000000_n.bin             -   Surface normals. Size (3, 240, 320) and type float32.
├── 000000_oit.bin           -   Object-centered (inward normal direction) instance thickness. Size (240, 320) and type float32.
....
```

The PyTorch dataset object can be found in [dataset/v8.py](/python/scene3d/dataset/v8.py)

If you wanted to inspect the dataset or read the `.bin` files separately, you can use the `read_array_compressed` method in [io_utils.py](/python/scene3d/io_utils.py#L517)


You can find the RGB image for house `0004dd3cb11e50530676f77b55262d38` and camera `000000` in `pbrs/mlt_v2/0004dd3cb11e50530676f77b55262d38/000000_mlt.png`

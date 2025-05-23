- `NormalMapSimple`: The NormalMapSimple node is designed to generate normal maps from images, applying a transformation that simulates the appearance of surface variations and depth based on lighting. It utilizes image gradients to create a texture that represents the orientation of the surface in three-dimensional space, enhancing the visual perception of depth in 2D images.
    - Inputs:
        - `images` (Required): The 'images' input represents the source images for which normal maps are to be generated. This input is crucial for determining the texture and depth information that will be transformed into a normal map. Type should be `IMAGE`.
        - `scale_XY` (Required): The 'scale_XY' parameter adjusts the intensity of the surface variation effect in the generated normal map. A higher value increases the perceived depth by scaling the x and y components of the normal vectors. Type should be `FLOAT`.
    - Outputs:
        - `image`: The output is a transformed version of the input images, represented as normal maps. These maps encode the surface orientation and depth information, enhancing the 3D appearance of the original 2D images. Type should be `IMAGE`.

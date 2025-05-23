- `easy imageToBase64`: The `easy imageToBase64` node is designed to convert images from a tensor format to a Base64-encoded string. This functionality is essential for web applications and APIs where images need to be transmitted over the internet in a text-based format.
    - Inputs:
        - `image` (Required): The `image` parameter is the input tensor representing the image to be converted. It plays a crucial role in the conversion process, as it is the source image that will be transformed into a Base64-encoded string. Type should be `IMAGE`.
    - Outputs:
        - `string`: The output is a Base64-encoded string representation of the input image. This format is widely used for embedding images in HTML or CSS files, or for transmitting images over networks where binary data is not supported. Type should be `STRING`.

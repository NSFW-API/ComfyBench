- `AMT VFI`: The AMT_VFI node is designed for advanced motion transfer and frame interpolation in video processing. It leverages deep learning models to analyze and synthesize frames, achieving high-quality video frame interpolation by understanding and replicating the motion between consecutive frames.
    - Inputs:
        - `ckpt_name` (Required): The checkpoint name for the model, selecting the specific pre-trained model for frame interpolation. Type should be `COMBO[STRING]`.
        - `frames` (Required): The sequence of frames to be interpolated, serving as the input for the frame interpolation process. Type should be `IMAGE`.
        - `clear_cache_after_n_frames` (Required): Controls how often the cache is cleared during the frame interpolation process, optimizing memory usage. Type should be `INT`.
        - `multiplier` (Required): The factor by which the frame rate is increased, determining the number of frames generated between each pair of input frames. Type should be `INT`.
        - `optional_interpolation_states` (Optional): Optional states for interpolation, allowing for customization of the interpolation process. Type should be `INTERPOLATION_STATES`.
    - Outputs:
        - `image`: The output interpolated frames, showcasing the node's capability in enhancing video fluidity and detail. Type should be `IMAGE`.

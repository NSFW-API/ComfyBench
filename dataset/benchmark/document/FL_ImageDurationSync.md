- `FL_ImageDurationSync`: The FL_ImageDurationSync node is designed to synchronize the duration of a sequence of images with a specific musical beat per minute (BPM), frame count, and bars, adjusting the number of frames each image is held to match the desired total duration. This functionality is particularly useful in creating visuals that align with audio tracks, ensuring that the visual elements progress in harmony with the music's tempo.
    - Inputs:
        - `images` (Required): The sequence of images to be synchronized with the audio duration. This input is crucial for determining the base visual content that will be adjusted in duration. Type should be `IMAGE`.
        - `frame_count` (Required): Specifies the total number of frames for the output sequence, influencing how the images are stretched or compressed to fit the desired duration. Type should be `INT`.
        - `bpm` (Required): The beats per minute of the audio track, which is used to calculate the duration of each bar and, consequently, the total duration of the image sequence. Type should be `INT`.
        - `fps` (Required): The frames per second rate at which the images will be displayed, affecting the calculation of how long each image is held. Type should be `INT`.
        - `bars` (Required): The number of musical bars over which the images will be synchronized, directly impacting the total duration of the visual sequence. Type should be `FLOAT`.
    - Outputs:
        - `output_images`: The sequence of images adjusted to match the desired duration, ensuring synchronization with the audio track. Type should be `IMAGE`.
        - `hold_frames`: The number of frames each image is held, calculated based on the BPM, bars, and FPS to achieve the desired synchronization. Type should be `INT`.

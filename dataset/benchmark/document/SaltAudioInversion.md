- `SaltAudioInversion`: The node is designed to invert the waveform of an audio file, effectively flipping its phase. This process can be used to create unique sound effects or to cancel out specific audio components when mixed with the original.
    - Inputs:
        - `audio` (Required): The audio input is the raw audio data that will be inverted. This process alters the audio's waveform by reversing its phase, which can be useful for sound design or audio correction purposes. Type should be `AUDIO`.
    - Outputs:
        - `audio`: The output is the inverted audio data, with its waveform phase reversed from the original input. This can be used for creative audio effects or technical audio manipulation. Type should be `AUDIO`.

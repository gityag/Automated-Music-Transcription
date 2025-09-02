# Automated Music Transcription

This project is an implementation of an Automated Music Transcription (AMT) system. The goal of AMT is to convert an audio recording of a musical performance into a symbolic notation, such as a MIDI file. This implementation uses a deep learning approach to transcribe piano music from audio files.

## Features

*   **Audio to MIDI Conversion:** The core functionality of the project is to take an audio file (e.g., in WAV format) and transcribe it into a MIDI file.
*   **Constant-Q Transform (CQT):** The system uses CQT to convert the raw audio waveform into a time-frequency representation, which is more suitable for music analysis.
*   **Deep Learning Model:** A deep neural network is used to process the CQT spectrograms and predict the probability of each musical note being played at each time step.
*   **Data Processing Pipeline:** The project includes scripts for creating a dataset, encoding MIDI files into a suitable format for training, and decoding the model's predictions back into MIDI format.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You will need to have Python 3 and the following libraries installed:

*   `numpy`
*   `scipy`
*   `librosa`
*   `mido`
*   `tensorflow` or `keras`

You can install these dependencies using pip:

```bash
pip install numpy scipy librosa mido tensorflow
```

## Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/gityag/Automated-Music-Transcription.git
    ```

2.  Navigate to the project directory:
    ```bash
    cd Automated-Music-Transcription
    ```

## How to Use

The project is divided into several scripts, each responsible for a specific part of the transcription process:

*   `create_dataset.py`: This script is used to create a dataset for training the model. You will need a collection of audio files and their corresponding MIDI transcriptions.
*   `encode_midi_segments.py`: This script processes the MIDI files in your dataset and encodes them into a format that can be used to train the neural network.
*   `cqt.py`: This script can be used to perform the Constant-Q Transform on your audio files.
*   `models.py`: This script defines the architecture of the deep learning model used for transcription.
*   `get_model_prediction.py`: After training a model, you can use this script to get predictions for a new audio file.
*   `decode_midi.py`: This script takes the predictions from the model and decodes them into a standard MIDI file.
*   `get_results.py`: This script can be used to evaluate the performance of the transcription system.

## Future Work

This project provides a solid foundation for an Automated Music Transcription system, but there are many potential areas for improvement and future development:

*   **Multi-Instrument Transcription:** The current system is designed for piano music. It could be extended to transcribe music with multiple instruments.
*   **Real-Time Transcription:** A more advanced version of the system could be developed to perform music transcription in real-time.
*   **Improved Model Architecture:** The accuracy of the transcription could be improved by experimenting with different neural network architectures, such as recurrent neural networks (RNNs) or transformers.
*   **User Interface:** A graphical user interface (GUI) could be created to make the system more user-friendly.
*   **Integration with Music Software:** The transcription system could be integrated with music notation software or digital audio workstations (DAWs).

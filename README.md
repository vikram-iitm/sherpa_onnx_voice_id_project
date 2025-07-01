# Sherpa ONNX Voice ID Project

This project integrates Sherpa ONNX for voice identification tasks. It provides scripts for benchmarking, enrolling speakers, and identifying them using ONNX models.

## Features

- **ONNX Model Integration:** Utilizes Sherpa ONNX for efficient and portable voice identification.
- **Speaker Enrollment:** Enroll speakers by generating and storing their voice embeddings.
- **Voice Identification:** Identify speakers from audio files against enrolled voice profiles.
- **Benchmarking:** Tools to evaluate the performance of the voice identification system.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vikram-iitm/sherpa_onnx_voice_id_project.git
    cd sherpa_onnx_voice_id_project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv_sherpa_onnx
    source venv_sherpa_onnx/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Enroll Speakers

(Instructions for `sherpa_onnx_enroll.py` would go here, based on its functionality.)

### 2. Identify Speakers

(Instructions for `sherpa_onnx_identify.py` would go here, based on its functionality.)

### 3. Benchmarking

(Instructions for `sherpa_onnx_benchmark.py` would go here, based on its functionality.)

## Project Structure

- `scripts/`: Contains Python scripts for enrollment, identification, and benchmarking.
- `sherpa_onnx_salesperson_embeddings/`: Stores pre-computed embeddings of enrolled salespersons.
- `venv_sherpa_onnx/`: (Ignored) Python virtual environment.
- `models/`: (Ignored) Directory for ONNX models.
- `output/`: (Ignored) Directory for output files.
- `requirements.txt`: Project dependencies.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

(Specify your project's license here, e.g., MIT, Apache 2.0, etc.)

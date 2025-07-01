import argparse
import os
import numpy as np
import torch
import torchaudio
import sherpa_onnx
import soundfile as sf
import tempfile
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/pranavsinghpundir/sherpa_onnx_voice_id_project/.env")

# --- Constants ---
SHERPA_ONNX_SPEAKER_MODEL = "/Users/pranavsinghpundir/sherpa_onnx_voice_id_project/models/wespeaker_en_voxceleb_resnet34.onnx"
SHERPA_ONNX_VAD_MODEL = "/Users/pranavsinghpundir/sherpa_onnx_voice_id_project/models/silero_vad.onnx"
DATASET_BASE_PATH = "/Users/pranavsinghpundir/Desktop/new_data"

# Global Sherpa-ONNX objects
speaker_extractor = None
vad_detector = None

def get_sherpa_onnx_speaker_extractor():
    global speaker_extractor
    if speaker_extractor is None:
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=SHERPA_ONNX_SPEAKER_MODEL,
            num_threads=1, # Can be adjusted
            debug=False,
            provider="cpu", # Use "cuda" if GPU is available and sherpa-onnx is built with CUDA
        )
        if not config.validate():
            raise ValueError(f"Invalid speaker extractor config: {config}")
        speaker_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        print("Sherpa-ONNX Speaker Extractor loaded successfully.")
    return speaker_extractor

def get_sherpa_onnx_vad_detector():
    global vad_detector
    if vad_detector is None:
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = SHERPA_ONNX_VAD_MODEL
        vad_config.silero_vad.min_silence_duration = 0.25
        vad_config.silero_vad.min_speech_duration = 0.25
        vad_config.sample_rate = 16000 # Sherpa-ONNX VAD expects 16kHz
        if not vad_config.validate():
            raise ValueError(f"Invalid VAD config: {vad_config}")
        vad_detector = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)
        print("Sherpa-ONNX VAD Detector loaded successfully.")
    return vad_detector

def create_voice_print(audio_path, output_dir, speaker_id_for_filename):
    """
    Creates a voice print for the speaker using Sherpa-ONNX VAD and embedding.
    """
    print("Creating voice print...")
    speaker_extractor = get_sherpa_onnx_speaker_extractor()
    vad_detector = get_sherpa_onnx_vad_detector()

    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform_np = waveform.squeeze().numpy()

    samples_per_read = int(0.1 * 16000) # 0.1 second chunks
    total_samples = waveform_np.shape[0]
    processed_samples = 0

    all_speech_segments = []

    while processed_samples < total_samples:
        chunk = waveform_np[processed_samples : processed_samples + samples_per_read]
        vad_detector.accept_waveform(chunk)
        processed_samples += chunk.shape[0]

        while not vad_detector.empty():
            segment = vad_detector.front
            segment_samples_np = np.array(segment.samples)
            if segment_samples_np.shape[0] > 24000: # Min length of 1.5s (24000 samples at 16kHz)
                all_speech_segments.append(segment_samples_np)
            vad_detector.pop()

    if not all_speech_segments:
        print("No speech segments found for this audio. Cannot create voice print.")
        return

    # Combine all speech segments for a single, robust embedding
    combined_speech_waveform = np.concatenate(all_speech_segments)

    stream = speaker_extractor.create_stream()
    stream.accept_waveform(sample_rate=16000, waveform=combined_speech_waveform)
    stream.input_finished()
    embedding = speaker_extractor.compute(stream)
    
    reference_embedding = np.array(embedding)
    embedding_path = os.path.join(output_dir, f"{speaker_id_for_filename}.npy")
    np.save(embedding_path, reference_embedding)
    print(f"Voice print saved to: {embedding_path}")

def main():
    parser = argparse.ArgumentParser(description="Enroll speakers from a dataset and create voice prints using Sherpa-ONNX.")
    args = parser.parse_args()

    output_dir_base = "/Users/pranavsinghpundir/sherpa_onnx_voice_id_project"
    embeddings_dir = os.path.join(output_dir_base, "sherpa_onnx_salesperson_embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    # Discover training audio files
    speaker_audio_files = defaultdict(list)
    training_audio_dir = os.path.join(DATASET_BASE_PATH, "Training Audio")
    for root, dirs, files in os.walk(training_audio_dir):
        for file in files:
            if file.endswith(".wav"):
                speaker_id = os.path.basename(root) # e.g., Abhay-001
                full_audio_path = os.path.join(root, file)
                speaker_audio_files[speaker_id].append(full_audio_path)

    for speaker_id, audio_paths in speaker_audio_files.items():
        print(f"\nProcessing speaker: {speaker_id}")
        # Concatenate audio files for enrollment
        concatenated_waveform = []
        for ap in audio_paths:
            try:
                waveform, sample_rate = torchaudio.load(ap)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                concatenated_waveform.append(waveform)
            except Exception as e:
                print(f"Error loading {ap}: {e}")
                continue
        
        if not concatenated_waveform:
            print(f"Skipping {speaker_id}: No valid audio files found.")
            continue

        concatenated_waveform = torch.cat(concatenated_waveform, dim=1)
        
        # Save concatenated audio to a temporary file for processing by create_voice_print
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
            sf.write(temp_audio_file.name, concatenated_waveform.squeeze().numpy(), 16000)
            create_voice_print(temp_audio_file.name, embeddings_dir, speaker_id)

if __name__ == "__main__":
    main()
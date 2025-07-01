import argparse
import os
import numpy as np
import torch
import torchaudio
import sherpa_onnx
import soundfile as sf
import tempfile
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
import time

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/pranavsinghpundir/sherpa_onnx_voice_id_project/.env")

# --- Constants ---
SIMILARITY_THRESHOLD = 0.55  # Adjust this value based on testing (0.0 to 1.0)
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

def load_salesperson_embeddings(embeddings_dir):
    """
    Loads all salesperson embeddings from the specified directory.
    """
    print("Loading salesperson embeddings...")
    embeddings = {}
    for file_name in os.listdir(embeddings_dir):
        if file_name.endswith(".npy"):
            salesperson_id = os.path.splitext(file_name)[0]
            embedding_path = os.path.join(embeddings_dir, file_name)
            embeddings[salesperson_id] = np.load(embedding_path)
            print(f"Loaded embedding for: {salesperson_id}")
    return embeddings

def identify_speaker_in_audio(audio_path, salesperson_embeddings):
    """
    Identifies a speaker in a single audio file using Sherpa-ONNX VAD and iterative embedding.
    Returns the identified speaker ID or None if no match.
    """
    speaker_extractor = get_sherpa_onnx_speaker_extractor()
    vad_detector = get_sherpa_onnx_vad_detector()

    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Convert waveform to numpy array for sherpa-onnx
    waveform_np = waveform.squeeze().numpy()

    # Process audio in chunks for VAD
    samples_per_read = int(0.1 * 16000) # 0.1 second chunks
    total_samples = waveform_np.shape[0]
    processed_samples = 0

    all_speech_segments = [] # Store all detected speech segments

    while processed_samples < total_samples:
        chunk = waveform_np[processed_samples : processed_samples + samples_per_read]
        vad_detector.accept_waveform(chunk)
        processed_samples += chunk.shape[0]

        while not vad_detector.empty():
            segment = vad_detector.front
            # Ensure segment is long enough for embedding (1.5s = 24000 samples at 16kHz)
            segment_samples_np = np.array(segment.samples)
            if segment_samples_np.shape[0] > 24000:
                all_speech_segments.append(segment_samples_np)
            vad_detector.pop()

    # Iterative Query Embedding Strategy
    # Sort segments by length (longest first) to prioritize more informative segments
    all_speech_segments.sort(key=lambda x: x.shape[0], reverse=True)

    query_tiers = [15, 30, 60, 120, float('inf')] # in seconds
    current_audio_duration = 0
    current_embeddings = []

    best_match_id = None
    best_similarity = -1
    all_similarities = {}

    found_match = False
    for segment_samples in all_speech_segments:
        # Extract embedding for the current segment
        stream = speaker_extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=segment_samples)
        stream.input_finished()
        embedding = speaker_extractor.compute(stream)
        embedding = np.array(embedding)
        current_embeddings.append(embedding)

        current_audio_duration += segment_samples.shape[0] / 16000.0

        # Check if we've reached a new query tier or processed all segments
        if current_audio_duration >= query_tiers[0] or segment_samples is all_speech_segments[-1]:
            query_embedding = np.mean(np.array(current_embeddings), axis=0)

            best_match_id = None
            best_similarity = -1
            all_similarities = {}

            for salesperson_id, reference_embedding in salesperson_embeddings.items():
                similarity = 1 - cosine(query_embedding, reference_embedding)
                all_similarities[salesperson_id] = similarity
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = salesperson_id

            if best_similarity > SIMILARITY_THRESHOLD:
                return best_match_id, best_similarity, all_similarities

            if query_tiers[0] != float('inf'):
                query_tiers.pop(0)
                if not query_tiers: # If no more tiers, set to infinity to process remaining segments
                    query_tiers.append(float('inf'))

    return None, best_similarity, all_similarities

def main():
    parser = argparse.ArgumentParser(description="Benchmark salesperson identification using Sherpa-ONNX.")
    args = parser.parse_args()

    output_dir_base = "/Users/pranavsinghpundir/sherpa_onnx_voice_id_project"
    embeddings_dir = os.path.join(output_dir_base, "sherpa_onnx_salesperson_embeddings")

    salesperson_embeddings = load_salesperson_embeddings(embeddings_dir)
    if not salesperson_embeddings:
        print("No salesperson embeddings found. Please enroll speakers first.")
        return

    enrolled_speaker_ids = set(salesperson_embeddings.keys())
    print(f"Benchmarking with {len(enrolled_speaker_ids)} enrolled speakers.")

    test_audio_files = []
    testing_audio_dir = os.path.join(DATASET_BASE_PATH, "Testing Audio")
    for root, dirs, files in os.walk(testing_audio_dir):
        for file in files:
            if file.endswith(".wav"):
                file_name_parts = os.path.splitext(file)[0].split('_')
                ground_truth_speaker_name = file_name_parts[0]

                ground_truth_speaker_id = None
                for enrolled_id in enrolled_speaker_ids:
                    if enrolled_id.startswith(ground_truth_speaker_name + "-"):
                        ground_truth_speaker_id = enrolled_id
                        break
                
                if ground_truth_speaker_id and ground_truth_speaker_id in enrolled_speaker_ids:
                    full_audio_path = os.path.join(root, file)
                    test_audio_files.append({
                        "path": full_audio_path,
                        "ground_truth_id": ground_truth_speaker_id
                    })

    if not test_audio_files:
        print("No test audio files found for the enrolled speakers.")
        return

    correct_identifications = 0
    total_identifications = 0
    total_time_taken = 0

    print(f"Starting benchmark on {len(test_audio_files)} filtered test audio files...")
    for i, test_item in enumerate(test_audio_files):
        audio_path = test_item["path"]
        ground_truth_id = test_item["ground_truth_id"]
        
        print(f"\nProcessing test file {i+1}/{len(test_audio_files)}: {os.path.basename(audio_path)}")
        start_time = time.time()
        identified_speaker_id, best_similarity, all_similarities = identify_speaker_in_audio(audio_path, salesperson_embeddings)
        end_time = time.time()
        time_taken = end_time - start_time
        total_time_taken += time_taken

        total_identifications += 1
        if identified_speaker_id == ground_truth_id:
            correct_identifications += 1
            print(f"Correctly identified {ground_truth_id} in {time_taken:.2f} seconds.")
        else:
            print(f"Incorrectly identified {ground_truth_id} as {identified_speaker_id} in {time_taken:.2f} seconds.")
            print(f"  Best similarity: {best_similarity:.4f}")
            print(f"  All similarities: {all_similarities}")

    accuracy = (correct_identifications / total_identifications) * 100 if total_identifications > 0 else 0
    average_time = total_time_taken / total_identifications if total_identifications > 0 else 0

    print("\n--- Benchmarking Results ---")
    print(f"Total Test Files Processed: {total_identifications}")
    print(f"Correct Identifications: {correct_identifications}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Identification Time: {average_time:.2f} seconds per file")

if __name__ == "__main__":
    main()
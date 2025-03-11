import time
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, pipeline
from transformers.models.whisper import WhisperFeatureExtractor, WhisperTokenizer

def transcribe_audio(audio_path=None, model_path="distil-whisper/distil-large-v2", 
                    language="english", batch_size=16, chunk_length=15,
                    use_sample_audio=False):
    """
    Transcribe audio using Whisper model.
    
    Args:
        audio_path (str): Path to the audio file to transcribe
        model_path (str): Path to the model or HuggingFace repo ID
        language (str): Language of the audio
        batch_size (int): Batch size for processing chunks
        chunk_length (int): Length of audio chunks in seconds
        use_sample_audio (bool): Whether to use a sample audio from LibriSpeech
        
    Returns:
        tuple: (transcription, inference time in seconds)
    """
    print(f"Loading model from {model_path}...")
    
    # Check device
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the model and move it to XPU
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
    model.to(device)
    model.config.forced_decoder_ids = None
    
    # Load feature extractor and tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, language=language)
    
    # Create the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        chunk_length_s=chunk_length,
        device=device
    )
    
    # Get audio data
    if use_sample_audio:
        print("Loading sample audio from LibriSpeech dataset...")
        dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        audio = dataset[0]["audio"]
    elif audio_path:
        # This requires loading the audio file manually
        from datasets import Audio
        import numpy as np
        
        print(f"Loading audio from {audio_path}...")
        audio_loader = Audio(sampling_rate=16000)
        
        if os.path.exists(audio_path):
            audio = audio_loader.decode_example({"array": np.array([]), "path": audio_path})
        else:
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
    else:
        raise ValueError("Either audio_path or use_sample_audio must be provided")
    
    # Perform transcription
    print("Transcribing audio...")
    start = time.time()
    prediction = pipe(audio, batch_size=batch_size)
    end = time.time()
    
    inference_time = end - start
    transcription = prediction["text"]
    
    return transcription, inference_time

def main():
    print("Whisper Speech-to-Text Application")
    print("---------------------------------")
    
    # Get user inputs
    model_choice = input("Enter model path or HuggingFace repo ID (default: distil-whisper/distil-large-v2): ").strip()
    if not model_choice:
        model_choice = "distil-whisper/distil-large-v2"
    
    audio_source = input("Use sample audio? (y/n, default: y): ").strip().lower()
    use_sample = audio_source != 'n'
    
    audio_path = None
    if not use_sample:
        audio_path = input("Enter path to audio file: ").strip()
        if not os.path.exists(audio_path):
            print(f"Warning: File {audio_path} not found. Using sample audio instead.")
            use_sample = True
    
    language = input("Enter language (default: english): ").strip()
    if not language:
        language = "english"
    
    try:
        batch_size = int(input("Enter batch size (default: 16): ") or "16")
    except ValueError:
        batch_size = 16
        
    try:
        chunk_length = int(input("Enter chunk length in seconds (default: 15): ") or "15")
    except ValueError:
        chunk_length = 15
    
    print("\nStarting transcription...\n")
    
    try:
        transcription, inference_time = transcribe_audio(
            audio_path=audio_path,
            model_path=model_choice,
            language=language,
            batch_size=batch_size,
            chunk_length=chunk_length,
            use_sample_audio=use_sample
        )
        
        print("\n" + "="*50)
        print("Transcription:")
        print(transcription)
        print("="*50)
        print(f"\nInference time: {inference_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during transcription: {e}")

if __name__ == '__main__':
    main()
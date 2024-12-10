from pyannote.audio import Pipeline
from pyannote.core import Segment
import whisper
import csv
import time
import torch

# Define helper functions
def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res['segments']:
        start = item['start']
        end = item['end']
        text = item['text']
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts

def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text

def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence

PUNC_SENT_END = ['.', '?', '!']

def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk
        elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text

def diarize_text(transcribe_res, diarization_result):
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed

# Main processing workflow
def process_audio(audio_file, auth_token):
    # Initialize pipeline and model
    print("Initializing models...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            use_auth_token=auth_token)

        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))

        model = whisper.load_model("tiny.en")
    except Exception as e:
        print(f"Error during model initialization: {e}")
        return None

    # Perform transcription and speaker diarization
    print("Processing audio for transcription and diarization...")
    try:
        #Transcribe
        start_time = time.time()
        asr_result = model.transcribe(audio_file)
        elapsed_time = time.time() - start_time
        print(f"Processing time for transcribing: {elapsed_time:.2f} seconds")

        #Diarize
        start_time = time.time()
        diarization_result = pipeline(audio_file)
        elapsed_time = time.time() - start_time
        print(f"Processing time for diarizing: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"Error during processing: {e}")
        return None

    # Merge results
    print("Merging transcription and diarization results...")
    final_result = diarize_text(asr_result, diarization_result)

    if not final_result:
        print("No results obtained after merging.")
    else:
        print("Processing complete.")

    return final_result
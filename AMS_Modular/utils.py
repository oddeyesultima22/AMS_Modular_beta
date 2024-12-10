from pyannote.core import Segment
import csv

def get_text_with_timestamp(transcribe_res):
    return [(Segment(item['start'], item['end']), item['text']) for item in transcribe_res['segments']]

def add_speaker_info_to_text(timestamp_texts, diarization_result):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = diarization_result.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text

def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence

def merge_sentence(spk_text):
    PUNC_SENT_END = ['.', '?', '!']
    merged_spk_text, pre_spk, text_cache = [], None, []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and text_cache:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
        elif text and text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
        else:
            text_cache.append((seg, spk, text))
        pre_spk = spk
    if text_cache:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text

# def save_to_csv(merged_text, output_csv):
#     with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['start_time', 'end_time', 'speaker', 'transcript']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         writer.writeheader()
#         for seg, spk, text in merged_text:
#             writer.writerow({
#                 'start_time': f'{seg.start:.2f}',
#                 'end_time': f'{seg.end:.2f}',
#                 'speaker': spk,
#                 'transcript': text
#             })
#     print(f"Results saved to: {output_csv}")

def save_to_csv(merged_text, output_csv):
    speaker_mapping = {"SPEAKER_00": "A", "SPEAKER_01": "C"}  # Add more mappings as needed

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Person', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for seg, spk, text in merged_text:
            # Map speaker name
            spk = speaker_mapping.get(spk, spk)
            writer.writerow({
                'Person': spk,
                'Text': text
            })
    print(f"Formatted results saved to: {output_csv}")

import argparse
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from datasets import load_dataset
#from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import os
# https://github.com/linto-ai/whisper-timestamped
import whisper_timestamped as whisper

def main(input_path, output_path, save_timestamps_path):
    # Extract audio from video
    """
    extract_audio_from_video(input_path)

    # Generate timestamps with transcriptions from audio
    if args.language:
        timestamps = generate_timestamps("tmp/audio.mp3", language=args.language)
    else:
        timestamps = generate_timestamps("tmp/audio.mp3")

    if save_timestamps_path:
        with open(save_timestamps_path, "w") as f:
            f.write(str(timestamps))
    """

    # Generate video with captions
    with open("captions.txt", "r") as f:
        timestamps = eval(f.read())
    generate_video(timestamps, input_path, output_path)

    # Delete tmp/audio.mp3
    os.remove("tmp/audio.mp3")

def extract_audio_from_video(input_path):
    print("Separating out mp3...")
    video = mp.VideoFileClip(input_path)
    audio = video.audio
    audio.write_audiofile("tmp/audio.mp3")


def generate_timestamps(input_path, language="en"):
    print("Generating timestamps...")
    audio = whisper.load_audio(input_path)
    model = whisper.load_model("large", device="cpu")
    result = whisper.transcribe(model, audio, language=language)

    return result


def generate_video(timestamps, input_path, output_path):
    print("Generating video with captions...")
    # Load the background video
    video = VideoFileClip(input_path)

    # Create a list to hold the caption clips
    captions = []

    # For each chunk in the timestamps
    for chunk in timestamps['segments']:

        # Iterate through each word in the segment and create a caption
        for word in chunk['words']:
            # Create a TextClip for each word
            word_caption = TextClip(word['text'], fontsize=40, color='white', stroke_color='black', stroke_width=2, align='center', font='Arial-Bold', method='caption', size=video.size).set_position(('center', 'bottom'))
            
            # Set the start and end times for the word caption
            start_time, end_time = word['start'], word['end']
            word_caption = word_caption.set_start(start_time).set_duration(end_time - start_time)
            
            # Add the word caption to the list
            captions.append(word_caption)


    # Overlay the captions on the video
    video = CompositeVideoClip([video] + captions)
    video.duration = video.clips[0].duration

    # Write the video to a file
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some videos.")
    parser.add_argument("--input", type=str, required=True, help="Input mp4 file path")
    parser.add_argument("--output", type=str, required=True, help="Output mp4 file path")
    parser.add_argument("--save_timestamps", type=str, required=False, help="Path to save timestamps txt file")
    parser.add_argument("--language", type=str, required=False, help="Language for speech recognition")
    
    args = parser.parse_args()
    main(args.input, args.output, args.save_timestamps)

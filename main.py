import argparse
import moviepy.editor as mp
from moviepy.editor import *
from moviepy.video.tools.segmenting import findObjects
from datasets import load_dataset
#from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import os
# https://github.com/linto-ai/whisper-timestamped
import whisper_timestamped as whisper

# Define text style
font = 'Arial-Bold'
fontsize = 40
color = 'white'
stroke_color = 'black'
stroke_width = 2

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
    
    timestamps = modify_timestamps(timestamps)
    line_timestamps = split_text_into_lines(timestamps)

    generate_video(line_timestamps, input_path, output_path)

    # Delete tmp/audio.mp3
    #os.remove("tmp/audio.mp3")

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

def modify_timestamps(timestamps):
    print("Modifying timestamps to line level...")
    wordlevel_info = []
    for segment in timestamps['segments']:
        for word in segment['words']:
            wordlevel_info.append({'word': word['text'], 'start': word['start'], 'end': word['end']})

    return wordlevel_info

def generate_video(line_timestamps, input_path, output_path):
    print("Generating video with captions...")
    input_video = VideoFileClip(input_path)
    frame_size = input_video.size

    all_linelevel_splits = []

    for i, line in enumerate(line_timestamps):
        print("Caption generation progress: ", i, "/", len(line_timestamps))
        out_clips, positions = create_caption(line, frame_size)

        max_width = 0
        max_height = 0

        for position in positions:
            x_pos, y_pos = position['x_pos'], position['y_pos']
            width, height = position['width'], position['height']

            max_width = max(max_width, x_pos + width)
            max_height = max(max_height, y_pos + height)

        color_clip = ColorClip(size=(int(max_width * 1.1), int(max_height * 1.1)),
                               color=(64, 64, 64))
        color_clip = color_clip.set_opacity(.6)
        color_clip = color_clip.set_start(line['start']).set_duration(line['end'] - line['start'])

        clip_to_overlay = CompositeVideoClip([color_clip] + out_clips)
        clip_to_overlay = clip_to_overlay.set_position("bottom")

        all_linelevel_splits.append(clip_to_overlay)

    input_video_duration = input_video.duration

    final_video = CompositeVideoClip([input_video] + all_linelevel_splits)

    # Set the audio of the final video to be the same as the input video
    final_video = final_video.set_audio(input_video.audio)

    # Write the video to a file
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

def split_text_into_lines(data):
    MaxChars = 30
    #maxduration in seconds
    MaxDuration = 2.5
    #Split if nothing is spoken (gap) for these many seconds
    MaxGap = 1.5

    subtitles = []
    line = []
    line_duration = 0
    line_chars = 0

    for idx, word_data in enumerate(data):
        word = word_data["word"]
        start = word_data["start"]
        end = word_data["end"]

        line.append(word_data)
        line_duration += end - start

        temp = " ".join(item["word"] for item in line)

        # Check if adding a new word exceeds the maximum character count or duration
        new_line_chars = len(temp)

        duration_exceeded = line_duration > MaxDuration
        chars_exceeded = new_line_chars > MaxChars
        if idx > 0:
            gap = word_data['start'] - data[idx-1]['end']
            maxgap_exceeded = gap > MaxGap
        else:
            maxgap_exceeded = False

        if duration_exceeded or chars_exceeded or maxgap_exceeded:
            if line:
                subtitle_line = {
                    "word": " ".join(item["word"] for item in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "textcontents": line
                }
                subtitles.append(subtitle_line)
                line = []
                line_duration = 0
                line_chars = 0

    if line:
        subtitle_line = {
            "word": " ".join(item["word"] for item in line),
            "start": line[0]["start"],
            "end": line[-1]["end"],
            "textcontents": line
        }
        subtitles.append(subtitle_line)

    return subtitles

def create_caption(textJSON, framesize,font = "Helvetica",color='white', highlight_color='yellow',stroke_color='black',stroke_width=1.5):
    wordcount = len(textJSON['textcontents'])
    full_duration = textJSON['end']-textJSON['start']

    word_clips = []
    xy_textclips_positions =[]

    x_pos = 0
    y_pos = 0
    line_width = 0  # Total width of words in the current line
    frame_width = framesize[0]
    frame_height = framesize[1]

    x_buffer = frame_width*1/10

    max_line_width = frame_width - 2 * (x_buffer)

    fontsize = int(frame_height * 0.075) #7.5 percent of video height

    space_width = ""
    space_height = ""

    for index,wordJSON in enumerate(textJSON['textcontents']):
      duration = wordJSON['end']-wordJSON['start']
      word_clip = TextClip(wordJSON['word'], font = font,fontsize=fontsize, color=color,stroke_color=stroke_color,stroke_width=stroke_width).set_start(textJSON['start']).set_duration(full_duration)
      word_clip_space = TextClip(" ", font = font,fontsize=fontsize, color=color).set_start(textJSON['start']).set_duration(full_duration)
      word_width, word_height = word_clip.size
      space_width,space_height = word_clip_space.size
      if line_width + word_width+ space_width <= max_line_width:
            # Store info of each word_clip created
            xy_textclips_positions.append({
                "x_pos":x_pos,
                "y_pos": y_pos,
                "width" : word_width,
                "height" : word_height,
                "word": wordJSON['word'],
                "start": wordJSON['start'],
                "end": wordJSON['end'],
                "duration": duration
            })

            word_clip = word_clip.set_position((x_pos, y_pos))
            word_clip_space = word_clip_space.set_position((x_pos+ word_width, y_pos))

            x_pos = x_pos + word_width+ space_width
            line_width = line_width+ word_width + space_width
      else:
            # Move to the next line
            x_pos = 0
            y_pos = y_pos+ word_height+10
            line_width = word_width + space_width

            # Store info of each word_clip created
            xy_textclips_positions.append({
                "x_pos":x_pos,
                "y_pos": y_pos,
                "width" : word_width,
                "height" : word_height,
                "word": wordJSON['word'],
                "start": wordJSON['start'],
                "end": wordJSON['end'],
                "duration": duration
            })

            word_clip = word_clip.set_position((x_pos, y_pos))
            word_clip_space = word_clip_space.set_position((x_pos+ word_width , y_pos))
            x_pos = word_width + space_width


      word_clips.append(word_clip)
      word_clips.append(word_clip_space)


    for highlight_word in xy_textclips_positions:

      word_clip_highlight = TextClip(highlight_word['word'], font = font,fontsize=fontsize, color=highlight_color,stroke_color=stroke_color,stroke_width=stroke_width).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
      word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
      word_clips.append(word_clip_highlight)

    return word_clips,xy_textclips_positions



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some videos.")
    parser.add_argument("--input", type=str, required=True, help="Input mp4 file path")
    parser.add_argument("--output", type=str, required=True, help="Output mp4 file path")
    parser.add_argument("--save_timestamps", type=str, required=False, help="Path to save timestamps txt file")
    parser.add_argument("--language", type=str, required=False, help="Language for speech recognition")
    
    args = parser.parse_args()
    main(args.input, args.output, args.save_timestamps)

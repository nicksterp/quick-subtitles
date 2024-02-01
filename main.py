import argparse
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def main(input_path, output_path, save_timestamps_path):

    # Extract audio from video
    extract_audio_from_video(input_path)

    # Generate timestamps with transcriptions from audio
    timestamps = generate_timestamps("/tmp/audio.mp3")

    if save_timestamps_path:
        with open(save_timestamps_path, "w") as f:
            f.write(timestamps)

    # Generate video with captions
    generate_video(timestamps, input_path, output_path)

    # Delete tmp/audio.mp3
    os.remove("/tmp/audio.mp3")

def extract_audio_from_video(input_path):
    video = mp.VideoFileClip(input_path)
    audio = video.audio
    audio.write_audiofile("/tmp/audio.mp3")


def generate_timestamps(input_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audio_path, return_timestamps=True)

    return result


def generate_video(timestamps, input_path, output_path):
    # Load the background video
    video = VideoFileClip(input_path)

    # Create a list to hold the caption clips
    captions = []

    # For each chunk in the timestamps
    for chunk in timestamps['chunks']:
        # Create a TextClip for the caption, centered horizontally
        caption = TextClip(chunk['text'], fontsize=40, color='white', stroke_color='black', stroke_width=2, align='center', font='Arial-Bold', method='caption', size=video.size).set_position(('center', 'bottom'))

        # Set the start and end times for the caption
        start_time, end_time = chunk['timestamp']
        caption = caption.set_start(start_time).set_duration(end_time - start_time)

        # Add the caption to the list
        captions.append(caption)

    # Overlay the captions on the video
    video = CompositeVideoClip([video] + captions)

    # Write the video to a file
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some videos.")
    parser.add_argument("--input", type=str, required=True, help="Input mp4 file path")
    parser.add_argument("--output", type=str, required=True, help="Output mp4 file path")
    parser.add_argument("--save-timestamps", type=str, required=False, help="Path to save timestamps txt file")
    
    args = parser.parse_args()
    main(args.input, args.output, args.save_timestamps)

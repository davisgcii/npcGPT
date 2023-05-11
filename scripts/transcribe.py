import os
import re
import whisper
import yt_dlp
from whisper.utils import get_writer
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import wave
import contextlib
import datetime
import subprocess
from PIL import Image
from io import BytesIO

NUM_SPEAKERS = 5  # number of speakers to diarize
PAUSE_THRESHOLD = 2.5  # how many seconds to wait between dialogue groups
DATA_FOLDER = "data"  # parent folder to save the output image/caption pair folders
TEMP_FOLDER = "temp_files"  # folder to save temporary video, audio, text files
WHISPER_MODEL = "small.en"  # seems to do well enough while running locally

# see link below for more details on whisper models
# https://huggingface.co/openai/whisper-medium.en


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\/:*?"<>|]', "_", filename).replace("：", "_")


def transcribe_and_diarize(input_file, num_speakers=NUM_SPEAKERS):
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(input_file)
    segments = result["segments"]

    duration = get_duration(input_file)
    if len(segments) == 1:
        segments[0]["speaker"] = "SPEAKER 1"
    else:
        embeddings = make_embeddings(input_file, segments, duration)
        add_speaker_labels(segments, embeddings, num_speakers)
    output = get_output(segments)
    return output


def get_duration(path):
    with contextlib.closing(wave.open(path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def make_embeddings(path, segments, duration):
    embeddings = np.zeros(shape=(len(segments), 192))
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    audio = Audio()
    for i, segment in enumerate(segments):
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(path, clip)
        embeddings[i] = embedding_model(waveform[None])
    return np.nan_to_num(embeddings)


def add_speaker_labels(segments, embeddings, num_speakers):
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)


def time(secs):
    return datetime.timedelta(seconds=round(secs))


def get_output(segments):
    output = ""
    for i, segment in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            if i != 0:
                output += "\n\n"
            output += segment["speaker"] + " " + str(time(segment["start"])) + "\n\n"
        output += segment["text"][1:] + " "
    return output


def generate_captions(url):
    video_url = url

    with yt_dlp.YoutubeDL() as ydl:
        info = ydl.extract_info(video_url, download=False)
        sanitized_title = sanitize_filename(info["title"])

    yt_opts = {
        "outtmpl": f"{TEMP_FOLDER}/{sanitized_title}.%(ext)s",
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(yt_opts) as ydl:
        ydl.download([video_url])
        input_file = f"{TEMP_FOLDER}/{sanitized_title}.mp4"

        # Extract audio from video and save as WAV file
        audio_file = f"{TEMP_FOLDER}/{sanitized_title}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_file,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                audio_file,
            ]
        )

        # Transcribe and diarize the audio
        output = transcribe_and_diarize(audio_file, num_speakers=NUM_SPEAKERS)

        # Save the output
        output_directory = (
            TEMP_FOLDER  # Change this to the directory you want to save the output
        )
        with open(os.path.join(output_directory, f"{sanitized_title}.txt"), "w") as f:
            f.write(output)

        return sanitized_title


def save_image_caption_pair(caption_group, timestamp, video_file, output_folder):
    # only save captions if there is more than one line of dialogue
    if len(caption_group) <= 1:
        return

    # Concatenate captions in the group
    caption_text = " ".join(caption_group)

    # Extract image still from the video
    image_data = subprocess.check_output(
        [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            video_file,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-",
        ]
    )
    image = Image.open(BytesIO(image_data))

    # Save the image
    image_filename = os.path.join(output_folder, f"image_{timestamp}.png")
    image.save(image_filename)

    # Save the caption
    caption_filename = os.path.join(output_folder, f"caption_{timestamp}.txt")
    with open(caption_filename, "w") as f:
        f.write(caption_text)


def create_image_caption_pairs(
    output_txt, video_file, output_folder, pause_threshold=2
):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Parse the output text file
    with open(output_txt, "r") as f:
        lines = f.readlines()

    # Variables to keep track of current group of captions
    current_group = []
    current_timestamp = None
    last_timestamp = None
    current_speaker = ""  # Initialize the current_speaker variable

    # Iterate through the lines in the output text file
    for line in lines:
        # Match speaker lines (e.g., "SPEAKER 2 0:00:00")
        speaker_match = re.match(r"(SPEAKER \d+) (\d+:\d+:\d+)", line)
        if speaker_match:
            speaker_label = speaker_match.group(1)
            timestamp_str = speaker_match.group(2)
            timestamp = sum(
                int(x) * 60**i
                for i, x in enumerate(reversed(timestamp_str.split(":")))
            )

            # Initialize current_timestamp if it's the first speaker line
            if current_timestamp is None:
                current_timestamp = timestamp

            # Check if the pause between captions is greater than the threshold
            if (
                last_timestamp is not None
                and (timestamp - last_timestamp) > pause_threshold
            ):
                # Save the current group of captions and extract the corresponding image still
                save_image_caption_pair(
                    current_group, current_timestamp, video_file, output_folder
                )

                # Start a new group of captions
                current_group = []
                current_timestamp = timestamp  # Update the current timestamp

            last_timestamp = timestamp
            current_speaker = speaker_label  # Update the current speaker
        else:
            # if there is a line of dialoge, add the caption to the current group
            if line.strip() != "":
                current_group.append(current_speaker + ": " + line.strip() + "\n")

    # Save the last group of captions
    if current_group:
        save_image_caption_pair(
            current_group, current_timestamp, video_file, output_folder
        )


def process_video(url):
    video_title = generate_captions(url)
    output_folder = f"{DATA_FOLDER}/{video_title}"

    create_image_caption_pairs(
        f"{TEMP_FOLDER}/{video_title}.txt",
        f"{TEMP_FOLDER}/{video_title}.mp4",
        output_folder,
        pause_threshold=PAUSE_THRESHOLD,
    )

    # Delete the video and audio files and transcript
    os.remove(f"{TEMP_FOLDER}/{video_title}.txt")
    os.remove(f"{TEMP_FOLDER}/{video_title}.wav")
    os.remove(f"{TEMP_FOLDER}/{video_title}.mp4")


def get_urls_from_playlist(url: str) -> list:
    ydl_opts = {"extract_flat": True, "quiet": False}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        # print(info)
        return [entry["url"] for entry in info["entries"]]


def main():
    playlist_url = ()

    urls = get_urls_from_playlist(playlist_url)
    print(f"Processing playlist. There are {len(urls)} videos.")

    for url in urls:
        try:
            process_video(url)
        except Exception as e:
            print(f"Error processing {url}: {e}")


if __name__ == "__main__":
    main()

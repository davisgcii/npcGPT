import yt_dlp
import cv2
import os
import re


def download_video_and_captions(url):
    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt",
        "outtmpl": "video.%(ext)s",
        "format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        info = ydl.extract_info(url, download=False)
        captions = info.get("subtitles", {}).get("en", [])
        if captions:
            caption_url = captions[0]["url"]
            ydl.download([caption_url])


def parse_captions(file_path, max_time_difference=2.5):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    # Use regular expressions to extract timestamps and spoken text
    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> \d{2}:\d{2}:\d{2}\.\d{3}.*(?:\n|\r\n)([^\n<>]+)(?:\n|\r\n)"
    )

    matches = pattern.findall(content)
    timestamps = [match[0] for match in matches]
    texts = [match[1] for match in matches]

    def timestamp_to_seconds(timestamp):
        hours, minutes, seconds = map(float, timestamp.split(":"))
        return hours * 3600 + minutes * 60 + seconds

    grouped_captions = []
    grouped_timestamps = []

    if texts[0].strip() == "":
        current_group = ["a "]

    else:
        current_group = [texts[0]]

    current_timestamp = timestamps[0]

    for i in range(1, len(texts)):
        if texts[i].strip() == "":
            continue

        time_difference = timestamp_to_seconds(timestamps[i]) - timestamp_to_seconds(
            timestamps[i - 1]
        )
        # print(f"text: {texts[i]}")
        # print(f"current: {current_group}")

        if time_difference <= max_time_difference:
            if (
                current_group[-1].strip() != texts[i].strip()
            ):  # don't append if the same
                print(f"appending: {texts[i]}")
                current_group.append(texts[i])
        else:
            grouped_captions.append(" ".join(current_group))
            grouped_timestamps.append(current_timestamp)

            current_group = [texts[i]]
            current_timestamp = timestamps[i]

    # Add the last group of captions
    if current_group:
        grouped_captions.append(" ".join(current_group))
        grouped_timestamps.append(current_timestamp)

    return grouped_timestamps, grouped_captions


def extract_images(video_path, timestamps, grouped_captions):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    image_text_pairs = []
    for i, timestamp in enumerate(timestamps):
        # Convert timestamp to seconds
        hours, minutes, seconds = map(float, timestamp.split(":"))
        start_seconds = hours * 3600 + minutes * 60 + seconds
        frame_number = int(start_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            image_text_pairs.append((frame, grouped_captions[i]))
    cap.release()
    return image_text_pairs


def save_image_text_pairs(image_text_pairs, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, (image, text) in enumerate(image_text_pairs):
        image_path = os.path.join(output_dir, f"image_{i}.png")
        text_path = os.path.join(output_dir, f"text_{i}.txt")
        cv2.imwrite(image_path, image)
        with open(text_path, "w", encoding="utf-8") as file:
            file.write(text)


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=ImpgxJqwOXM"  # Replace VIDEO_ID with the actual video ID
    output_dir = "output"  # Set the output directory path here
    download_video_and_captions(url)
    timestamps, grouped_captions = parse_captions("video.en.vtt")
    image_text_pairs = extract_images("video.mp4", timestamps, grouped_captions)
    save_image_text_pairs(image_text_pairs, output_dir)

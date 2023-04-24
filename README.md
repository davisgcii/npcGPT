# npcGPT

## Setup
I'm using venv for the virtual environment. To set up, run:

>source venv/bin/activate
>venv> pip install -r requirements.txt

Install new packages using pip:

>venv> pip install {package_name}

And then update `requirements.txt`:

>venv> pip freeze > requirements.txt


## Data creation
`scripts/transcribe.py` can be used to create image-caption pairs using youtube videos.

To transcribe a video, copy the youtube video's url. Make sure to only include the base url and video id; don't include any query string parameters (e.g., `https://www.youtube.com/watch?v=vhII1qlcZ4E&t=7s` includes an unwanted `&t=7s` parameters).

Paste the video url into the `urls` list at the bottom of `transcribe.py`. Run the script:

>venv> python scripts/transcribe.py

This will download the video to the `temp_files` folder, use ffmpeg to create an audio file, create transcripts of the audio using OpenAI's Whisper (which runs locally), and then uses clustering to assign speakers to each caption based on vocal similarity.

From there, captions are grouped together based on spacing between lines of dialogue. I've messed around with this and have found that 4-6 seconds is an appropriate gap between groupings of captions.

Caption groupings are only retained if they contain multiple lines of dialgue. For example, if a character says "hello" and then nobody says anything else for more than `PAUSE_THRESHOLD` seconds, that caption grouping will only have one line of dialogue and it will be ignored.

Finally, the remaining caption groupings are used to create dialogue-image pairs, where each grouping of captions is associated with the image from the video at the same timestamp as the first line of dialogue in the caption grouping.

These pairs are saved in `data/{video_title}` as `caption_{timestamp}` and `image_{timestapm}`.

The `data` and `temp_files` folders are git-ignored to prevent long pushes/pulls. We can keep this dataset in google docs or a separate repo and periodically update it.
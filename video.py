import sys
import os
import ffmpeg

def create_video_from_images(image_folder, output_video, frame_rate=30):
    """
    Create a video from a series of images in a folder.

    Parameters:
    - image_folder: str, path to the folder containing images
    - output_video: str, path for the output video file
    - frame_rate: int, frames per second for the video (default is 30)
    """
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    input_pattern = os.path.join(image_folder, 'frame_%04d.png')

    try:
        (
            ffmpeg
            .input(input_pattern, framerate=frame_rate)
            .output(output_video)
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Video created successfully: {output_video}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode('utf8')}")

create_video_from_images('output_images', 'output_video.mp4')

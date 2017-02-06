from moviepy.editor import VideoFileClip
import imageio
from camera import Camera
from lane import Lane
from image_pipeline import detection_pipeline
import numpy as np

imageio.plugins.ffmpeg.download()
camera = Camera()
lane = Lane()

white_output = 'project_video_annotated.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(detection_pipeline)
white_clip.write_videofile(white_output, audio=False)

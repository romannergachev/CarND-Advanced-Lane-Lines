from moviepy.editor import VideoFileClip
import imageio
from camera import Camera
from image_pipeline import detection_pipeline

imageio.plugins.ffmpeg.download()
camera = Camera()

white_output = 'project_video_annotated.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(detection_pipeline)
white_clip.write_videofile(white_output, audio=False)

import logging
import os
import re
import subprocess


class ContentManager:
    def __init__(self):
        self._transcript_to_video_path = {}
    
    def _get_video_path(self, transcript_path):
        return self._transcript_to_video_path[transcript_path]

    def add_video(self, video_path):
        base_path = os.path.splitext(video_path)[0]
        transcript_path = base_path + '.srt'
        self._transcript_to_video_path[transcript_path] = video_path
        video_name = video_path.split('/')[-1]
        return transcript_path, video_name
    
    def get_video_name(self, transcript_path):
        return self._get_video_path(transcript_path).split('/')[-1]
    
    def get_frame(self, transcript_path, srttime):
        video_path = self._get_video_path(transcript_path)
        t = srttime.replace(',', '.')
        frame_path = os.path.splitext(video_path)[0] + t + '.png'
        frame_path = re.sub(r'[^a-zA-Z0-9._/-]', '_', frame_path)
        cmd = ['ffmpeg', '-ss', str(t), '-i', video_path, '-frames:v', '1', frame_path]
        logging.info('Running %s', cmd)
        subprocess.check_call(cmd)
        return frame_path
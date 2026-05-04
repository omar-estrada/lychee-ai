import logging
import os
import queue
import os
from threading import Thread
import time

import whisper

import utils


class Transcriber:

    def __init__(self):
        """
        Initializes the VideoProcessor with OpenAI's Whisper model.
        """
        self._model = whisper.load_model('base')
        self._video_queue = queue.Queue()
        self._processor_thread = Thread(target=self._transcriber_thread, daemon=True)
        self._keep_running = True
        self._processor_thread.start()
        self._csv = utils.Csv('video_metrics.csv', ['video', 'size_mb', 'transcription_secs'])

    def obtain_transcript(self, video_path, transcript_path, callback):
        """
        Return: the path to the SRT file with the timed transcript or None if the video
            can't be transcribed.
        """
        self._video_queue.put((video_path, transcript_path, callback))
    
    def _transcriber_thread(self):
        logging.info('Video processor thread started')
        while self._keep_running:
            logging.info('Waiting for a video to process...')
            video_path, transcript_path, callback = self._video_queue.get()
            logging.info('Processing %s...', video_path)
            self._get_transcript(video_path, transcript_path)
            callback(transcript_path)
            self._video_queue.task_done()
            
    def _get_transcript(self, video_path, transcript_path):
        logging.info('Transcribing %s...', video_path)
        start = time.time()
        try:
            result = self._model.transcribe(video_path)
            self._save_as_srt(result['segments'], transcript_path)
            self._csv.add_row([video_path, os.path.getsize(video_path) / 1024 / 1024, time.time() - start])
            logging.info('Transcribed video to %s', transcript_path)
            return transcript_path
        except Exception as e:
            logging.error('Error transcribing %s: %s', video_path, e)
            return None

    def _save_as_srt(self, segments, srt_path):
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, start=1):
                start = self._format_timestamp(segment['start'])
                end = self._format_timestamp(segment['end'])
                text = segment['text'].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    def _format_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

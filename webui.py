"""Entrypoint for the web app.

Search in the output for the Gradio URL to access the web UI remotely.
"""
import logging
import os
import sys

import gradio as gr

from content_manager import ContentManager
from rag  import Generator
from transcriber import Transcriber


def create_ui(generator, transcriber, content_manager: ContentManager):
    messages = []

    with gr.Blocks() as demo:
        gr.Markdown('# 🍒 Lychee AI')
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(label='Lecture Assistant', resizable=True)
                question = gr.Textbox(label='Question')
                submit = gr.Button('Ask')

            with gr.Column(scale=0.2):
                gr.Markdown('### 📂 Manage Videos')
                videos_files = gr.File(label='Upload Video', file_count='multiple')
        
        def on_transcript_generated(transcript_path):
            generator.add_transcript(transcript_path)
            video_name = content_manager.get_video_name(transcript_path)
            gr.Info(f'{video_name} has been added to the knowledge DB.', duration=3)
    
        def on_video_added(videos_paths: list[str]):
            logging.info('on_video_added: %s', videos_paths)
            for video_path in videos_paths:
                transcript_path, video_name = content_manager.add_video(video_path)
                gr.Info(f'{video_name} is being processed...', duration=3)
                transcriber.obtain_transcript(video_path, transcript_path, on_transcript_generated)
        
        def ask_question(question):
            (answer, transcript_path, srttime) = generator.invoke(question)
            frame_path = content_manager.get_frame(transcript_path, srttime)
            messages.append({'role': 'user', 'content': question})
            messages.append({'role': 'assistant', 'content': [answer, gr.Image(frame_path)]})
            return messages
        
        submit.click(fn=ask_question, inputs=question, outputs=chatbot)
        question.submit(fn=ask_question, inputs=question, outputs=chatbot)
        videos_files.upload(fn=on_video_added, inputs=videos_files)
    return demo


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    generator = Generator()
    demo = create_ui(generator, Transcriber(), ContentManager())
    demo.launch(share=True)


if __name__ == '__main__':
    main()

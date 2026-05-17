"""Script to run all the queries in transcripts_qa.csv.

The answers are output to rag_metrics.csv.
"""
import csv
import logging

from rag import Generator

TRANSCRIPTS_DIR = 'transcripts'


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    generator = Generator()
    questions = []

    logging.info('Reading CSV and loading transcripts...')
    with open('transcripts_qa.csv') as qafile:
        csv_reader = csv.reader(qafile)
        next(csv_reader)  # skip the header
        added_transcripts = set()
        for i, row in enumerate(csv_reader):
            logging.info('Row: %s', row)
            try:
                (question, _, video_name_no_ext, _) = row
                transcript_path = f'{TRANSCRIPTS_DIR}/{video_name_no_ext}.srt'
                if transcript_path not in added_transcripts:
                    generator.add_transcript(transcript_path)
                questions.append(question)
            except ValueError as e:
                logging.warn('Error reading row %d', i+2)
    
    logging.info('Running queries...')
    for question in questions:
        generator.invoke(question)
    
    logging.info('Done')
        

if __name__ == '__main__':
    main()
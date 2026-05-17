"""Loads an SRT file keeping timestamps (langchain_community.document_loaders.SRTLoader trims them).

Adapted from https://medium.com/@hari.haran849/rag-series-3-6871db57d8d0
"""

import logging
import pysrt
from langchain_community.document_loaders.base import BaseLoader
from langchain.schema import Document

class TimestampedSrtLoader(BaseLoader):
    """Class that loads SRT (SubRip subtitles file) keeping the timestamps."""
    def __init__(self, path: str):
        self._path = path
        self._nchunks = 10
        self._overlap = 1
    
    def load(self):
        logging.debug('Loading %s', self._path)
        entries = pysrt.open(self._path)
        docs = []
        for i in range(0, len(entries), self._nchunks):
            entry = entries[i]
            transcript = ' '.join([e.text for e in entries[i:i+self._nchunks+self._overlap]])
            filename = ''.join(self._path.split('/')[-1].split('.')[:-1])
            content = f'Content: {transcript}\nSource: {filename}\nTime: {entry.start}'
            # logging.debug('transcript: %s', transcript)
            metadata = {
                'source': self._path,
                'start_srttime': str(entry.start),
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

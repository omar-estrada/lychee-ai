import logging
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from timestamped_srt_loader import TimestampedSrtLoader
import utils

#import langchain
# langchain.debug = True

LLM_MODEL = 'llama3'
RAG_MODEL_CACHE_PATH = './rag_model_cache'
VIDEO_CONTEXT_PROMPT_TEMPLATE = """Use the following transcripts from lecture videos as the context to answer the question at the end.
Each transcript has the format

Content: <CONTENT>\\nSource: <SOURCE>\\nTime: <TIME>

where:
  <CONTENT>: the transcript
  <SOURCE>: the name of the source video
  <TIME>: the time in the video for the transcript

If you don't know the answer, just say that you don't know, don't try to make up an answer.
Indicate the name of the video and the time in the response.

{context}

QUESTION: {question}
"""
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=['context', 'question'],
    template=VIDEO_CONTEXT_PROMPT_TEMPLATE
)

def _get_transcript_and_time(result):
    source_docs = result['source_documents']
    if not source_docs:
        return ('', '00:00:00,000')
    metadata = source_docs[0].metadata
    return (metadata['source'], metadata['start_srttime'])


class Generator:
    def __init__(self):
        logging.info('Generator#__init__')
        rag_model = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder=RAG_MODEL_CACHE_PATH,
        )
        self._vector_db = Chroma(
            collection_name='class_recording_transcripts',
            embedding_function=rag_model,
            persist_directory="./chroma_db"
        )

        logging.info('Initializing the local LLM via Ollama..,')
        llm = Ollama(model=LLM_MODEL)

        logging.info('Creating the RAG chain...')
        self._rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=self._vector_db.as_retriever(),
            return_source_documents=True,  # gets the metadata
            chain_type_kwargs={
                'prompt': PROMPT_TEMPLATE
            }
        )
        self._csv = utils.Csv('rag_metrics.csv', ['query,response,secs'])
    
    # TODO: add a method that allows specifying a video rather
    #  than a transcript and that automatically extracts the
    #  transcript.
    def add_transcript(self, transcript_path):
        logging.info('Generator#add_transcript: transcript_path=%s', transcript_path)
        loader = TimestampedSrtLoader(transcript_path)
        docs = loader.load()
        self._vector_db.add_documents(docs)
    
    # def delete_transcript()
    # self._vector_db.delete(where={'source': selected_video})
    
    def invoke(self, query):
        logging.info('Generator#invoke: query=%s', query)
        start = time.time()
        result = self._rag_chain.invoke(query)
        answer = result['result']
        transcript_path, srttime = _get_transcript_and_time(result)
        self._csv.add_row([query, answer, time.time() - start])
        return (answer, transcript_path, srttime)

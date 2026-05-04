# Install dependencies
# pip install langchain==0.3.0 langchain-community chromadb sentence-transformers ollama pysrt gradio
# sudo apt install zstd
# Install the Ollama server:
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull llama3

import logging

import langchain
from langchain_community.document_loaders import DirectoryLoader, SRTLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import timestamped_srt_loader

langchain.debug = True

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
PROMPT_TEMPLATE2 = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

QUESTION: {question}
"""
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=['context', 'question'],
    # template=PROMPT_TEMPLATE2
    template=VIDEO_CONTEXT_PROMPT_TEMPLATE
)

def dbg(msg):
    print(msg)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    loader = DirectoryLoader('transcripts/', glob="Lecture 4.2*.srt", loader_cls=timestamped_srt_loader.TimestampedSrtLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    dbg('Creating embeddings and storing them in ChromaDB...')
    rag_model = HuggingFaceEmbeddings(
        model_name='BAAI/bge-large-en-v1.5',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder=RAG_MODEL_CACHE_PATH,
    )
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=rag_model,
        persist_directory="./chroma_db"
    )

    dbg('Initializing the local LLM via Ollama..,')
    llm = Ollama(model=LLM_MODEL)

    dbg('Creating the RAG chain...')
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        return_source_documents=True,  # gets the metadata
        chain_type_kwargs={
            'prompt': PROMPT_TEMPLATE
        }
    )

    dbg('Invoking query...')
    query = "What is the goal for lecture 4.2?"
    result = rag_chain.invoke(query)

    print(f"Answer: {result['result']}")

    # retrieved_docs = result['source_documents']
    # for i, doc in enumerate(retrieved_docs):
    #     print(f"\n--- Snippet {i+1} ---")
    #     print(f"Content: {doc.page_content[:200]}...")
    #     print(f"Metadata: {doc.metadata}")


if __name__ == '__main__':
    main()

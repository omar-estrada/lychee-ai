# Setup

1. Clone the repo:

```
git clone https://github.com/omar-estrada/lychee-ai.git
```

2. Set up a Python virtual env and install the necessary dependencies:

```shell
sudo apt-get update
sudo apt-get install -y zstd python3-venv ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install langchain==0.3.0 langchain-community chromadb sentence-transformers ollama pysrt gradio openai-whisper
```

2. Install Ollama:

```shell
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > /dev/null 2>&1 &
sleep 5  # Allow the server to start
ollama pull llama3
```

3. Start the web app:

```shell
cd lychee-ai/
python3 webui.py
```

The first run will take more time since the HuggingFace embeddings model will be downloaded (but they are cached, so subsequent runs are faster).

# Usage

1. Open http://127.0.0.1:7860/ in a browser.

2. Load a video using the right-hand panel.

3. Type a question in the box and click on `Ask`.
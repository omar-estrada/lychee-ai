
1. Set up a Python virtual env and install the necessary dependencies:

```shell
sudo apt-get update
sudo apt-get install -y zstd python3-venv ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install langchain==0.3.0 langchain-community chromadb sentence-transformers ollama pysrt gradio
```

2. Download and build whisper.cpp (a pure C++ version of OpenAI's Whisper for edge devices):

```shell
# From https://github.com/ggml-org/whisper.cpp#quick-start
sudo apt install g++ git make cmake
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
sh ./models/download-ggml-model.sh base.en
# make
cmake -B build
cmake --build build -j --config Release
```

3. Install Ollama:

```shell
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > /dev/null 2>&1 &
ollama pull llama3
```

4. Start the web app:

```shell
python3 webui.py <WHISPER.CPP GIT CHECKOUT DIR>
```

# Usage

1. Open http://127.0.0.1:7860/ in a browser.
# VidRAG

**Retrieval-Augmented Generation system for asking questions about YouTube videos.**

A Python application that applies Retrieval-Augmented Generation (RAG) to answer questions about YouTube videos.

## Project Motivation

Large video content can be difficult to search and navigate. This project explores how Retrieval-Augmented Generation (RAG) can be applied to YouTube transcripts to enable question-answering over video content.

The system extracts transcripts from YouTube videos, converts them into vector embeddings, and retrieves the most relevant sections when a user asks a question.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
## System Flow
The system processes YouTube videos using a Retrieval-Augmented Generation pipeline:

YouTube Video → Transcript Extraction → Text Chunking → Embeddings → FAISS Vector Store → Similarity Retrieval → LLM → Answer

This allows users to ask natural language questions about video content and receive context-aware answers.

##  Features

-  **YouTube Transcript Extraction**: Automatically extracts video transcripts
-  **Question answering using GPT models**: Generates contextual answers from retrieved transcript segments
-  **Vector Search**: FAISS-based similarity search for relevant content
-  **Smart Chunking**: Recursive text splitting with overlap for better context
-  **Fast Retrieval**: Optimized top-4 retrieval (k=4) for reduced hallucinations
-  **LangChain Integration**: Built with LangChain for modularity and extensibility
-  **Interactive Interface**: Command-line interface for easy interaction

## 🛠️ Tech Stack

- **Python 3.8+**
- **LangChain** - RAG framework and chain orchestration
- **OpenAI** – language models for question answering and embeddings
- **FAISS** - Vector similarity search and storage
- **youtube-transcript-api** - YouTube transcript extraction
- **tiktoken** - Text tokenization
- **python-dotenv** - Environment variable management

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sinhaaarushi/youtube-video-rag.git
   cd VidRaG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the application**
   ```bash
   python vidrag.py
   ```

##  Usage

### Basic Usage

```python
from vidrag import VidRAG

# Initialize VidRAG
vidrag = VidRAG(openai_api_key="your-api-key")

# Process a YouTube video
video_id = "Gfr50f6ZBvo"  # Just the ID, not the full URL
vidrag.process_video(video_id)

# Ask questions
answer = vidrag.ask_question("What is the main topic of this video?")
print(answer)
```

### Interactive Mode

```bash
python vidrag.py
```

The application will prompt you to:
1. Enter a YouTube video ID
2. Ask questions about the video
3. Get AI-powered answers based on the transcript

### Example Questions

- "What is the main topic discussed in this video?"
- "Can you summarize the key points?"
- "What are the technical details mentioned?"
- "Who are the speakers and what are their roles?"

### Key Components

- **VidRAG Class**: Main orchestrator for the RAG pipeline
- **Transcript Extraction**: Handles YouTube API integration
- **Text Processing**: Manages chunking and embedding generation
- **Vector Search**: FAISS-based similarity search
- **Question Answering**: LangChain-powered Q&A chain
  
## System Architecture

The system follows a typical RAG pipeline:

YouTube Video  
↓  
Transcript Extraction  
↓  
Text Chunking  
↓  
Embedding Generation  
↓  
FAISS Vector Database  
↓  
Similarity Retrieval  
↓  
GPT Response Generation

##  Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
DEFAULT_VIDEO_ID=Gfr50f6ZBvo
```

### Customization

```python
# Custom chunk size and overlap
vidrag = VidRAG(
    openai_api_key="your-key",
    chunk_size=1500,      # Larger chunks
    chunk_overlap=300     # More overlap
)

# Custom retrieval parameters
vidrag.retriever = vidrag.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}  # Retrieve more chunks
)
```

## 📁 Project Structure

```
VidRaG/
├── vidrag.py              # Main application
├── rag_using_langchain.py # Original implementation
├── rag_using_langchain.ipynb # Jupyter notebook
├── requirements.txt       # Python dependencies
├── env.example           # Environment template
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
   
## Use Cases
This system can be used for:
- Learning from educational YouTube videos
- Searching technical tutorials
- Video content summarization
- Knowledge extraction from long videos
- AI-powered video assistants
## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) for the RAG framework
- [OpenAI](https://openai.com) for the language models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) for transcript extraction

Maintained by **Aarushi Sinha**


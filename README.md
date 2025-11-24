# Multimodal RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot capable of processing and understanding both text and images from PDF documents. This project uses local LLMs (ollama) and embedding models (CLIP) to provide a secure and private conversational interface for your document. It also serves as a learning template to grasp the basic structure of a multimodal RAG chatbot.

## Features

- **Multimodal Understanding**: Processes both text and images from PDF documents.
- **Local Processing**: Uses local models (Llama 3.2 Vision via Ollama) for privacy and offline capability.
- **Semantic Search**: Utilizes CLIP embeddings to retrieve the most relevant text and images based on user queries.
- **Interactive UI**: Features a user-friendly chat interface built with Gradio.
- **PDF Support**: Automatically extracts and indexes content from PDF files.

## Technologies Used

- **LLM**: Llama 3.2 Vision (via Ollama)
- **Embeddings**: OpenAI CLIP (ViT-Base-Patch16)
- **Interface**: Gradio
- **PDF Processing**: PyMuPDF (fitz)
- **Frameworks**: PyTorch, Transformers

## Prerequisites

Before running the project, ensure you have the following installed:

1.  **Python 3.10+**
2.  **Ollama**: Download and install from [ollama.com](https://ollama.com).
3.  **Llama 3.2 Vision Model**:
    ```bash
    ollama pull llama3.2-vision
    ```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <https://github.com/3leopaul/multimodal_rag_chatbot>
    cd multimodal_rag_chatbot
    ```

2.  **Install dependencies**:
    ```
    pip install torch transformers gradio ollama pymupdf pillow
    ```

## Usage

### 1. Data Preparation
First, you need to process your documents to generate embeddings.

1.  Place your PDF files in the `data/raw_files/pdf/` directory.
2.  Open and run the `notebooks/data_prep.ipynb` notebook.
    - This will extract text and images.
    - Generate CLIP embeddings.
    - Save the processed data to `data/processed_files/`.

### 2. Running the Chatbot
Once the data is prepared, you can start the chat interface.

1.  Open and run the `notebooks/chatbot.ipynb` notebook.
2.  Run the cells to load the embeddings and launch the Gradio app.
3.  Click the local URL provided (usually `http://127.0.0.1:7860`) to interact with the chatbot.

## Project Structure

```
multimodal_rag_chatbot/
├── data/
│   ├── raw_files/          # Source PDF documents
│   └── processed_files/    # Extracted images, JSONs, and embeddings
├── notebooks/
│   ├── data_prep.ipynb     # Notebook for indexing documents
│   └── chatbot.ipynb       # Notebook for running the chat interface
├── src/
│   └── functions.py        # Core utility functions for processing and retrieval
├── README.md               # Project documentation
└── image.png               # Project screenshot
```
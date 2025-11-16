# AmbedkarGPT — RAG Q&A System

This project is a simple **command-line RAG (Retrieval-Augmented Generation)** system built for the Kalpit Pvt Ltd AI Intern assignment. It loads the provided **speech.txt**, indexes it into a vector store, retrieves relevant text, and answers questions using a **local LLM (Ollama + Mistral)**.

---

## Features

- Loads *speech.txt*
- Splits text into chunks
- Creates embeddings using **MiniLM-L6-v2**
- Stores embeddings in **ChromaDB**
- Retrieves relevant chunks for any question
- Generates answers using **Ollama Mistral**
- Simple command-line interface

---

## Installation

#### 1. Install Ollama
https://ollama.com

Pull the Mistral model:
```bash
ollama pull mistral
```

#### 2. Create environment
```bash
conda create -n ambedkar python=3.11 -y
conda activate ambedkar
```

3. Install dependencies
``` bash
pip install -r requirements.txt
```
4. Run the App
```bash
python main.py
```
Example questions:

What is the real remedy?

Why does Ambedkar criticize the shastras?


Type exit to quit.

#### How It Works

1. Load the text from speech.txt

2. Split it into chunks using a text splitter

3. Create embeddings using MiniLM-L6-v2

4. Store embeddings in ChromaDB

5. Retrieve relevant chunks based on user questions

6. Construct a prompt containing {context, question}

7. Generate a grounded answer using Mistral (via Ollama)

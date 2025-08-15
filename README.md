# 🔬 Scientific Literature RAG with Gemini

A Retrieval-Augmented Generation (RAG) system built to answer questions about scientific literature. This project uses papers from arXiv's Quantitative Biology (`q-bio`) section, processes them into a searchable vector database, and uses Google's Gemini Pro to generate research-backed answers with citations.

---

## ## Core Features

- **Domain-Specific Data:** Ingests and processes PDF research papers from a specific scientific domain.
- **Vectorized Database:** Uses **Sentence-Transformers** for high-quality embeddings and **ChromaDB** for efficient, persistent storage and retrieval.
- **Retrieval-Augmented Generation:** Retrieves the most relevant text chunks to a user's query and provides them as context to an LLM.
- **Evidence-Based Answers:** The Gemini model is instructed to answer questions **only** based on the provided context, ensuring answers are grounded in the source material.
- **Source Citations:** Automatically lists the titles of the source papers used to generate an answer.
- **Web Interface:** A simple and interactive UI built with **Gradio**.

---

## ## Tech Stack

- **Language:** Python 3.10+
- **LLM:** Google Gemini 1.5 Flash
- **Embedding Model:** `all-MiniLM-L6-v2` (from Hugging Face Sentence-Transformers)
- **Vector Database:** ChromaDB (Persistent, on-disk storage)
- **Data Source & Parsing:** `arxiv` library, PyMuPDF
- **Web Framework:** Gradio
- **Text Processing:** LangChain (`RecursiveCharacterTextSplitter`)

---

## ## Project Structure

The project is organized into three distinct scripts to maintain a clean workflow, separating data collection, processing, and application logic.

```
scientific-rag/
│
├── 📄 1_fetch_data.py           # Step 1: Downloads PDFs from arXiv
├── 🧠 2_process_and_embed.py      # Step 2: Chunks PDFs & builds the vector DB
├── 🚀 app.py                    # Step 3: Runs the Gradio web application
│
├── 📦 requirements.txt         # Project dependencies
├── 🔑 .env                      # For storing your API key (not committed)
├── 📂 papers/                  # (Created by script 1) Stores downloaded PDFs
├── 🗂️ chroma_db/                # (Created by script 2) Stores the persistent DB
└── 📜 README.md                  # This file
```

---

## ## Setup and Installation

Follow these steps to set up the project environment on your local machine.

### ### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/scientific-rag.git](https://github.com/your-username/scientific-rag.git)
cd scientific-rag
```

### ### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### ### 3. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### ### 4. Set Up Environment Variables

You'll need a Google Gemini API key to run the application.

1.  Create a file named `.env` in the root of the project directory.
2.  Get your API key from [Google AI Studio](https://aistudio.google.com/).
3.  Add your key to the `.env` file like this:

    ```env
    GOOGLE_API_KEY="AIzaSy...YourSecretKeyHere"
    ```

---

## ## How to Run the Application

Because the project is split into three parts, you must run the scripts in the correct order.

### ### Step 1: Fetch the Scientific Papers

This script connects to the `arxiv` API and downloads the 10 most recent papers from the Quantitative Biology (Genomics) category into the `papers/` directory.

```bash
python 1_fetch_data.py
```

### ### Step 2: Process the Papers and Build the Database

This script reads the downloaded PDFs, splits them into manageable chunks, generates embeddings for each chunk, and saves them to a persistent ChromaDB database in the `chroma_db/` directory.

```bash
python 2_process_and_embed.py
```

### ### Step 3: Launch the Web App

This script starts the Gradio web server. It connects to the database you just created and serves the user interface for you to ask questions.

```bash
python app.py
```

After running this command, open your web browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860`). You can now start querying your scientific documents!

https://huggingface.co/spaces/girishhardia/scientific-rag

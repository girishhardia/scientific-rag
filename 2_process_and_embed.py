import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

load_dotenv()

# --- 1. Document Loading and Chunking ---
def extract_text_from_pdfs(pdf_directory):
    """Extracts text from all PDFs in a directory and creates chunks."""
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            doc = fitz.open(filepath)
            
            # Extract metadata (title) from the PDF properties
            title = doc.metadata.get('title', 'No Title Found')
            
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            
            doc.close()

            # ⭐ Interview Gold: Chunking Strategy ⭐
            # A simple fixed-size chunking isn't ideal for research papers.
            # It can break sentences, formulas, and tables. A better approach
            # is "Recursive Character Splitting," which tries to split on logical
            # separators (paragraphs, sentences) first.
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, # Max characters per chunk
                chunk_overlap=200 # Overlap to maintain context between chunks
            )
            chunks = text_splitter.split_text(full_text)

            # Store each chunk with its source metadata
            for i, chunk_text in enumerate(chunks):
                documents.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": filename,
                        "page_chunk": i,
                        "title": title
                    }
                })
    return documents

# Process our papers
print("Processing PDFs and chunking text...")
all_chunks = extract_text_from_pdfs('papers')
print(f"Created {len(all_chunks)} chunks from the documents.")

# --- Code for Day 2 will continue in this file ---

# --- (Code from Day 1 is above this) ---

# --- 2. Embedding ---
print("Loading embedding model...")
# We use a powerful but compact model, great for starting out.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

print("Creating embeddings for all chunks...")
# This can take a moment depending on the number of chunks
embeddings = embedding_model.encode([chunk['text'] for chunk in all_chunks], show_progress_bar=True)

# --- (Code from previous steps is above this) ---

# --- 3. Vector Store Initialization and Population ---
print("Setting up vector database (ChromaDB)...")
client = chromadb.PersistentClient(path="./chroma_db") # For ephemeral in-memory storage

# If a collection with this name already exists, delete it to start fresh
if "scientific_papers" in [c.name for c in client.list_collections()]:
    client.delete_collection(name="scientific_papers")

# Create a new collection
# The embedding_function is automatically handled by ChromaDB if we use sentence-transformers
collection = client.create_collection(name="scientific_papers")

print("Adding documents to the collection...")
# ChromaDB needs unique IDs for each entry. We can create them simply.
ids = [f"chunk_{i}" for i in range(len(all_chunks))]

# Add the text chunks, their embeddings, and metadata to the collection
collection.add(
    embeddings=embeddings,
    documents=[chunk['text'] for chunk in all_chunks],
    metadatas=[chunk['metadata'] for chunk in all_chunks],
    ids=ids
)

print("\n✅ Setup complete! Your scientific RAG data is ready.")
print(f"Total documents in collection: {collection.count()}")
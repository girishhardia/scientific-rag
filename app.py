import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# --- NEW: Import Google's library ---
import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- NEW: Configure the Gemini API key ---
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    genai.configure(api_key=google_api_key)
except ValueError as e:
    print(e)
    # Handle the error gracefully, maybe exit or show a message in the UI
    # For now, we'll just print it.

# --- 1. Initialize Global Objects (do this once) ---
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to vector database...")
# Use PersistentClient to connect to the saved database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="scientific_papers") 

# --- NEW: Set up the Gemini model ---
# We'll use gemini-1.5-flash, which is fast, capable, and has a large context window.
generation_config = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                              generation_config=generation_config)


# --- 2. The Core RAG Function ---
def get_research_backed_answer(query):
    """
    This function takes a user query, retrieves relevant documents,
    and generates an answer using the Gemini LLM.
    """
    if not query:
        return "Please ask a question."
        
    # 1. Embed the user's query
    query_embedding = embedding_model.encode([query])[0].tolist()

    # 2. Retrieve relevant chunks from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  # Retrieve top 5 most relevant chunks
    )
    
    retrieved_docs = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]

    # 3. Format the context for the LLM
    context = "\n\n---\n\n".join(retrieved_docs)
    
    # This prompt template is crucial. It instructs the LLM on how to behave.
    prompt_template = f"""
    You are a specialized scientific assistant for biology.
    Your task is to answer the user's question based *only* on the provided scientific research excerpts.
    Do not use any external knowledge or information you were trained on.
    
    Here is the relevant context from research papers:
    ---
    {context}
    ---

    Based on the context provided, please answer the following question:
    Question: {query}
    
    After providing the answer, list the sources you used in a "Citations" section.
    For each piece of information, cite the title of the paper it came from.
    """

    # --- 4. NEW: Generate the answer using Gemini ---
    try:
        # Pass the full prompt to the model
        response = model.generate_content(prompt_template)
        answer = response.text
        
        # 5. Add citations from metadata
        citations = "\n\n**Citations:**\n"
        unique_titles = set(meta['title'] for meta in retrieved_metadatas if meta.get('title'))
        for title in unique_titles:
            citations += f"- {title}\n"
        
        return answer + citations

    except Exception as e:
        # Specific error handling for Gemini can be added here
        return f"An error occurred while generating the answer: {e}"


# --- 3. Gradio User Interface ---
iface = gr.Interface(
    fn=get_research_backed_answer,
    inputs=gr.Textbox(lines=2, placeholder="e.g., What are the latest findings in CRISPR gene editing?"),
    outputs=gr.Markdown(label="Answer with Citations"),
    title="🔬 Scientific RAG powered by Gemini",
    description="Ask a question about quantitative biology. The AI will answer based on recent arXiv papers and provide citations.",
    theme="soft"
)

# Launch the app
if __name__ == "__main__":
    print("Launching Gradio App...")
    iface.launch()
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# --- NEW IMPORTS FOR AUDIO,Video ---
from llama_index.readers.file import VideoAudioReader
from pathlib import Path

# --- SETUP ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
print("API Key loaded.")

# --- 1. LOAD YOUR DATA ---
print("Loading documents... This may take a while for audio files.")
documents = []

# --- 1a. Load Text-Based Files (PDF, DOCX, TXT, etc.) ---
try:
    text_docs = SimpleDirectoryReader("./data").load_data()
    documents.extend(text_docs)
    print(f"Loaded {len(text_docs)} text-based documents.")
except Exception as e:
    print(f"Error loading text documents: {e}")

# --- 1b. Load Local Audio Files (MP3) ---
try:
    # Initialize the audio transcriber (uses Whisper)
    audio_loader = VideoAudioReader()
    
    # Use Pathlib to find all .mp3 files in the data directory
    data_dir = Path("./data")
    audio_files = list(data_dir.rglob("*.mp3"))
    
    if audio_files:
        print(f"Found {len(audio_files)} audio file(s) to transcribe...")
        for f in audio_files:
            print(f"Transcribing {f.name}...")
            # Load_data returns a list, so we use extend
            audio_docs = audio_loader.load_data(file=f)
            documents.extend(audio_docs)
        print("Audio transcription complete.")

except Exception as e:
    print(f"Error loading audio files: {e}")
    print("This often fails if 'ffmpeg' is not installed on your system.")
    print("You can install ffmpeg from: https://ffmpeg.org/download.html")

if not documents:
    print("No documents were loaded. Exiting.")
    exit()

print(f"\nTotal documents loaded: {len(documents)}")


# --- 2. CONFIGURE THE LLM ---
llm = GoogleGenAI(model_name="models/gemini-pro", api_key=api_key)
print("Gemini LLM configured.")


# --- 2b. CONFIGURE THE EMBEDDING MODEL ---
print("Configuring local embedding model (BAAI/bge-small-en-v1.5)...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# --- 2c. SET THE GLOBALS ---
Settings.llm = llm
Settings.embed_model = embed_model
print("Global LLM and embedding model set.")

# --- 3. CREATE THE INDEX (THE KNOWLEDGE BASE) ---
print("Creating index... (This may take time depending on document size)")
index = VectorStoreIndex.from_documents(documents)
print("Index created successfully.")


# --- 4. CREATE THE QUERY ENGINE ---
query_engine = index.as_query_engine()
print("🚀 Query engine is ready. Ask away!")
print("-" * 50)


# --- 5. ASK QUESTIONS! ---
while True:
    user_query = input("Ask a question about your documents (or type 'exit'): ")
    if user_query.lower() == 'exit':
        break
    
    response = query_engine.query(user_query)
    
    print("\nAnswer:")
    print(response)
    print("-" * 50)
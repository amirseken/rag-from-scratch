"""Shared settings. Tweak these as you experiment."""
 
INDEX_DIR = "./index"             # where the vectors + chunk texts are saved
 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small (~90 MB), runs on CPU
EMBED_CACHE_DIR = None            # set e.g. "D:/models" to store the model on another drive
 
CHUNK_WORDS = 200                 # words per chunk
CHUNK_OVERLAP = 40                # words shared between neighbouring chunks
 
TOP_K = 5                         # how many chunks to retrieve per question
 
# Generation model (Gemini). Model names change over time: run
#   python list_models.py
# to see which ones your key can use, and paste a name here.
LLM_MODEL = "gemini-3.5-flash-lite"
MAX_TOKENS = 2048                 # some Gemini models spend part of this on "thinking"
 
 
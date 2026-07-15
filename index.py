import gc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer
import openpyxl
import pickle

# PDF extraction
bible_merge = PdfReader("MERGE_BIBLE_DOCTRINE.pdf")
text = ""
for page in bible_merge.pages:
    text += page.extract_text()

# Improved chunking — lower filter + sliding window
pg = text.split("\n")
lines = [p.strip() for p in pg if len(p.strip()) > 30]

chunks = []
for i in range(0, len(lines), 2):
    chunk = " ".join(lines[i:i+3])
    if len(chunk) > 50:
        chunks.append(chunk)
# Remove question list chunks
chunks = [c for c in chunks if "quels sont les différents" not in c.lower()
          or len(c) > 300]
print(f"Total text length: {len(text)}")
print(f"Total lines before filter: {len(pg)}")
print(f"Total lines after filter: {len(lines)}")
print(f"Sample chunk: {chunks[1][:200]}")
print(f"Total chunks after sliding window: {len(chunks)}")

# Excel extraction
def extract_excel(path):
    file = openpyxl.load_workbook(path, data_only=True)
    t = ""
    for sheet in file.worksheets:
        for row in sheet.iter_rows(values_only=True):
            for cell in row:
                if cell is not None:
                    t += str(cell) + " | "
    return t

extract_file1 = extract_excel("Tabernacle apocalypse.xlsx")
extract_file2 = extract_excel("Voyages de Paul2.pptx.xlsx")

chunk_1 = [i for i in extract_file1.split(" | ") if len(i) > 30]
chunk_2 = [e for e in extract_file2.split(" | ") if len(e) > 30]
chunks.extend(chunk_1)
chunks.extend(chunk_2)
print(f"Total chunks including Excel: {len(chunks)}")

# BM25 and IDs
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)
ids = [f"id_{i}" for i in range(len(chunks))]

# ChromaDB
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
db_client = chromadb.PersistentClient(path="./chroma_doctrine_db")
collection = db_client.get_or_create_collection(name="bible_doctrine_v3")

if collection.count() == 0:
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_embeddings = model.encode(
            batch_chunks,
            show_progress_bar=True
        ).tolist()
        collection.add(
            embeddings=batch_embeddings,
            ids=batch_ids,
            documents=batch_chunks
        )
        gc.collect()
        print(f"Batch {i//batch_size + 1} done")

# Save to disk
with open("chunks_doctrine.pkl", "wb") as f:
    pickle.dump(chunks,f)

with open("bm25_doctrine.pkl", "wb") as f:
    pickle.dump(bm25,f)

print("Indexing complete.")
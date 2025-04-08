# %%

import uuid
import pickle
import os
import multiprocessing
import concurrent.futures
from pathlib import Path
import re
import requests
from tqdm import tqdm
import numpy as np
from supabase import create_client, Client
from langchain_core.documents import Document
# %%

# Configuration
SUPABASE_URL = "http://127.0.0.1:54321"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
TEI_URL = "http://localhost:8011/embed"

# Chunking configuration
MAX_CHUNK_SIZE = 8000  # Approximate max token size (in characters)
CHUNK_OVERLAP = 200  # Characters of overlap between chunks

# Calculate number of cores to use (half of available)
NUM_CORES = max(1, multiprocessing.cpu_count() // 2)
print(f"Using {NUM_CORES} cores for parallel processing")

# %%


# Function to embed a single text
def embed_single_text(text):
    """Embed a single text"""
    try:
        response = requests.post(
            TEI_URL,
            json={"inputs": text},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise ValueError(
                f"Embedding request failed with status {response.status_code}: {response.text}"
            )

        embedding_nested = response.json()

        # Fix for nested embeddings - flatten if needed
        if (
            isinstance(embedding_nested, list)
            and len(embedding_nested) > 0
            and isinstance(embedding_nested[0], list)
        ):
            embedding = embedding_nested[0]  # Take the first element of the outer array
        else:
            embedding = embedding_nested  # Just in case it's already flat

        return embedding
    except Exception as e:
        print(f"Error embedding text: {e}")
        # Return a zero vector as fallback
        return np.zeros(768).tolist()


def find_best_split_point(text, max_length):
    """Find the best point to split text, prioritizing page markers"""
    if len(text) <= max_length:
        return len(text)  # Return the full text length if it fits

    # Look for page markers (e.g., "Page 46") in the text up to max_length
    search_text = text[:max_length]
    matches = list(re.finditer(r"\nPage \d+", search_text))

    if matches:
        # Split at the last page marker within the max length
        return matches[-1].start() + 1  # +1 to keep the newline with the previous chunk

    # If no page markers found, only split at max_length as a last resort
    return max_length


def chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Process text into chunks with the following priority:
    1. Keep entire document as one chunk if it fits within max_chunk_size
    2. If document is too large, split only at page markers
    3. Only resort to character-level splitting if no page markers are available
    """
    # If the entire text fits within the max chunk size, return it as a single chunk
    if len(text) <= max_chunk_size:
        return [text.strip()]

    # Otherwise, we need to split the document
    chunks = []
    start = 0

    while start < len(text):
        # Find the end point for this chunk
        end = find_best_split_point(text[start:], max_chunk_size) + start

        # Add the chunk
        if start < end:
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

        # If we split at a page marker, start the next chunk at that marker
        # If we split at max length, just move to the next position
        start = end

    return chunks


def process_markdown_file(file_path):
    """Process a single markdown file into chunks"""
    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Chunk the content
        chunks = chunk_text(content)

        # Create documents from chunks
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": str(file_path),
                "filename": file_path.stem,
                "file_path": str(file_path),
                "chunk_id": i,
                "total_chunks": len(chunks),
            }

            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)

        return documents
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def load_markdown_files(directory, cache_file="processed_documents_cache.pkl"):
    """Load and process all markdown files from a directory"""
    # Check for cache file
    if os.path.exists(cache_file):
        os.remove(cache_file)  # Remove existing cache

    # Find all markdown files
    md_files = list(Path(directory).glob("**/*.md"))
    print(f"Found {len(md_files)} markdown files in {directory}")

    # Process files one by one with progress bar
    all_documents = []
    for file_path in tqdm(md_files, desc="Processing markdown files"):
        documents = process_markdown_file(file_path)
        all_documents.extend(documents)

    # Save to cache
    print(f"Saving {len(all_documents)} document chunks to cache")
    with open(cache_file, "wb") as f:
        pickle.dump(all_documents, f)

    return all_documents


def insert_batch(batch):
    """Insert a batch of documents with embeddings into Supabase"""
    batch_index, docs = batch

    try:
        # Initialize a new client for each process to avoid sharing connections
        local_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        # Prepare data for insertion
        rows = []
        for doc in docs:
            try:
                # Get embedding for the current document
                embedding = embed_single_text(doc.page_content)

                # Create row with document and embedding
                rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": embedding,
                    }
                )
            except Exception as e:
                print(f"Error processing document in batch {batch_index}: {e}")

        if not rows:
            print(f"Batch {batch_index}: No valid documents to insert")
            return False, batch_index

        # Insert into Supabase
        response = local_supabase.table("jfk_documents").insert(rows).execute()

        # Verify the response
        if hasattr(response, "data") and response.data:
            print(f"Batch {batch_index}: Inserted {len(rows)} documents")
            return True, batch_index
        else:
            print(f"Batch {batch_index}: Insertion returned no data")
            print(f"Response: {response}")
            return False, batch_index

    except Exception as e:
        print(f"Error processing batch {batch_index}: {e}")
        return False, batch_index


def ingest_documents_to_supabase(documents, batch_size=5):
    """Ingest documents into Supabase with parallel processing using ThreadPoolExecutor."""
    print(f"Preparing to ingest {len(documents)} documents into jfk_documents table")

    # Split documents into batches
    batches = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batches.append((i // batch_size, batch))

    print(f"Split documents into {len(batches)} batches of size {batch_size}")

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid the daemonic process issue
    # ThreadPoolExecutor is better for I/O bound tasks like API calls and database operations
    successful_batches = 0
    failed_batches = 0

    # Process in small chunks to avoid memory issues
    chunk_size = 100  # Process 100 batches at a time (reduced from 1000)

    for chunk_start in range(0, len(batches), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(batches))
        current_chunk = batches[chunk_start:chunk_end]

        print(
            f"Processing chunk {chunk_start//chunk_size + 1}/{(len(batches)-1)//chunk_size + 1} "
            f"(batches {chunk_start}-{chunk_end-1})"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
            futures = [executor.submit(insert_batch, batch) for batch in current_chunk]

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(current_chunk),
                desc=f"Ingesting batches {chunk_start}-{chunk_end-1}",
            ):
                try:
                    success, batch_index = future.result()
                    if success:
                        successful_batches += 1
                    else:
                        failed_batches += 1
                except Exception as e:
                    print(f"Error in future completion: {e}")
                    failed_batches += 1

    print(
        f"Ingestion complete. Successful batches: {successful_batches}, Failed batches: {failed_batches}"
    )


def verify_documents_in_db():
    """Verify that documents were actually inserted in the database."""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        # Check if table exists and count documents
        result = (
            supabase.table("jfk_documents").select("count(*)", count="exact").execute()
        )

        if hasattr(result, "count") and result.count is not None:
            print(f"Found {result.count} documents in the jfk_documents table")
        else:
            # Alternative approach to get count
            count_data = result.data
            if count_data and len(count_data) > 0:
                count = count_data[0].get("count", "unknown")
                print(f"Found {count} documents in the jfk_documents table")
            else:
                print("Could not get document count")
                print(f"Response: {result}")

        # Get a sample document to verify structure
        sample = supabase.table("jfk_documents").select("*").limit(1).execute()
        if hasattr(sample, "data") and sample.data:
            print("Sample document:")
            print(f"ID: {sample.data[0].get('id')}")
            print(
                f"Content (first 100 chars): {sample.data[0].get('content', '')[:100]}..."
            )
            print(f"Content length: {len(sample.data[0].get('content', ''))}")

            # Check if embedding exists and has the right dimensions
            embedding = sample.data[0].get("embedding")
            if embedding:
                print(f"Embedding exists with dimension: {len(embedding)}")
            else:
                print("Warning: No embedding found in sample document")
        else:
            print("No documents found in the table")

    except Exception as e:
        print(f"Error verifying documents: {e}")


def main():
    # Path to your JFK markdown files
    jfk_files_dir = "./jfk_files_md"

    # Load documents (sequential processing for reliability)
    documents = load_markdown_files(jfk_files_dir)
    print(f"Loaded {len(documents)} document chunks")

    # Ingest documents into Supabase
    print("Ingesting documents into Supabase...")
    ingest_documents_to_supabase(documents, batch_size=5)

    # Verify documents in the database
    print("\nVerifying documents in the database...")
    verify_documents_in_db()

    print("All processing complete!")


if __name__ == "__main__":
    main()

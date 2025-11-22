#!/usr/bin/env python3
"""
FDNY Books Indexer

Creates a vector store index from the consolidated books for RAG-based retrieval.
"""

import os
import json
import pickle
from pathlib import Path
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# Configuration
CONSOLIDATED_DIR = Path("/home/user/fdny_tutor_2025/data/consolidated_books")
METADATA_FILE = Path("/home/user/fdny_tutor_2025/data/books_metadata.json")
INDEX_DIR = Path("/home/user/fdny_tutor_2025/data/vector_store")

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_consolidated_books() -> list[Document]:
    """Load all consolidated books and create LangChain documents with metadata."""
    documents = []

    # Load metadata
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Loading {metadata['total_categories']} consolidated books...")

    for category in tqdm(metadata["categories"], desc="Loading books"):
        book_path = CONSOLIDATED_DIR / category["filename"]

        if not book_path.exists():
            print(f"  Warning: {book_path} not found")
            continue

        content = book_path.read_text(encoding="utf-8")

        # Create document with rich metadata
        doc = Document(
            page_content=content,
            metadata={
                "source": category["filename"],
                "category": category["category"],
                "document_count": category["document_count"],
                "total_pages": category["total_pages"],
            }
        )
        documents.append(doc)

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n---\n", "\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )

    all_chunks = []
    for doc in tqdm(documents, desc="Chunking documents"):
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    return all_chunks


def create_vector_store(chunks: list[Document]) -> FAISS:
    """Create FAISS vector store from document chunks."""
    print(f"\nCreating embeddings for {len(chunks)} chunks...")
    print("(This may take a few minutes depending on your API rate limits)")

    embeddings = OpenAIEmbeddings()

    # Process in batches to avoid rate limits
    batch_size = 100
    vectorstore = None

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i:i + batch_size]

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_store)

    return vectorstore


def save_vector_store(vectorstore: FAISS):
    """Save the vector store to disk."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"\nVector store saved to: {INDEX_DIR}")


def main():
    """Main indexing process."""
    print("=" * 60)
    print("FDNY Books Indexer")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Load books
    documents = load_consolidated_books()
    print(f"\nLoaded {len(documents)} consolidated books")

    # Chunk documents
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Create vector store
    vectorstore = create_vector_store(chunks)

    # Save to disk
    save_vector_store(vectorstore)

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

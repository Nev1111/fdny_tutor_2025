#!/usr/bin/env python3
"""
FDNY Books Consolidator

Extracts text from all PDFs in each book category and consolidates them
into single markdown files with chapter metadata preserved.
"""

import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm
import json


# Configuration
SOURCE_DIR = Path("/home/user/Downloads/fdny books 2025 3rd qtr")
OUTPUT_DIR = Path("/home/user/fdny_tutor_2025/data/consolidated_books")
METADATA_FILE = Path("/home/user/fdny_tutor_2025/data/books_metadata.json")


def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """Extract text from a PDF file and return (text, page_count)."""
    try:
        doc = fitz.open(str(pdf_path))
        all_text = ""
        for page in doc:
            all_text += page.get_text()
        page_count = len(doc)
        doc.close()
        return all_text.strip(), page_count
    except Exception as e:
        print(f"  Error extracting {pdf_path.name}: {e}")
        return "", 0


def get_chapter_name_from_filename(filename: str) -> str:
    """Extract a clean chapter name from a PDF filename."""
    # Remove .pdf extension
    name = filename.replace(".pdf", "").replace(".PDF", "")
    # Replace underscores with spaces
    name = name.replace("_", " ")
    return name


def process_directory_recursive(dir_path: Path, depth: int = 0) -> list[dict]:
    """
    Recursively process a directory and all its subdirectories.
    Returns a list of document entries with metadata.
    """
    documents = []

    # Get all items in directory, sorted naturally
    try:
        items = sorted(dir_path.iterdir(), key=lambda x: natural_sort_key(x.name))
    except PermissionError:
        return documents

    for item in items:
        if item.is_file() and item.suffix.lower() == ".pdf":
            text, page_count = extract_text_from_pdf(item)
            if text:
                documents.append({
                    "filename": item.name,
                    "path": str(item.relative_to(SOURCE_DIR)),
                    "chapter_name": get_chapter_name_from_filename(item.name),
                    "text": text,
                    "page_count": page_count,
                    "depth": depth
                })
        elif item.is_dir():
            # Recursively process subdirectories
            subdocs = process_directory_recursive(item, depth + 1)
            documents.extend(subdocs)

    return documents


def consolidate_category(category_path: Path) -> dict:
    """
    Consolidate all PDFs in a category into a single markdown file.
    Returns metadata about the consolidated book.
    """
    category_name = category_path.name
    print(f"\nProcessing: {category_name}")

    # Process all PDFs recursively
    documents = process_directory_recursive(category_path)

    if not documents:
        print(f"  No PDFs found in {category_name}")
        return None

    print(f"  Found {len(documents)} PDFs")

    # Build consolidated markdown
    md_content = f"# {category_name}\n\n"
    md_content += f"*Consolidated from {len(documents)} source documents*\n\n"
    md_content += "---\n\n"

    total_pages = 0
    chapters = []

    for doc in tqdm(documents, desc="  Consolidating", leave=False):
        # Add chapter header
        header_level = "#" * min(doc["depth"] + 2, 6)  # h2 to h6
        md_content += f"{header_level} {doc['chapter_name']}\n\n"
        md_content += f"*Source: {doc['path']} ({doc['page_count']} pages)*\n\n"
        md_content += doc["text"]
        md_content += "\n\n---\n\n"

        total_pages += doc["page_count"]
        chapters.append({
            "name": doc["chapter_name"],
            "source": doc["path"],
            "pages": doc["page_count"]
        })

    # Create safe filename
    safe_name = re.sub(r'[^\w\s-]', '', category_name)
    safe_name = re.sub(r'[-\s]+', '_', safe_name).strip('_')
    output_file = OUTPUT_DIR / f"{safe_name}.md"

    # Write consolidated file
    output_file.write_text(md_content, encoding="utf-8")
    print(f"  Saved: {output_file.name} ({len(md_content):,} chars)")

    return {
        "category": category_name,
        "filename": output_file.name,
        "document_count": len(documents),
        "total_pages": total_pages,
        "chapters": chapters,
        "size_chars": len(md_content)
    }


def main():
    """Main consolidation process."""
    print("=" * 60)
    print("FDNY Books Consolidator")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all category directories (those starting with numbers)
    categories = sorted([
        d for d in SOURCE_DIR.iterdir()
        if d.is_dir() and d.name[0:2].replace("_", "").isdigit()
    ], key=lambda x: natural_sort_key(x.name))

    print(f"\nFound {len(categories)} book categories to process")

    # Process each category
    all_metadata = []
    for category_path in categories:
        metadata = consolidate_category(category_path)
        if metadata:
            all_metadata.append(metadata)

    # Save metadata
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total_categories": len(all_metadata),
            "total_documents": sum(m["document_count"] for m in all_metadata),
            "total_pages": sum(m["total_pages"] for m in all_metadata),
            "categories": all_metadata
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("CONSOLIDATION COMPLETE")
    print("=" * 60)
    print(f"Categories processed: {len(all_metadata)}")
    print(f"Total PDFs consolidated: {sum(m['document_count'] for m in all_metadata)}")
    print(f"Total pages: {sum(m['total_pages'] for m in all_metadata):,}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata file: {METADATA_FILE}")


if __name__ == "__main__":
    main()

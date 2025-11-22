# FDNY AI Tutor 2025

An intelligent tutoring application for FDNY training materials, inspired by Google NotebookLM.

## Features

- **Chat**: Ask questions about any FDNY procedure or protocol
- **Quiz**: Generate practice quizzes on any topic
- **Summary**: Get comprehensive summaries of training topics
- **Study Notes**: Create organized study notes
- **Flashcards**: Generate flashcards for quick review

## Training Materials

This application contains **22 book categories** with over **1,030 chapters** covering:

- All Unit Circulars (AUCs)
- BISP Manual (Building Inspection Safety Program)
- CFR-D Manual (First Responders - Defibrillation)
- Communications Manual
- FFP Volumes 1-4 (Firefighting Procedures)
- Hazardous Materials
- ICS Manual (Incident Command System)
- PAIDs (Personnel Administrative Information Directives)
- Safety Bulletins
- Training Bulletins
- And more...

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-key-here'
```

3. Run the indexing script (first time only):
```bash
python src/index_books.py
```

4. Launch the app:
```bash
streamlit run app.py
```

## Project Structure

```
fdny_tutor_2025/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── src/
│   ├── consolidate_books.py  # PDF extraction & consolidation
│   └── index_books.py        # Vector store indexing
├── data/
│   ├── consolidated_books/   # Merged book files (22 categories)
│   ├── vector_store/         # FAISS index
│   └── books_metadata.json   # Book metadata
└── config/
```

## Technology Stack

- **Streamlit**: Web UI framework
- **LangChain**: LLM orchestration
- **OpenAI GPT-4**: Language model
- **FAISS**: Vector similarity search
- **PyMuPDF**: PDF text extraction

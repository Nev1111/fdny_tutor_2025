#!/usr/bin/env python3
"""
FDNY AI Tutor 2025

A NotebookLM-style learning application for FDNY training materials.
Features: Chat, Quiz, Summary, Study Notes, Flashcards
"""

import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Load environment
load_dotenv()

# Configuration
INDEX_DIR = Path(__file__).parent / "data" / "vector_store"
METADATA_FILE = Path(__file__).parent / "data" / "books_metadata.json"


# --- Cached Resources ---
@st.cache_resource
def load_vector_store():
    """Load the FAISS vector store."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def get_llm():
    """Get the ChatOpenAI instance."""
    return ChatOpenAI(model_name="gpt-4", temperature=0.3)


@st.cache_data
def load_metadata():
    """Load books metadata."""
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Prompt Templates ---
QA_PROMPT = PromptTemplate(
    template="""You are an expert FDNY training instructor. Use the following context from official FDNY training materials to answer the question.
Be specific, accurate, and cite relevant procedures when applicable.

Context:
{context}

Question: {question}

Answer: Provide a clear, detailed answer based on the FDNY training materials above. If the information isn't in the context, say so.""",
    input_variables=["context", "question"]
)

SUMMARY_PROMPT = """You are an expert FDNY training instructor. Based on the following content from FDNY training materials, provide a comprehensive summary.

Content:
{context}

Please provide:
1. **Key Topics Covered**: List the main subjects
2. **Critical Procedures**: Highlight important procedures and protocols
3. **Safety Considerations**: Note any safety-related information
4. **Key Definitions**: Important terms and their meanings

Summary:"""

QUIZ_PROMPT = """You are an FDNY training instructor creating a quiz. Based on the following content, generate {num_questions} multiple-choice questions to test knowledge.

Content:
{context}

For each question, provide:
1. The question
2. Four options (A, B, C, D)
3. The correct answer
4. A brief explanation

Format each question as:
QUESTION: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
CORRECT: [letter]
EXPLANATION: [why this is correct]

---

Generate the quiz:"""

STUDY_NOTES_PROMPT = """You are an expert FDNY training instructor creating study notes. Based on the following content, create organized study notes.

Content:
{context}

Create study notes with:
1. **Main Concepts**: Bullet points of key ideas
2. **Procedures Step-by-Step**: Numbered steps for important procedures
3. **Remember This**: Critical points to memorize
4. **Common Mistakes to Avoid**: What NOT to do
5. **Practice Scenarios**: 2-3 scenarios to think through

Study Notes:"""

FLASHCARD_PROMPT = """You are an FDNY training instructor. Based on the following content, create {num_cards} flashcards for studying.

Content:
{context}

Format each flashcard as:
FRONT: [question or term]
BACK: [answer or definition]

---

Create the flashcards:"""


# --- Core Functions ---
def get_qa_chain(vectorstore, llm):
    """Create a RetrievalQA chain."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )


def chat_with_tutor(query: str, vectorstore, llm) -> tuple[str, list]:
    """Get an answer from the tutor."""
    qa_chain = get_qa_chain(vectorstore, llm)
    result = qa_chain.invoke({"query": query})
    sources = list(set([doc.metadata.get("category", "Unknown") for doc in result["source_documents"]]))
    return result["result"], sources


def generate_summary(topic: str, vectorstore, llm) -> str:
    """Generate a summary for a topic."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content[:2000] for doc in docs])

    prompt = SUMMARY_PROMPT.format(context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def generate_quiz(topic: str, num_questions: int, vectorstore, llm) -> str:
    """Generate a quiz on a topic."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content[:2000] for doc in docs])

    prompt = QUIZ_PROMPT.format(context=context, num_questions=num_questions)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def generate_study_notes(topic: str, vectorstore, llm) -> str:
    """Generate study notes for a topic."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content[:2000] for doc in docs])

    prompt = STUDY_NOTES_PROMPT.format(context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def generate_flashcards(topic: str, num_cards: int, vectorstore, llm) -> str:
    """Generate flashcards for a topic."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content[:2000] for doc in docs])

    prompt = FLASHCARD_PROMPT.format(context=context, num_cards=num_cards)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# --- UI Components ---
def render_sidebar():
    """Render the sidebar with book categories."""
    st.sidebar.title("FDNY Training Library")

    try:
        metadata = load_metadata()
        st.sidebar.markdown(f"**{metadata['total_categories']} Books** | **{metadata['total_documents']:,} Chapters** | **{metadata['total_pages']:,} Pages**")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Book Categories")

        for cat in metadata["categories"]:
            with st.sidebar.expander(cat["category"][:40] + "..." if len(cat["category"]) > 40 else cat["category"]):
                st.write(f"Documents: {cat['document_count']}")
                st.write(f"Pages: {cat['total_pages']}")
    except Exception as e:
        st.sidebar.error(f"Could not load metadata: {e}")


def render_chat_tab(vectorstore, llm):
    """Render the Chat tab."""
    st.header("Chat with FDNY Tutor")
    st.markdown("Ask any question about FDNY procedures, protocols, and training materials.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption(f"Sources: {', '.join(msg['sources'])}")

    # Chat input
    if prompt := st.chat_input("What would you like to learn about?"):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Searching training materials..."):
                response, sources = chat_with_tutor(prompt, vectorstore, llm)
                st.markdown(response)
                if sources:
                    st.caption(f"Sources: {', '.join(sources)}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })


def render_quiz_tab(vectorstore, llm):
    """Render the Quiz tab."""
    st.header("Practice Quiz")
    st.markdown("Test your knowledge on any FDNY topic.")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Enter a topic to quiz yourself on:", placeholder="e.g., High-rise firefighting procedures")
    with col2:
        num_questions = st.selectbox("Questions:", [3, 5, 10], index=1)

    if st.button("Generate Quiz", type="primary"):
        if topic:
            with st.spinner("Creating quiz..."):
                quiz = generate_quiz(topic, num_questions, vectorstore, llm)
                st.session_state.current_quiz = quiz

    if "current_quiz" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state.current_quiz)


def render_summary_tab(vectorstore, llm):
    """Render the Summary tab."""
    st.header("Topic Summary")
    st.markdown("Get a comprehensive summary of any FDNY topic.")

    topic = st.text_input("Enter a topic to summarize:", placeholder="e.g., Engine company operations")

    if st.button("Generate Summary", type="primary"):
        if topic:
            with st.spinner("Generating summary..."):
                summary = generate_summary(topic, vectorstore, llm)
                st.session_state.current_summary = summary

    if "current_summary" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state.current_summary)


def render_study_notes_tab(vectorstore, llm):
    """Render the Study Notes tab."""
    st.header("Study Notes")
    st.markdown("Generate organized study notes for any topic.")

    topic = st.text_input("Enter a topic for study notes:", placeholder="e.g., Ladder company forcible entry")

    if st.button("Generate Study Notes", type="primary"):
        if topic:
            with st.spinner("Creating study notes..."):
                notes = generate_study_notes(topic, vectorstore, llm)
                st.session_state.current_notes = notes

    if "current_notes" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state.current_notes)


def render_flashcards_tab(vectorstore, llm):
    """Render the Flashcards tab."""
    st.header("Flashcards")
    st.markdown("Create flashcards for quick review.")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Enter a topic for flashcards:", placeholder="e.g., SCBA operations")
    with col2:
        num_cards = st.selectbox("Cards:", [5, 10, 15, 20], index=1)

    if st.button("Generate Flashcards", type="primary"):
        if topic:
            with st.spinner("Creating flashcards..."):
                cards = generate_flashcards(topic, num_cards, vectorstore, llm)
                st.session_state.current_flashcards = cards

    if "current_flashcards" in st.session_state:
        st.markdown("---")

        # Parse and display as interactive cards
        cards_text = st.session_state.current_flashcards
        st.markdown(cards_text)


# --- Main App ---
def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="FDNY AI Tutor 2025",
        page_icon="🚒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set the OPENAI_API_KEY environment variable.")
        st.code("export OPENAI_API_KEY='your-api-key'")
        st.stop()

    # Check for vector store
    if not INDEX_DIR.exists():
        st.error("Vector store not found. Please run the indexing script first.")
        st.code("python src/index_books.py")
        st.stop()

    # Load resources
    try:
        vectorstore = load_vector_store()
        llm = get_llm()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

    # Title
    st.title("🚒 FDNY AI Tutor 2025")
    st.markdown("*Your intelligent study companion for FDNY training materials*")

    # Sidebar
    render_sidebar()

    # Main tabs
    tab_chat, tab_quiz, tab_summary, tab_notes, tab_flash = st.tabs([
        "💬 Chat",
        "📝 Quiz",
        "📋 Summary",
        "📚 Study Notes",
        "🎴 Flashcards"
    ])

    with tab_chat:
        render_chat_tab(vectorstore, llm)

    with tab_quiz:
        render_quiz_tab(vectorstore, llm)

    with tab_summary:
        render_summary_tab(vectorstore, llm)

    with tab_notes:
        render_study_notes_tab(vectorstore, llm)

    with tab_flash:
        render_flashcards_tab(vectorstore, llm)


if __name__ == "__main__":
    main()


import streamlit as st

from src.config import MAX_FILE_SIZE
from src.document_loader import load_pdf
from src.chunking import create_chunks
from src.vector_store import (
    get_collection,
    document_exists,
    store_chunks,
    clear_database,
)
from src.rag_pipeline import answer_question


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)


# --------------------------------------------------
# Page Header
# --------------------------------------------------

st.title("📚 RAG Document Assistant")

st.write(
    "Upload PDF documents and ask questions "
    "using Retrieval-Augmented Generation."
)


# --------------------------------------------------
# Sidebar — Document Management
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Process Documents",
        use_container_width=True,
    ):

        if not uploaded_files:
            st.warning(
                "Please upload at least one PDF."
            )

        else:
            for uploaded_file in uploaded_files:

                try:
                    pdf_bytes = uploaded_file.getvalue()

                    if len(pdf_bytes) > MAX_FILE_SIZE:
                        st.error(
                            f"{uploaded_file.name} exceeds "
                            "the 10 MB limit."
                        )
                        continue

                    # Step 1: Extract PDF text.
                    document_id, documents = load_pdf(
                        pdf_bytes=pdf_bytes,
                        filename=uploaded_file.name,
                    )

                    # Step 2: Check for duplicate documents.
                    collection = get_collection()

                    if document_exists(
                        collection,
                        document_id,
                    ):
                        st.info(
                            f"{uploaded_file.name} "
                            "is already indexed."
                        )
                        continue

                    if not documents:
                        st.warning(
                            f"No extractable text found in "
                            f"{uploaded_file.name}."
                        )
                        continue

                    # Step 3: Split text into chunks.
                    chunks = create_chunks(documents)

                    # Step 4: Generate and store embeddings.
                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):
                        stored_count = store_chunks(chunks)

                    st.success(
                        f"{uploaded_file.name}: "
                        f"{stored_count} chunks indexed."
                    )

                except Exception as error:
                    st.error(
                        f"Unable to process "
                        f"{uploaded_file.name}: {error}"
                    )

    st.divider()

    # Display database information.
    collection = get_collection()

    st.metric(
        "Indexed Chunks",
        collection.count(),
    )

    # Allow the user to reset the knowledge base.
    with st.expander("Database Settings"):

        confirm_clear = st.checkbox(
            "I understand this deletes all indexed documents."
        )

        if st.button(
            "Clear Knowledge Base",
            disabled=not confirm_clear,
            use_container_width=True,
        ):
            clear_database()
            st.success("Knowledge base cleared.")
            st.rerun()


# --------------------------------------------------
# Main Interface — Question Answering
# --------------------------------------------------

st.subheader("💬 Ask Your Documents")

question = st.text_input(
    "Enter your question",
    placeholder=(
        "How many annual leave days do employees receive?"
    ),
)

if st.button(
    "Generate Answer",
    type="primary",
):

    if not question.strip():
        st.warning(
            "Please enter a question."
        )

    elif get_collection().count() == 0:
        st.warning(
            "Please upload and process a PDF first."
        )

    else:
        try:
            with st.spinner(
                "Searching documents and generating answer..."
            ):

                answer, sources = answer_question(
                    question.strip()
                )

            # Display generated answer.
            st.subheader("Generated Answer")

            st.write(answer)

            # Display retrieved sources.
            if sources:

                st.subheader("Retrieved Sources")

                for index, source in enumerate(
                    sources,
                    start=1,
                ):

                    metadata = source["metadata"]

                    source_name = metadata.get(
                        "source",
                        "Unknown",
                    )

                    page = metadata.get(
                        "page",
                        "Unknown",
                    )

                    with st.expander(
                        f"Source {index}: "
                        f"{source_name} — Page {page}"
                    ):

                        st.write(
                            source["page_content"]
                        )

                        st.caption(
                            f"Similarity: "
                            f"{source['similarity']:.4f}"
                        )

                        st.caption(
                            f"Chunk ID: "
                            f"{source['chunk_id']}"
                        )

        except Exception as error:
            st.error(
                f"Unable to generate answer: {error}"
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "RAG Fundamentals to Advanced — Lesson 08"
)

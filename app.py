"""Streamlit user interface for the Research Intelligence Assistant.
Features Research Chat, Document Library, Cross-Paper Analysis, Evaluation Dashboard, and Source Viewer.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.paper_comparison import PaperComparator
from config import get_settings
from evaluation.benchmark_papers import create_benchmark_corpus
from evaluation.run_evaluation import run_full_evaluation
from generation.rag_pipeline import RAGPipeline
from ingestion.pipeline import IngestionPipeline
from retrieval.filters import MetadataFilter
from retrieval.retriever import SemanticRetriever
from storage.database import get_db, init_database
from storage.document_repository import DocumentRepository

# Page Configuration
st.set_page_config(
    page_title="Research Intelligence Assistant — Advanced RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize storage
init_database()
settings = get_settings()
doc_repo = DocumentRepository()

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .citation-badge {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .contradiction-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 12px;
        margin-bottom: 12px;
        border-radius: 4px;
    }
    .evidence-card {
        background-color: #fafafa;
        border: 1px solid #eeeeee;
        border-radius: 6px;
        padding: 12px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🔬 Research Assistant")
st.sidebar.caption("Production-Quality RAG Pipeline")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["Research Chat", "Document Library", "Cross-Paper Analysis", "Evaluation Dashboard", "Source Viewer"]
)

# Sidebar Settings & Stats
stats = doc_repo.get_document_stats()
st.sidebar.divider()
st.sidebar.subheader("Index Overview")
st.sidebar.write(f"📄 **Indexed Papers:** {stats['total_documents']}")
st.sidebar.write(f"📑 **Total Pages:** {stats['total_pages']}")
st.sidebar.write(f"🧩 **Indexed Chunks:** {stats['total_chunks']}")
if stats["ingestion_failures"] > 0:
    st.sidebar.warning(f"⚠️ Ingestion Failures: {stats['ingestion_failures']}")

st.sidebar.divider()
st.sidebar.caption("System Configuration")
top_k_select = st.sidebar.slider("Retrieval Depth (Top-K)", min_value=1, max_value=20, value=settings.default_top_k)
llm_model_display = settings.llm_model if settings.llm_api_key else "Local Grounded Synthesizer"
st.sidebar.info(f"**Embedding Model:** `bge-base-en-v1.5`\n\n**LLM Mode:** `{llm_model_display}`")


# ==============================================================================
# SECTION A: RESEARCH CHAT
# ==============================================================================
if nav_choice == "Research Chat":
    st.header("💬 Citation-Grounded Research Chat")
    st.caption("Ask natural-language questions across indexed research papers with verifiable, page-level citations.")

    if stats["total_documents"] == 0:
        st.warning("⚠️ No documents are currently indexed. Please go to **Document Library** to ingest papers or click below to load the benchmark suite.")
        if st.button("📥 Ingest Sample Research Benchmark Corpus"):
            with st.spinner("Ingesting benchmark papers..."):
                pdfs = create_benchmark_corpus()
                pipeline = IngestionPipeline()
                res = pipeline.ingest_files(pdfs)
                st.success(res.summary())
                st.rerun()

    # Session Chat History
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display past messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "claims" in msg and msg["claims"]:
                with st.expander("🔍 Verified Claims & Supporting Evidence"):
                    for c in msg["claims"]:
                        st.markdown(f"**Claim:** {c['claim_text']}")
                        st.markdown(f"**Source:** `{c['source_document']}` — **p. {c['source_page']}** | *Relevance:* `{c.get('evidence_relevance', 'direct')}`")
                        st.info(f"\"{c.get('supporting_evidence', '')}\"")
                        st.divider()

    # Chat Input
    query = st.chat_input("Enter your research question (e.g. 'What is the computational complexity of self-attention?')")
    if query:
        # Add user query
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and verifying citations..."):
                retriever = SemanticRetriever()
                rag_pipeline = RAGPipeline(retriever=retriever)
                resp = rag_pipeline.answer_question(query=query, top_k=top_k_select)

                # Render Answer
                if not resp.is_evidence_sufficient:
                    st.warning(f"⚠️ {resp.answer}")
                else:
                    st.markdown(resp.answer)

                # Inferences and interpretations
                if resp.interpretations_and_inferences:
                    with st.expander("💡 Inferences and Interpretations (Derived from Evidence)"):
                        for inf in resp.interpretations_and_inferences:
                            st.write(f"- {inf}")

                # Verified Claims & Evidence
                if resp.claims:
                    with st.expander(f"📚 Verified Citations ({len(resp.claims)} validated claim(s))"):
                        for c in resp.claims:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Claim:** {c.claim_text}")
                                st.caption(f"Source: **{c.source_document}**, Page {c.source_page}")
                                st.markdown(f"*{c.supporting_evidence}*")
                            with col2:
                                st.success("✅ Verified")
                                st.caption(f"Confidence: {c.confidence}")
                            st.divider()

                # Raw Retrieved Evidence
                if resp.retrieved_evidence:
                    with st.expander("🔬 Retrieved Source Chunks (Provenance & Scores)"):
                        for idx, chunk in enumerate(resp.retrieved_evidence, start=1):
                            st.markdown(f"**Chunk [{idx}]** — `{chunk['document_title']}` ({chunk['page_citation']}) | Similarity Score: `{chunk['similarity_score']}`")
                            st.text(chunk["text"])
                            st.divider()

                # Latency footer
                st.caption(f"⚡ Retrieval: {resp.retrieval_latency_ms:.1f}ms | Generation: {resp.generation_latency_ms:.1f}ms | Confidence: `{resp.overall_confidence}`")

                # Save assistant response to session state
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": resp.answer,
                    "claims": [c.model_dump() for c in resp.claims]
                })

        # Suggested follow-ups
        st.subheader("Suggested Follow-up Questions")
        cols = st.columns(3)
        suggestions = [
            "What limitations do the authors report?",
            "How do these experimental results compare to baselines?",
            "What dataset or evaluation protocol was used?"
        ]
        for idx, col in enumerate(cols):
            if col.button(suggestions[idx], key=f"sug_{idx}"):
                st.session_state["followup_query"] = suggestions[idx]
                st.rerun()


# ==============================================================================
# SECTION B: DOCUMENT LIBRARY
# ==============================================================================
elif nav_choice == "Document Library":
    st.header("📚 Document Library & Ingestion")
    st.caption("Upload, inspect, and manage indexed PDF research papers.")

    # Ingestion Tabs
    tab_upload, tab_manage, tab_benchmark = st.tabs(["Upload Research Papers", "Manage Indexed Papers", "Benchmark Corpus"])

    with tab_upload:
        st.subheader("Upload Research PDFs")
        uploaded_files = st.file_uploader(
            "Select one or multiple PDF documents (50+ papers supported)",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("🚀 Ingest Uploaded Documents"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            saved_paths = []
            for f in uploaded_files:
                dest = settings.papers_dir / f.name
                with open(dest, "wb") as buffer:
                    buffer.write(f.getbuffer())
                saved_paths.append(dest)

            pipeline = IngestionPipeline()

            def progress_cb(current, total, msg):
                progress_bar.progress(current / total)
                status_text.text(f"Processing [{current}/{total}]: {msg}")

            res = pipeline.ingest_files(saved_paths, progress_callback=progress_cb)
            progress_bar.progress(1.0)
            status_text.text("Ingestion completed!")

            st.success(res.summary())
            if res.errors:
                st.error("Encountered errors:")
                for e in res.errors:
                    st.write(f"- **{e['file']}**: {e['error']}")
            st.rerun()

    with tab_manage:
        st.subheader("Indexed Research Papers")
        search_kw = st.text_input("Filter by title or filename", "")
        docs = doc_repo.list_documents(search=search_kw if search_kw else None)

        if not docs:
            st.info("No documents found.")
        else:
            table_data = []
            for d in docs:
                chunks = doc_repo.get_chunks_by_document(d["id"])
                table_data.append({
                    "Document ID": d["id"],
                    "Title": d["title"],
                    "Filename": d["filename"],
                    "Year": d["publication_year"] or "N/A",
                    "Pages": d["total_pages"],
                    "Chunks": len(chunks),
                    "Status": d["ingestion_status"]
                })
            df_docs = pd.DataFrame(table_data)
            st.dataframe(df_docs, use_container_width=True)

            # Deletion control
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_del = st.selectbox("Select document to remove", [f"{d['id']} — {d['title']}" for d in docs])
            with col2:
                if st.button("🗑️ Delete Selected Document"):
                    doc_id_to_del = selected_del.split(" — ")[0]
                    pipeline = IngestionPipeline()
                    pipeline.delete_document(doc_id_to_del)
                    st.success(f"Deleted document {doc_id_to_del}")
                    st.rerun()

    with tab_benchmark:
        st.subheader("Automated Benchmark Suite Corpus")
        st.write("Generate and index the verified 8-paper benchmark corpus (Attention, ViT, ResNet, DeepConvLSTM, LoRA, QLoRA, DPR, FlashAttention).")
        if st.button("⚡ Generate & Ingest Benchmark Suite"):
            with st.spinner("Generating and indexing benchmark PDFs..."):
                pdfs = create_benchmark_corpus()
                pipeline = IngestionPipeline()
                res = pipeline.ingest_files(pdfs)
                st.success(res.summary())
                st.rerun()


# ==============================================================================
# SECTION C: CROSS-PAPER ANALYSIS
# ==============================================================================
elif nav_choice == "Cross-Paper Analysis":
    st.header("⚖️ Multi-Paper Comparison & Contradiction Detection")
    st.caption("Synthesize methodologies, experimental settings, findings, and identify conflicting conclusions across papers.")

    docs = doc_repo.list_documents()
    if len(docs) < 2:
        st.warning("⚠️ Cross-paper comparison requires at least 2 indexed research papers.")
    else:
        doc_options = {f"{d['title']} ({d['filename']})": d["id"] for d in docs}
        selected_titles = st.multiselect("Select papers to compare (leave blank to analyze entire indexed collection)", list(doc_options.keys()))
        selected_ids = [doc_options[t] for t in selected_titles] if selected_titles else None

        compare_query = st.text_input(
            "Comparative research question",
            value="Compare the architectures, methodologies, benchmarks, and reported findings across these papers."
        )

        if st.button("🔍 Execute Cross-Paper Analysis"):
            with st.spinner("Analyzing cross-paper evidence and detecting contradictions..."):
                comparator = PaperComparator()
                res = comparator.compare(query=compare_query, document_ids=selected_ids, top_k=20)

                st.subheader("Summary of Comparative Findings")
                st.markdown(res.summary)

                # Structured Comparison Table
                st.subheader("Structured Comparison Matrix")
                matrix_rows = []
                for r in res.rows:
                    matrix_rows.append({
                        "Paper": r.paper_title,
                        "Methodology": r.methodology,
                        "Dataset or Setting": r.dataset_or_setting,
                        "Main Finding": r.main_finding,
                        "Limitations": r.limitations,
                        "Source Pages": r.formatted_pages
                    })
                df_matrix = pd.DataFrame(matrix_rows)
                st.dataframe(df_matrix, use_container_width=True)

                # Disagreements and Contradictions
                st.subheader("Identified Disagreements & Potential Contradictions")
                if not res.disagreements:
                    st.info("No direct contradictions detected across the analyzed papers.")
                else:
                    for d in res.disagreements:
                        with st.container():
                            st.markdown(f"### ⚠️ Topic: {d.topic}")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"**Claim A:** {d.claim_a}")
                                st.markdown(f"**Citation:** `{d.citation_a}`")
                            with col_b:
                                st.markdown(f"**Claim B:** {d.claim_b}")
                                st.markdown(f"**Citation:** `{d.citation_b}`")

                            st.write(f"**Disagreement Type:** `{d.disagreement_type}` | **Evidence Strength:** `{d.evidence_strength}` | **Explicit:** `{d.is_explicit_contradiction}`")
                            if d.methodological_differences:
                                st.caption(f"**Methodological Divergence Note:** {d.methodological_differences}")
                            st.divider()

                # Research Gaps
                st.subheader("Documented Research Gaps & Limitations")
                for gap in res.research_gaps:
                    st.write(f"- {gap}")


# ==============================================================================
# SECTION D: EVALUATION DASHBOARD
# ==============================================================================
elif nav_choice == "Evaluation Dashboard":
    st.header("📊 Benchmark Evaluation Dashboard")
    st.caption("Empirically measured retrieval and citation validation metrics against 100+ research questions.")

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        if st.button("🚀 Run Live 105-Question Benchmark"):
            with st.spinner("Executing full benchmark suite on 105 research questions..."):
                run_full_evaluation()
                st.success("Evaluation completed and logged!")
                st.rerun()

    # Load latest evaluation from SQLite
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM evaluation_runs ORDER BY timestamp DESC LIMIT 1")
        latest_eval = cursor.fetchone()

    if not latest_eval:
        st.info("No evaluation runs recorded yet. Click 'Run Live 105-Question Benchmark' above to execute.")
    else:
        eval_data = dict(latest_eval)
        st.caption(f"Latest Benchmark Run: **{eval_data['timestamp']}** (Run ID: `{eval_data['id']}`)")

        # Metric KPI cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        recall_pct = (eval_data["recall_at_5"] or 0.0) * 100.0
        acc_pct = (eval_data["citation_accuracy"] or 0.0) * 100.0
        cov_pct = (eval_data["citation_coverage"] or 0.0) * 100.0

        kpi1.metric("Recall@5 (Measured)", f"{recall_pct:.1f}%", delta="Target: >=85.0%")
        kpi2.metric("Citation Accuracy", f"{acc_pct:.1f}%", delta="Target: >=90.0%")
        kpi3.metric("Citation Coverage", f"{cov_pct:.1f}%", delta="Comprehensive")
        kpi4.metric("Avg Retrieval Latency", f"{eval_data['avg_retrieval_latency_ms']:.1f} ms")

        # Full Report Markdown Viewer
        st.subheader("Detailed Evaluation Report")
        if eval_data.get("report_json"):
            st.markdown(eval_data["report_json"])


# ==============================================================================
# SECTION E: SOURCE VIEWER
# ==============================================================================
elif nav_choice == "Source Viewer":
    st.header("📄 PDF Source & Page Viewer")
    st.caption("Inspect extracted text, page layouts, headings, and chunk mappings.")

    docs = doc_repo.list_documents()
    if not docs:
        st.info("No documents indexed.")
    else:
        doc_map = {f"{d['title']} ({d['filename']})": d for d in docs}
        sel_doc_name = st.selectbox("Select Research Paper", list(doc_map.keys()))
        sel_doc = doc_map[sel_doc_name]

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM document_pages WHERE document_id = ? ORDER BY page_number ASC", (sel_doc["id"],))
            pages = cursor.fetchall()

        if not pages:
            st.warning("No page records found for this document.")
        else:
            page_numbers = [p["page_number"] for p in pages]
            sel_page_num = st.selectbox("Select Page Number", page_numbers)

            # Get chunks for this page
            all_chunks = doc_repo.get_chunks_by_document(sel_doc["id"])
            page_chunks = [c for c in all_chunks if c["start_page"] <= sel_page_num <= c["end_page"]]

            st.subheader(f"Page {sel_page_num} of {sel_doc['title']}")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Chunks on this page:** {len(page_chunks)}")
                for c in page_chunks:
                    st.info(f"**Chunk ID:** `{c['chunk_id']}` | **Tokens:** {c['token_count']} | **Heading:** `{c.get('section_heading') or 'None'}`\n\n{c['text']}")

            with col2:
                st.markdown("### Document Metadata")
                st.write(f"- **Filename:** `{sel_doc['filename']}`")
                st.write(f"- **Total Pages:** {sel_doc['total_pages']}")
                st.write(f"- **File Hash:** `{sel_doc['file_hash'][:16]}...`")
                st.write(f"- **Local Path:** `{sel_doc['file_path']}`")
                if Path(sel_doc["file_path"]).exists():
                    st.success("File available on disk.")

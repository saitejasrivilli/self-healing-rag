"""
Self-Healing RAG — Streamlit Demo
Run: streamlit run app.py
"""

import os
import sys
import time
import tempfile
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from rag_pipeline import SelfHealingRAG, RAGConfig

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Healing RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.stApp { background: #0a0e1a; color: #e2e8f0; }

.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
}

.confidence-bar {
    height: 8px;
    border-radius: 4px;
    background: #1e293b;
    overflow: hidden;
    margin-top: 6px;
}

.source-chip {
    display: inline-block;
    background: #1e3a5f;
    border: 1px solid #2563eb;
    color: #93c5fd;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px;
}

.answer-box {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-left: 4px solid #3b82f6;
    border-radius: 0 12px 12px 0;
    padding: 1.5rem;
    margin: 1rem 0;
    line-height: 1.8;
}

.heal-badge {
    background: #7c3aed22;
    border: 1px solid #7c3aed;
    color: #a78bfa;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
}

.verified-badge {
    background: #05966922;
    border: 1px solid #059669;
    color: #34d399;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
}

.failed-badge {
    background: #dc262622;
    border: 1px solid #dc2626;
    color: #f87171;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
}

.pipeline-step {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #1e293b;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.8rem;
    margin-right: 4px;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
  <h1 style="font-size:2.8rem; margin:0; background: linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    🧠 Self-Healing RAG
  </h1>
  <p style="color:#64748b; margin-top:0.5rem; font-size:0.95rem;">
    Retriever → Reranker → Generator → Verification Agent → Self-Healing Loop
  </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar — Config
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Config")

    gemini_key = st.text_input("Gemini API Key", type="password",
                                value=os.getenv("GEMINI_API_KEY", ""),
                                help="Get free key at ai.google.dev")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    st.divider()

    top_k_retrieve = st.slider("Top-K Retrieve", 5, 20, 10)
    top_k_rerank = st.slider("Top-K Rerank", 2, 8, 4)
    confidence_threshold = st.slider("Confidence Threshold", 0.3, 0.95, 0.65, 0.05)
    max_retries = st.slider("Max Self-Healing Retries", 1, 5, 3)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    st.markdown("**Architecture**")
    st.markdown("""
    - 🔍 **Retriever**: ChromaDB + BGE embeddings
    - 🎯 **Reranker**: CrossEncoder (ms-marco)
    - 🤖 **Generator**: Gemini 1.5 Flash
    - ✅ **Verifier**: LLM confidence scoring
    - 🔄 **Healer**: Query expansion on retry
    """)

    st.divider()
    st.caption("Built by [Sai Teja Srivilli](https://github.com/saitejasrivilli)")

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "history" not in st.session_state:
    st.session_state.history = []

# ──────────────────────────────────────────────
# Initialize pipeline
# ──────────────────────────────────────────────
def get_rag():
    config = RAGConfig(
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        confidence_threshold=confidence_threshold,
        max_retries=max_retries,
        temperature=temperature,
    )
    if st.session_state.rag is None:
        st.session_state.rag = SelfHealingRAG(config)
    else:
        st.session_state.rag.config = config
    return st.session_state.rag

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab_ingest, tab_query, tab_history = st.tabs(["📄 Ingest Documents", "🔍 Ask & Heal", "📊 Query History"])

# ── Tab 1: Ingest ──
with tab_ingest:
    st.markdown("### Upload documents to the knowledge base")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

    with col2:
        st.markdown("**Or use sample data**")
        if st.button("📦 Load Sample KB"):
            rag = get_rag()
            sample_dir = "./data/sample"
            os.makedirs(sample_dir, exist_ok=True)
            sample_text = """
# Transformer Architecture
Transformers use self-attention mechanisms to process sequences in parallel.
The model consists of encoder and decoder blocks with multi-head attention layers.
BERT uses only the encoder stack, while GPT uses only the decoder stack.
The attention formula is: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

# RAG Systems
Retrieval-Augmented Generation combines parametric (LLM) and non-parametric (retrieval) memory.
The retriever fetches relevant documents which are passed as context to the generator.
Dense retrieval uses bi-encoders; reranking uses cross-encoders for higher precision.
Self-healing RAG adds a verification loop that retries when confidence is below threshold.

# Vector Databases
ChromaDB is an open-source embedding database for AI applications.
Pinecone and Weaviate are cloud-native alternatives.
FAISS is optimized for pure similarity search without persistence.
Embeddings are typically 768 or 1536 dimensional vectors.

# LLM Fine-Tuning
LoRA (Low-Rank Adaptation) reduces trainable parameters by 99% vs full fine-tuning.
QLoRA combines quantization + LoRA enabling fine-tuning on consumer GPUs.
RLHF aligns models with human preferences using a reward model.
SFT (Supervised Fine-Tuning) is the first stage before RLHF.
"""
            with open(f"{sample_dir}/ml_knowledge_base.txt", "w") as f:
                f.write(sample_text)
            with st.spinner("Ingesting sample knowledge base..."):
                n = rag.ingest(sample_dir)
            st.session_state.ingested = True
            st.success(f"✅ Sample KB loaded — {n} chunks indexed")

    if uploaded:
        rag = get_rag()
        with tempfile.TemporaryDirectory() as tmpdir:
            for file in uploaded:
                path = os.path.join(tmpdir, file.name)
                with open(path, "wb") as f:
                    f.write(file.read())
            with st.spinner(f"Processing {len(uploaded)} file(s)..."):
                n = rag.ingest(tmpdir)
        st.session_state.ingested = True
        st.success(f"✅ Indexed {n} chunks from {len(uploaded)} file(s)")

    if st.session_state.ingested:
        st.markdown("""
        <div class="metric-card">
          <span style="color:#34d399">✓</span> Knowledge base ready — start querying in the <strong>Ask & Heal</strong> tab
        </div>
        """, unsafe_allow_html=True)

# ── Tab 2: Query ──
with tab_query:
    if not st.session_state.ingested:
        st.info("⬅️ Ingest documents first (or load the sample KB)")
    else:
        st.markdown("### Ask anything grounded in your documents")

        query = st.text_area("Your question", placeholder="What is LoRA fine-tuning?",
                              height=80, label_visibility="collapsed")

        col_btn, col_space = st.columns([1, 4])
        with col_btn:
            run = st.button("🔍 Run Query", use_container_width=True)

        if run and query.strip():
            rag = get_rag()

            # Pipeline progress
            progress_placeholder = st.empty()
            steps = ["🔍 Retrieving", "🎯 Reranking", "🤖 Generating", "✅ Verifying"]
            for i, step in enumerate(steps):
                with progress_placeholder.container():
                    st.markdown(
                        "".join(f'<span class="pipeline-step">{s}</span>' + (" → " if j < 3 else "") for j, s in enumerate(steps[:i+1])),
                        unsafe_allow_html=True,
                    )
                time.sleep(0.3)

            with st.spinner("Running self-healing pipeline..."):
                response = rag.query(query)

            progress_placeholder.empty()

            # ── Answer ──
            healed = response.attempts > 1
            verified_badge = '<span class="verified-badge">✅ VERIFIED</span>' if response.verified else '<span class="failed-badge">⚠️ LOW CONFIDENCE</span>'
            heal_badge = f'<span class="heal-badge">🔄 SELF-HEALED ({response.attempts} attempts)</span>' if healed else ""

            st.markdown(f"""
            <div style="margin-bottom:0.5rem">
              {verified_badge} {heal_badge}
            </div>
            <div class="answer-box">
              {response.answer.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # ── Metrics ──
            m1, m2, m3, m4 = st.columns(4)
            conf_color = "#34d399" if response.confidence >= 0.65 else "#f59e0b" if response.confidence >= 0.4 else "#f87171"

            with m1:
                st.metric("Confidence", f"{response.confidence:.0%}")
            with m2:
                st.metric("Attempts", response.attempts)
            with m3:
                st.metric("Latency", f"{response.latency_ms:.0f}ms")
            with m4:
                st.metric("Sources Used", len(response.sources))

            # Confidence bar
            bar_w = int(response.confidence * 100)
            st.markdown(f"""
            <div class="confidence-bar">
              <div style="width:{bar_w}%; height:100%; background: linear-gradient(90deg, {conf_color}88, {conf_color}); border-radius:4px; transition: width 0.8s ease;"></div>
            </div>
            <p style="color:#64748b; font-size:0.8rem; margin-top:4px;">Verifier reasoning: {response.reasoning}</p>
            """, unsafe_allow_html=True)

            if healed:
                st.markdown(f"""
                <div class="metric-card" style="border-color:#7c3aed">
                  <strong style="color:#a78bfa">🔄 Self-Healing activated</strong><br>
                  <span style="color:#94a3b8; font-size:0.85rem">
                    Original: <em>{query}</em><br>
                    Expanded: <em>{response.query_used}</em>
                  </span>
                </div>
                """, unsafe_allow_html=True)

            # ── Sources ──
            with st.expander("📚 Retrieved Sources", expanded=False):
                for i, chunk in enumerate(response.sources):
                    score_color = "#34d399" if chunk.score > 0.7 else "#f59e0b" if chunk.score > 0.4 else "#f87171"
                    st.markdown(f"""
                    <div class="metric-card">
                      <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <span class="source-chip">📄 {os.path.basename(chunk.source)}</span>
                        <span style="color:{score_color}; font-size:0.85rem">Score: {chunk.score:.3f}</span>
                      </div>
                      <p style="color:#cbd5e1; font-size:0.85rem; line-height:1.6; margin:0">{chunk.text[:400]}{'...' if len(chunk.text) > 400 else ''}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Store history
            st.session_state.history.append({
                "query": query,
                "answer": response.answer,
                "confidence": response.confidence,
                "attempts": response.attempts,
                "verified": response.verified,
                "latency_ms": response.latency_ms,
            })

# ── Tab 3: History ──
with tab_history:
    if not st.session_state.history:
        st.info("No queries yet — run some queries in the Ask & Heal tab.")
    else:
        st.markdown(f"### {len(st.session_state.history)} queries logged")

        avg_conf = sum(h["confidence"] for h in st.session_state.history) / len(st.session_state.history)
        healed_count = sum(1 for h in st.session_state.history if h["attempts"] > 1)
        verified_count = sum(1 for h in st.session_state.history if h["verified"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries", len(st.session_state.history))
        c2.metric("Avg Confidence", f"{avg_conf:.0%}")
        c3.metric("Self-Healed", healed_count)
        c4.metric("Verified", verified_count)

        st.divider()

        for i, h in enumerate(reversed(st.session_state.history)):
            badge = "✅" if h["verified"] else "⚠️"
            healed_label = f" 🔄×{h['attempts']}" if h["attempts"] > 1 else ""
            with st.expander(f"{badge} [{i+1}] {h['query'][:80]}... | {h['confidence']:.0%}{healed_label}"):
                st.markdown(f"**Answer:** {h['answer']}")
                st.caption(f"Latency: {h['latency_ms']:.0f}ms | Attempts: {h['attempts']} | Verified: {h['verified']}")

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

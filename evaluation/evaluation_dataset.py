"""Curated evaluation dataset containing 100+ research questions with reference answers,
ground-truth document titles, and expected page citations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from config import get_settings


class EvaluationQuestion(BaseModel):
    question_id: str
    question: str
    category: str
    expected_document_titles: List[str]
    expected_pages: List[int]
    expected_key_facts: List[str]
    reference_answer: str


def get_curated_evaluation_dataset() -> List[EvaluationQuestion]:
    """Return the curated suite of 105 research benchmark questions."""
    raw_questions = [
        # --- Category 1: Attention & Transformer Architectures (Q001 - Q020) ---
        {
            "id": "Q001", "cat": "Architectures",
            "q": "What is the computational complexity of the self-attention mechanism per layer relative to sequence length n and representation dimension d?",
            "docs": ["Attention Is All You Need", "Transformer Architecture Analysis"],
            "pages": [3, 6],
            "facts": ["Self-attention complexity is O(n^2 * d)", "Recurrent layers require O(n * d^2) sequential operations"],
            "ans": "Self-attention requires O(n^2 * d) operations per layer, connecting all positions with a constant number of sequentially executed operations [Attention Is All You Need, p. 6]."
        },
        {
            "id": "Q002", "cat": "Architectures",
            "q": "How does the Transformer model compute Scaled Dot-Product Attention?",
            "docs": ["Attention Is All You Need"],
            "pages": [3, 4],
            "facts": ["Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V", "Scaling factor 1/sqrt(d_k) prevents vanishing gradients in softmax"],
            "ans": "Scaled Dot-Product Attention is computed as softmax(QK^T / sqrt(d_k))V, where queries and keys of dimension d_k are divided by sqrt(d_k) to counteract small gradients [Attention Is All You Need, p. 4]."
        },
        {
            "id": "Q003", "cat": "Architectures",
            "q": "Why is multi-head attention beneficial compared to performing a single attention function with full dimension d_model?",
            "docs": ["Attention Is All You Need"],
            "pages": [4, 5],
            "facts": ["Allows model to jointly attend to information from different representation subspaces at different positions", "Maintains total computational cost similar to single-head attention"],
            "ans": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, which single attention pooling inhibits [Attention Is All You Need, p. 5]."
        },
        {
            "id": "Q004", "cat": "Architectures",
            "q": "What positional encoding formulation is used in the original Transformer architecture?",
            "docs": ["Attention Is All You Need"],
            "pages": [5, 6],
            "facts": ["Sinusoidal functions of different frequencies: sin and cos functions of position and dimension", "Allows the model to easily learn to attend by relative positions"],
            "ans": "The Transformer uses sine and cosine functions of varying frequencies: PE(pos, 2i) = sin(pos/10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos/10000^(2i/d_model)) [Attention Is All You Need, p. 6]."
        },
        {
            "id": "Q005", "cat": "Architectures",
            "q": "What is the primary role of residual connections and layer normalization in deep Transformer encoders?",
            "docs": ["Attention Is All You Need", "Deep Residual Learning for Image Recognition"],
            "pages": [3, 7],
            "facts": ["Residual connections prevent vanishing gradients across deep stacks", "LayerNorm stabilizes internal activations"],
            "ans": "Residual connections followed by LayerNorm (LayerNorm(x + Sublayer(x))) facilitate stable gradient backpropagation throughout deep encoder and decoder layers [Attention Is All You Need, p. 3]."
        },
        {
            "id": "Q006", "cat": "Architectures",
            "q": "How do Vision Transformers (ViT) process 2D images using standard Transformer encoder blocks?",
            "docs": ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"],
            "pages": [2, 3],
            "facts": ["Flattens images into non-overlapping patches of size 16x16", "Projects patches linearly and adds learnable 1D position embeddings"],
            "ans": "ViT flattens 2D images into patches (e.g. 16x16), projects them linearly into embedding vectors, prepends a learnable [class] token, and feeds them to an encoder [An Image is Worth 16x16 Words, p. 3]."
        },
        {
            "id": "Q007", "cat": "Architectures",
            "q": "What inductive biases do Vision Transformers lack compared to traditional Convolutional Neural Networks?",
            "docs": ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"],
            "pages": [1, 4],
            "facts": ["Lacks translation equivariance and 2D locality inductive bias in hidden layers", "Requires extensive pre-training on large datasets like JFT-300M"],
            "ans": "Unlike CNNs with translation equivariance and local receptive fields, ViT possesses minimal image-specific inductive biases and must learn spatial relationships from massive data [An Image is Worth 16x16 Words, p. 4]."
        },
        {
            "id": "Q008", "cat": "Architectures",
            "q": "How does FlashAttention optimize the computational speed and memory overhead of self-attention?",
            "docs": ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"],
            "pages": [1, 2],
            "facts": ["Tiling to compute softmax incrementally without writing NxN attention matrix to HBM", "Reduces memory accesses between SRAM and high-bandwidth GPU memory"],
            "ans": "FlashAttention uses tiling and online softmax rescaling to compute exact attention in GPU fast SRAM, eliminating costly reads/writes of the O(n^2) attention matrix to HBM [FlashAttention, p. 2]."
        },
        {
            "id": "Q009", "cat": "Architectures",
            "q": "What is Rotary Position Embedding (RoPE) and how does it encode relative token distances?",
            "docs": ["RoFormer: Enhanced Transformer with Rotary Position Embedding"],
            "pages": [2, 5],
            "facts": ["Multiplies key and query representations by orthogonal rotation matrices", "Inner product naturally decays with increasing token distance"],
            "ans": "RoPE applies complex orthogonal rotation matrices to query and key vectors such that their inner product directly incorporates relative positional distance [RoFormer, p. 5]."
        },
        {
            "id": "Q010", "cat": "Architectures",
            "q": "How does Grouped-Query Attention (GQA) balance latency and model quality compared to Multi-Head and Multi-Query Attention?",
            "docs": ["GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"],
            "pages": [1, 3],
            "facts": ["Shares single key and value heads across groups of query heads", "Achieves MQA inference speeds while maintaining MHA expressive capacity"],
            "ans": "GQA partitions query heads into G groups sharing single key and value heads, dramatically reducing KV cache memory bandwidth while preserving MHA quality [GQA, p. 3]."
        },
        {
            "id": "Q011", "cat": "Architectures",
            "q": "What is the feed-forward sublayer structure in each Transformer encoder block?",
            "docs": ["Attention Is All You Need"],
            "pages": [5],
            "facts": ["Consists of two linear transformations with a ReLU activation in between: FFN(x) = max(0, xW1 + b1)W2 + b2", "Inner layer dimensionality is typically 4x d_model"],
            "ans": "Each block contains a position-wise feed-forward network with two linear transformations and a non-linear activation: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2 with inner dimension 4*d_model [Attention Is All You Need, p. 5]."
        },
        {
            "id": "Q012", "cat": "Architectures",
            "q": "Why is masking required in the Transformer decoder multi-head attention sublayer?",
            "docs": ["Attention Is All You Need"],
            "pages": [3, 4],
            "facts": ["Prevents positions from attending to subsequent positions during autoregressive generation", "Preserves the auto-regressive property"],
            "ans": "Masked attention sets subsequent positions to -infinity before softmax, ensuring predictions for position i depend solely on known outputs at positions less than i [Attention Is All You Need, p. 4]."
        },
        {
            "id": "Q013", "cat": "Architectures",
            "q": "What learning rate schedule was utilized during the training of the original Transformer base model?",
            "docs": ["Attention Is All You Need"],
            "pages": [7, 8],
            "facts": ["Linear warmup for warmup_steps (4000) followed by inverse square root decay", "lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))"],
            "ans": "The authors used an inverse square root decay schedule with 4000 linear warmup steps: lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)) [Attention Is All You Need, p. 7]."
        },
        {
            "id": "Q014", "cat": "Architectures",
            "q": "How does Swin Transformer introduce hierarchical representations into vision backbones?",
            "docs": ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"],
            "pages": [1, 3],
            "facts": ["Computes self-attention within non-overlapping local windows", "Shifted window partitioning allows cross-window connections with linear computational complexity"],
            "ans": "Swin Transformer limits self-attention to local windows and shifts window boundaries between consecutive layers to enable cross-window connections with linear image-size scaling [Swin Transformer, p. 3]."
        },
        {
            "id": "Q015", "cat": "Architectures",
            "q": "What is the difference between Pre-LN and Post-LN Transformer block configurations?",
            "docs": ["On Layer Normalization in the Transformer Architecture"],
            "pages": [2, 4],
            "facts": ["Post-LN applies LayerNorm after residual addition; Pre-LN applies LayerNorm inside residual branch before sublayer", "Pre-LN avoids gradient vanishing at initialization and enables warmup-free training"],
            "ans": "Pre-LN applies Layer Normalization before each sublayer rather than after the residual addition, preventing vanishing/exploding gradients at early iterations and stabilizing deep models [On Layer Normalization, p. 4]."
        },
        {
            "id": "Q016", "cat": "Architectures",
            "q": "What regularization techniques were applied to prevent overfitting in the Transformer?",
            "docs": ["Attention Is All You Need"],
            "pages": [7, 8],
            "facts": ["Residual dropout of P_drop = 0.1 applied to sublayer outputs", "Label smoothing of epsilon_ls = 0.1 applied during cross-entropy loss"],
            "ans": "Regularization included residual dropout (rate 0.1), embedding dropout, and label smoothing (value 0.1) which penalizes overconfident predictions [Attention Is All You Need, p. 7]."
        },
        {
            "id": "Q017", "cat": "Architectures",
            "q": "What hardware and training time were required for the Transformer base model on WMT 2014 English-to-German translation?",
            "docs": ["Attention Is All You Need"],
            "pages": [7, 8],
            "facts": ["8 NVIDIA P100 GPUs for 100,000 steps (12 hours)", "Total training cost was substantially lower than competitive recurrent or convolutional models"],
            "ans": "The base model was trained on 8 NVIDIA P100 GPUs for 100,000 steps over 12 hours, achieving higher BLEU at a fraction of prior training costs [Attention Is All You Need, p. 8]."
        },
        {
            "id": "Q018", "cat": "Architectures",
            "q": "How does Mixture-of-Experts (MoE) scale parameter count without proportionally increasing inference FLOPs?",
            "docs": ["Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"],
            "pages": [2, 4],
            "facts": ["Replaces dense FFN with multiple expert networks", "A routing gate dynamically routes each token to top-1 or top-k experts"],
            "ans": "MoE replaces dense FFN layers with multiple expert networks and uses a routing gate to direct each token to only top-1 or top-2 experts, keeping computational FLOPs constant [Switch Transformers, p. 4]."
        },
        {
            "id": "Q019", "cat": "Architectures",
            "q": "What is the purpose of ALiBi (Attention with Linear Biases) in extending context window length?",
            "docs": ["Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"],
            "pages": [1, 3],
            "facts": ["Biases query-key attention scores with static linear penalties proportional to distance", "Enables models to extrapolate to sequences longer than those seen during training"],
            "ans": "ALiBi adds static linear distance penalties directly to attention logits, enabling models to extrapolate seamlessly to sequences significantly longer than training sequences [Train Short, Test Long, p. 3]."
        },
        {
            "id": "Q020", "cat": "Architectures",
            "q": "How does KV caching reduce computational redundancy during autoregressive token generation?",
            "docs": ["Efficient Memory Management for Large Language Model Serving with PagedAttention"],
            "pages": [2, 3],
            "facts": ["Stores key and value vectors of previously generated tokens in memory", "Avoids recomputing attention states for historical prompt and generation tokens"],
            "ans": "KV caching retains keys and values from preceding time steps in GPU memory, avoiding redundant re-computation of historical states at each sequential decoding step [PagedAttention, p. 3]."
        },

        # --- Category 2: Retrieval-Augmented Generation & Dense Retrieval (Q021 - Q040) ---
        {
            "id": "Q021", "cat": "RAG & Retrieval",
            "q": "How does Retrieval-Augmented Generation (RAG) combine parametric and non-parametric memory?",
            "docs": ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"],
            "pages": [1, 2],
            "facts": ["Parametric memory is a pre-trained seq2seq generator (BART)", "Non-parametric memory is a dense vector index of Wikipedia passages retrieved by DPR"],
            "ans": "RAG couples parametric memory (a pre-trained seq2seq neural generator) with non-parametric memory (a dense vector index queried via dense passage retrieval) [RAG for Knowledge-Intensive NLP, p. 2]."
        },
        {
            "id": "Q022", "cat": "RAG & Retrieval",
            "q": "What is the difference between RAG-Sequence and RAG-Token models in marginalizing over retrieved documents?",
            "docs": ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"],
            "pages": [2, 3],
            "facts": ["RAG-Sequence conditions the entire generated sequence on the same retrieved document", "RAG-Token can select different retrieved documents for each individual token generated"],
            "ans": "RAG-Sequence uses the same retrieved document to predict all output tokens, whereas RAG-Token can draw on different passages when generating each sequential token [RAG for Knowledge-Intensive NLP, p. 3]."
        },
        {
            "id": "Q023", "cat": "RAG & Retrieval",
            "q": "How does Dense Passage Retrieval (DPR) train its dual-encoder architecture for open-domain question answering?",
            "docs": ["Dense Passage Retrieval for Open-Domain Question Answering"],
            "pages": [2, 3],
            "facts": ["Uses two independent BERT encoders for question and passage", "Trained with negative log-likelihood loss using in-batch negatives and hard BM25 negatives"],
            "ans": "DPR trains dual BERT encoders for questions and passages using contrastive loss with in-batch negatives and gold BM25 mining to maximize similarity of relevant pairs [Dense Passage Retrieval, p. 3]."
        },
        {
            "id": "Q024", "cat": "RAG & Retrieval",
            "q": "Why is Inner Product (IP) similarity preferred over Euclidean distance (L2) when index vectors are L2-normalized?",
            "docs": ["FAISS: A Library for Efficient Similarity Search", "BGE Technical Report"],
            "pages": [4, 7],
            "facts": ["Cosine similarity equals inner product for normalized vectors", "Inner product avoids computing square root and norm operations during matrix multiplication"],
            "ans": "When embeddings are L2-normalized to unit length, inner product is mathematically identical to cosine similarity and requires fewer floating-point operations [FAISS, p. 4]."
        },
        {
            "id": "Q025", "cat": "RAG & Retrieval",
            "q": "What advantages does hybrid search combining BM25 and dense retrieval provide over pure vector search?",
            "docs": ["Dense Passage Retrieval for Open-Domain Question Answering", "Hybrid Search for Information Retrieval"],
            "pages": [5, 6],
            "facts": ["BM25 captures exact keyword matches, rare terminology, and serial numbers", "Dense retrieval handles synonymy, paraphrase, and semantic intent"],
            "ans": "Hybrid search pairs lexical precision (BM25 for exact terms, acronyms, and product IDs) with dense vector recall (handling semantic paraphrasing and conceptual similarity) [Dense Passage Retrieval, p. 5]."
        },
        {
            "id": "Q026", "cat": "RAG & Retrieval",
            "q": "How does cross-encoder re-ranking improve retrieval quality compared to bi-encoder first-stage retrieval?",
            "docs": ["Passage Re-ranking with BERT", "Dense Passage Retrieval"],
            "pages": [3, 4],
            "facts": ["Cross-encoders compute full all-to-all cross-attention between query and passage tokens", "Bi-encoders separate query and document encoding into independent vectors"],
            "ans": "Cross-encoders process query and passage together through all transformer layers, allowing deep token-level cross-attention that significantly outperforms isolated bi-encoder embeddings [Passage Re-ranking with BERT, p. 3]."
        },
        {
            "id": "Q027", "cat": "RAG & Retrieval",
            "q": "What chunking trade-offs arise when selecting small (~200 token) versus large (~1000 token) chunk sizes in RAG?",
            "docs": ["Chunking Strategies for Retrieval-Augmented Generation", "Dense Passage Retrieval"],
            "pages": [3, 7],
            "facts": ["Small chunks yield high embedding specificity but lose broad context across boundaries", "Large chunks preserve cross-paragraph context but dilute semantic similarity vectors"],
            "ans": "Small chunks optimize embedding specificity for targeted fact retrieval, whereas larger chunks preserve global narrative context at the expense of vector concentration [Chunking Strategies for RAG, p. 3]."
        },
        {
            "id": "Q028", "cat": "RAG & Retrieval",
            "q": "How does BAAI/bge-base-en-v1.5 format query inputs during retrieval tasks?",
            "docs": ["C-Pack: Packaged Resources To Advance General Chinese and English Embedding"],
            "pages": [3, 4],
            "facts": ["Prepends instruction: 'Represent this sentence for searching relevant passages: '", "Passages do not require the instruction prefix"],
            "ans": "BGE models instruct queries with 'Represent this sentence for searching relevant passages: ' to align asymmetric query-document vector representations [C-Pack BGE Report, p. 4]."
        },
        {
            "id": "Q029", "cat": "RAG & Retrieval",
            "q": "What is the role of Maximum Marginal Relevance (MMR) in diverse chunk retrieval?",
            "docs": ["The Use of MMR, Diversity-Based Reranking for Reordering Documents"],
            "pages": [1, 2],
            "facts": ["Balances relevance to the query with dissimilarity to previously selected chunks", "Reduces redundancy in the LLM context window"],
            "ans": "MMR balances query relevance with novelty relative to already selected chunks, preventing the retrieved context from being dominated by repetitive passages [The Use of MMR, p. 2]."
        },
        {
            "id": "Q030", "cat": "RAG & Retrieval",
            "q": "How does the 'Lost in the Middle' phenomenon affect LLM utilization of retrieved context passages?",
            "docs": ["Lost in the Middle: How Language Models Use Long Contexts"],
            "pages": [1, 3],
            "facts": ["LLM performance is highest when relevant information is at the very beginning or end of context", "Performance degrades substantially when key facts are placed in the middle of long contexts"],
            "ans": "Language models recall information effectively when it appears at the beginning or end of input context, but performance plummets when relevant passages are positioned in the middle [Lost in the Middle, p. 3]."
        },
        {
            "id": "Q031", "cat": "RAG & Retrieval",
            "q": "What are the core failure modes of standard naive RAG pipelines?",
            "docs": ["Seven Failure Points When Engineering a Retrieval Augmented Generation System"],
            "pages": [2, 5],
            "facts": ["Missing content, missed top ranked chunks, out of context chunks, wrong format, hallucination", "Extraction failures and incomplete retrieval synthesis"],
            "ans": "Primary failure modes include retrieval misses (relevant documents not retrieved), incomplete synthesis (failing to extract all facts), and hallucinated extrapolation beyond retrieved facts [Seven Failure Points, p. 2]."
        },
        {
            "id": "Q032", "cat": "RAG & Retrieval",
            "q": "How does Hypothetical Document Embeddings (HyDE) bridge the lexical gap in zero-shot retrieval?",
            "docs": ["Precise Zero-Shot Dense Retrieval without Relevance Labels"],
            "pages": [2, 3],
            "facts": ["Instructs LLM to generate a hypothetical answer document for the query", "Embeds the hypothetical document to search actual corpus chunks via dense vector similarity"],
            "ans": "HyDE prompts an LLM to generate a hypothetical answer document and encodes that hallucinated passage to retrieve genuine documents matching its vector distribution [Precise Zero-Shot Dense Retrieval, p. 3]."
        },
        {
            "id": "Q033", "cat": "RAG & Retrieval",
            "q": "What metric does Recall@K evaluate in a retrieval evaluation benchmark?",
            "docs": ["Information Retrieval: Implementing and Evaluating Search Engines"],
            "pages": [12, 14],
            "facts": ["Fraction of relevant ground-truth documents captured within the top-k retrieved list", "Recall@K = |Retrieved@K intersect Relevant| / |Relevant|"],
            "ans": "Recall@K measures the proportion of known relevant documents or chunks successfully retrieved in the first K search positions [Information Retrieval, p. 12]."
        },
        {
            "id": "Q034", "cat": "RAG & Retrieval",
            "q": "What is Mean Reciprocal Rank (MRR) and how is it calculated?",
            "docs": ["Information Retrieval: Implementing and Evaluating Search Engines"],
            "pages": [15],
            "facts": ["Average of the reciprocal ranks of the first relevant document across all evaluation queries", "MRR = (1/|Q|) * sum(1 / rank_i)"],
            "ans": "MRR is the average of reciprocal ranks (1/rank) of the first relevant document retrieved across all evaluation queries [Information Retrieval, p. 15]."
        },
        {
            "id": "Q035", "cat": "RAG & Retrieval",
            "q": "How does Self-RAG train an LLM to selectively retrieve and evaluate passage relevance?",
            "docs": ["Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"],
            "pages": [2, 4],
            "facts": ["Generates reflection tokens: [Retrieve], [IsRel], [IsSup], [IsUse]", "Determines dynamically whether retrieval is needed and critiques factual grounding"],
            "ans": "Self-RAG outputs special reflection tokens to decide dynamically whether to retrieve passages, critique relevance ([IsRel]), and check factual support ([IsSup]) [Self-RAG, p. 4]."
        },
        {
            "id": "Q036", "cat": "RAG & Retrieval",
            "q": "What is the difference between Inverted File (IVF) and Hierarchical Navigable Small World (HNSW) index structures in FAISS?",
            "docs": ["Billion-scale similarity search with GPUs", "Efficient and robust approximate nearest neighbor search using HNSW graphs"],
            "pages": [3, 8],
            "facts": ["IVF partitions vector space into Voronoi cells using k-means clustering", "HNSW constructs a multi-layer proximity graph supporting logarithmic search time"],
            "ans": "IVF clusters vectors into Voronoi cells for inverted list lookup, whereas HNSW creates a hierarchical graph for rapid graph traversal [Billion-scale similarity search with GPUs, p. 3]."
        },
        {
            "id": "Q037", "cat": "RAG & Retrieval",
            "q": "How does ColBERT achieve efficient multi-vector token-level interaction during retrieval?",
            "docs": ["ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"],
            "pages": [2, 4],
            "facts": ["Encodes query and passage tokens into bags of vectors", "Applies MaxSim operator to compute max inner product for each query token across passage tokens"],
            "ans": "ColBERT retains token-level embeddings and computes late interaction using the MaxSim operator, combining bi-encoder speed with token-level expressive power [ColBERT, p. 2]."
        },
        {
            "id": "Q038", "cat": "RAG & Retrieval",
            "q": "Why is document deduplication critical prior to vector indexing in academic RAG?",
            "docs": ["Deduplication of Training Data Makes Language Models Better"],
            "pages": [1, 3],
            "facts": ["Prevents duplicate chunks from occupying multiple slots in top-k retrieval results", "Ensures diversity of retrieved evidence across papers"],
            "ans": "Deduplication prevents identical or revision papers from saturating the top-k retrieved list with redundant chunks, preserving context budget for distinct evidence [Deduplication of Data, p. 3]."
        },
        {
            "id": "Q039", "cat": "RAG & Retrieval",
            "q": "How do metadata filters improve precision in scientific question answering?",
            "docs": ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"],
            "pages": [4, 6],
            "facts": ["Constrains vector candidate search by author, publication year, or section heading", "Eliminates out-of-scope papers before similarity ranking"],
            "ans": "Metadata filters restrict candidate search spaces to designated authors, publication years, or sections, eliminating out-of-scope matches prior to similarity ranking [RAG for Knowledge-Intensive NLP, p. 4]."
        },
        {
            "id": "Q040", "cat": "RAG & Retrieval",
            "q": "What is the effect of chunk overlap on sentence integrity across chunk boundaries?",
            "docs": ["Chunking Strategies for Retrieval-Augmented Generation"],
            "pages": [4, 5],
            "facts": ["Overlap prevents boundary sentences from being split in half", "Preserves semantic context across adjacent windows"],
            "ans": "Chunk overlap ensures sentences near partition boundaries remain intact in at least one chunk, maintaining grammatical continuity and local context [Chunking Strategies for RAG, p. 4]."
        },

        # --- Category 3: Activity Recognition & Wearable Sensing (Q041 - Q060) ---
        {
            "id": "Q041", "cat": "Activity Recognition",
            "q": "What are the main sensor modalities utilized for Human Activity Recognition (HAR)?",
            "docs": ["Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [2, 3],
            "facts": ["Inertial sensors: 3-axis accelerometers and gyroscopes", "Wearable physiological sensors, ambient sensors, and computer vision feeds"],
            "ans": "Wearable HAR predominantly relies on inertial sensors (triaxial accelerometers and gyroscopes), supplemented by barometers and heart rate monitors [Deep Learning for Sensor-based HAR, p. 2]."
        },
        {
            "id": "Q042", "cat": "Activity Recognition",
            "q": "What limitations of manual feature engineering in HAR did deep learning models address?",
            "docs": ["Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [3, 4],
            "facts": ["Manual features (mean, variance, FFT peaks) require domain expertise and fail on complex activities", "Deep models automatically extract hierarchical temporal-spatial features directly from raw signals"],
            "ans": "Manual feature extraction requires hand-crafted statistical metrics that struggle with complex composite activities; deep neural networks learn representations directly from raw sensor streams [Deep Learning for Sensor-based HAR, p. 4]."
        },
        {
            "id": "Q043", "cat": "Activity Recognition",
            "q": "How does DeepConvLSTM combine convolutional and recurrent layers for multimodal wearable sensing?",
            "docs": ["Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition"],
            "pages": [1, 3],
            "facts": ["Four 1D convolutional layers act as feature extractors over sensor channels", "Two recurrent LSTM layers model temporal dynamics and sequential dependencies"],
            "ans": "DeepConvLSTM applies 1D convolutions across multichannel sensor data to extract spatial feature maps, followed by recurrent LSTM layers that model temporal transitions [DeepConvLSTM, p. 3]."
        },
        {
            "id": "Q044", "cat": "Activity Recognition",
            "q": "Why does sensor placement on the human body (wrist, waist, thigh, chest) impact HAR classification performance?",
            "docs": ["Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [5, 6],
            "facts": ["Waist and chest sensors capture whole-body locomotion (walking, running, stairs)", "Wrist sensors capture upper-limb gestures (eating, writing) but experience high peripheral noise"],
            "ans": "Waist placement aligns with center of mass for accurate locomotion tracking, while wrist placement excels at gesture detection but suffers from excessive extraneous movements [Deep Learning for Sensor-based HAR, p. 5]."
        },
        {
            "id": "Q045", "cat": "Activity Recognition",
            "q": "What sliding window lengths are standard for segmenting continuous inertial sensor streams?",
            "docs": ["Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [4, 7],
            "facts": ["Window durations typically range between 1 and 5 seconds (default ~2.56 seconds)", "50% overlap is standard to prevent missing transitions"],
            "ans": "Studies standardly adopt sliding windows of 1 to 5 seconds (commonly 2.56 seconds with 128 samples at 50 Hz) with 50% overlap [Deep Learning for Sensor-based HAR, p. 7]."
        },
        {
            "id": "Q046", "cat": "Activity Recognition",
            "q": "How does domain adaptation resolve cross-subject variability in wearable sensing models?",
            "docs": ["Domain Adaptation for Human Activity Recognition"],
            "pages": [2, 5],
            "facts": ["Aligns feature distributions between source and target subjects", "Minimizes Maximum Mean Discrepancy (MMD) or adversarial discriminator loss"],
            "ans": "Domain adaptation mitigates inter-subject variance by aligning latent feature distributions across users via adversarial or metric discrepancy minimization [Domain Adaptation for HAR, p. 2]."
        },
        {
            "id": "Q047", "cat": "Activity Recognition",
            "q": "What challenges does class imbalance present in continuous real-world activity monitoring?",
            "docs": ["Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [8, 9],
            "facts": ["Null or sedentary classes dominate 80%+ of daily recordings", "Rare activities like falls or specific sports have sparse training examples"],
            "ans": "Everyday activity datasets are heavily skewed towards sitting/standing, leading naive classifiers to over-predict dominant sedentary classes and miss rare critical events like falls [Deep Learning for Sensor-based HAR, p. 8]."
        },
        {
            "id": "Q048", "cat": "Activity Recognition",
            "q": "How do attention mechanisms benefit sensor-based activity recognition over pure LSTMs?",
            "docs": ["Attention-based Multimodal Deep Learning for Activity Recognition"],
            "pages": [3, 4],
            "facts": ["Assigns dynamic importance weights to key time intervals and relevant sensor axes", "Focuses computation on decisive motion moments rather than uniform temporal pooling"],
            "ans": "Temporal attention highlights specific decisive movement bursts while channel attention dynamically weights the most informative sensor axes [Attention-based Multimodal DL for HAR, p. 3]."
        },
        {
            "id": "Q049", "cat": "Activity Recognition",
            "q": "What is the Opportunity Activity Recognition benchmark and what types of activities does it evaluate?",
            "docs": ["The Opportunity Challenge: A benchmark database for on-body activity recognition"],
            "pages": [1, 2],
            "facts": ["Includes 72 body-worn, object, and ambient sensors across naturalistic daily living scenarios", "Evaluates high-level locomotion and fine-grained manipulative gestures (opening doors, drinking)"],
            "ans": "The Opportunity benchmark evaluates naturalistic daily living through body-worn and object sensors, capturing fine-grained gestures and locomotion [The Opportunity Challenge, p. 2]."
        },
        {
            "id": "Q050", "cat": "Activity Recognition",
            "q": "How does self-supervised contrastive learning (SimCLR / CPC) train HAR backbones on unlabeled sensor data?",
            "docs": ["Contrastive Predictive Coding for Human Activity Recognition"],
            "pages": [2, 4],
            "facts": ["Applies temporal jitter, permutation, or noise augmentations to sensor segments", "Pulls augmented views of the same window together while pushing negative windows apart"],
            "ans": "Self-supervised frameworks apply temporal perturbations to inertial windows and use contrastive InfoNCE loss to learn representations without manual annotations [Contrastive Predictive Coding for HAR, p. 4]."
        },
        {
            "id": "Q051", "cat": "Activity Recognition",
            "q": "Why is battery and computational efficiency critical for edge HAR on microcontrollers?",
            "docs": ["TinyHAR: A Lightweight Deep Learning Model for Human Activity Recognition"],
            "pages": [1, 2],
            "facts": ["Wearable smartwatches and IoT nodes have constrained RAM (under 512 KB) and battery life", "Complex recurrent models cause thermal throttling and fast battery drain"],
            "ans": "Edge wearables feature microcontrollers with strict memory constraints and tiny batteries, demanding lightweight quantized models that avoid heavy recurrent activations [TinyHAR, p. 2]."
        },
        {
            "id": "Q052", "cat": "Activity Recognition",
            "q": "How does 8-bit integer quantization (INT8) affect HAR model accuracy and latency?",
            "docs": ["TinyHAR: A Lightweight Deep Learning Model for Human Activity Recognition"],
            "pages": [4, 5],
            "facts": ["Reduces model memory footprint by 4x", "Decreases inference latency by 2-3x with less than 1% degradation in F1-score"],
            "ans": "INT8 post-training quantization cuts model size by 75% and speeds up inference by 2-3x while maintaining within 1% of full-precision F1 accuracy [TinyHAR, p. 5]."
        },
        {
            "id": "Q053", "cat": "Activity Recognition",
            "q": "What is the PAMAP2 dataset and what experimental setup was used during its collection?",
            "docs": ["PAMAP2 Physical Activity Monitoring Benchmark"],
            "pages": [1, 3],
            "facts": ["3 IMUs placed on wrist, chest, and ankle alongside a heart rate monitor", "Covers 18 different physical activities from 9 subjects"],
            "ans": "PAMAP2 monitors 18 activities performed by 9 subjects using three IMUs (wrist, chest, ankle) and a heart rate monitor sampling at 100 Hz [PAMAP2 Physical Activity Monitoring, p. 3]."
        },
        {
            "id": "Q054", "cat": "Activity Recognition",
            "q": "What sensor sampling frequency is sufficient for identifying human ambulatory activities?",
            "docs": ["Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [4],
            "facts": ["Human body movements generally peak under 10-15 Hz", "Sampling rates of 20 Hz to 50 Hz capture almost all kinematic dynamics"],
            "ans": "Frequencies between 20 Hz and 50 Hz are sufficient because human bodily motions rarely exceed 15 Hz in frequency spectrum [Deep Learning for Sensor-based HAR, p. 4]."
        },
        {
            "id": "Q055", "cat": "Activity Recognition",
            "q": "How do Graph Convolutional Networks (GCN) model human skeleton topology in video-based HAR?",
            "docs": ["Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"],
            "pages": [2, 3],
            "facts": ["Represents human joints as graph nodes and natural bones as graph edges", "Spatial graph convolutions aggregate neighbor joint features over temporal sequences"],
            "ans": "ST-GCN models anatomical joints as graph vertices and physical bones as edges, applying spatial-temporal convolutions along skeletal links [ST-GCN for Skeleton Action Recognition, p. 3]."
        },
        {
            "id": "Q056", "cat": "Activity Recognition",
            "q": "What evaluation metric is preferred over accuracy when evaluating imbalanced activity recognition benchmarks?",
            "docs": ["Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [8],
            "facts": ["Macro-averaged F1-score or Weighted F1-score", "Standard accuracy masks poor classification on minority classes"],
            "ans": "Macro-F1 score is the preferred metric because it weights all activity classes equally regardless of their sample prevalence in the test set [Deep Learning for Sensor-based HAR, p. 8]."
        },
        {
            "id": "Q057", "cat": "Activity Recognition",
            "q": "How does sensor orientation variance affect accelerometer readings during daily living?",
            "docs": ["Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [6, 7],
            "facts": ["Gravitational acceleration (1g) shifts across x, y, z axes when sensor tilts", "Magnitude of acceleration vector sqrt(x^2 + y^2 + z^2) provides orientation invariance"],
            "ans": "Tilting shifts the 1g gravity component across Cartesian axes; computing total acceleration magnitude sqrt(x^2 + y^2 + z^2) yields orientation-invariant signals [Deep Learning for Sensor-based HAR, p. 6]."
        },
        {
            "id": "Q058", "cat": "Activity Recognition",
            "q": "What are the limitations of video-based activity recognition compared to wearable IMUs in home healthcare?",
            "docs": ["Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [1, 2],
            "facts": ["Video introduces severe visual privacy violations in private spaces (bathrooms, bedrooms)", "Suffers from occlusion, lighting variation, and fixed field of view"],
            "ans": "Video sensors raise acute privacy concerns and suffer from room occlusions and lighting shifts, whereas wearable IMUs operate anywhere without recording visuals [Deep Learning for Sensor-based HAR, p. 2]."
        },
        {
            "id": "Q059", "cat": "Activity Recognition",
            "q": "How can generative adversarial networks (GANs) assist with missing sensor modality imputation?",
            "docs": ["Adversarial Data Imputation in Multimodal Sensor Networks"],
            "pages": [2, 4],
            "facts": ["A generator network predicts missing IMU axes conditioned on available channels", "Discriminator enforces realistic joint temporal correlations"],
            "ans": "Conditional GANs synthesize missing sensor data channels conditioned on surviving sensor streams, preserving temporal covariance during dropouts [Adversarial Data Imputation, p. 4]."
        },
        {
            "id": "Q060", "cat": "Activity Recognition",
            "q": "What is the impact of user stride frequency variations on gait recognition accuracy?",
            "docs": ["Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [7, 8],
            "facts": ["Walking speeds alter signal frequency and peak-to-peak amplitude", "Fixed-size windowing models experience domain shift when walking speed changes"],
            "ans": "Changes in user gait pace alter the fundamental harmonic frequency in inertial signals, leading to classification errors unless adaptive normalization is applied [Deep Learning for Sensor-based HAR, p. 7]."
        },

        # --- Category 4: Model Optimization & Efficient Training (Q061 - Q080) ---
        {
            "id": "Q061", "cat": "Optimization",
            "q": "How does Low-Rank Adaptation (LoRA) freeze pre-trained weights while adapting LLMs?",
            "docs": ["LoRA: Low-Rank Adaptation of Large Language Models"],
            "pages": [1, 3],
            "facts": ["Decomposes weight update matrix into low-rank matrices: delta_W = B * A, where B in R^(d x r) and A in R^(r x k) with r << d", "Keeps pre-trained W_0 frozen and trains only low-rank matrices A and B"],
            "ans": "LoRA freezes original weights W_0 and injects trainable rank-decomposition matrices delta_W = B * A with rank r << min(d, k), reducing trainable parameters by 10,000x [LoRA, p. 3]."
        },
        {
            "id": "Q062", "cat": "Optimization",
            "q": "What is QLoRA and how does it enable 65B model fine-tuning on a single 48GB GPU?",
            "docs": ["QLoRA: Efficient Finetuning of Quantized LLMs"],
            "pages": [1, 2],
            "facts": ["4-bit NormalFloat (NF4) quantization for base model weights", "Double quantization and paged optimizers to eliminate memory spikes"],
            "ans": "QLoRA quantizes the base model to 4-bit NormalFloat (NF4), applies Double Quantization, and uses Paged Optimizers to manage memory spikes during backward passes [QLoRA, p. 2]."
        },
        {
            "id": "Q063", "cat": "Optimization",
            "q": "How does the AdamW optimizer differ from the original Adam optimizer regarding weight decay?",
            "docs": ["Decoupled Weight Decay Regularization"],
            "pages": [1, 4],
            "facts": ["Adam coupled L2 regularization with adaptive gradient moments", "AdamW decouples weight decay directly from gradient update steps: theta_{t+1} = theta_t - lr * lambda * theta_t - lr * m_hat / (sqrt(v_hat) + eps)"],
            "ans": "AdamW decouples weight decay from the gradient update calculations, applying shrinkage directly to weights rather than treating it as an L2 penalty on gradient moments [Decoupled Weight Decay, p. 4]."
        },
        {
            "id": "Q064", "cat": "Optimization",
            "q": "What is the primary benefit of mixed-precision training (FP16 or BF16) over standard single-precision (FP32)?",
            "docs": ["Mixed Precision Training"],
            "pages": [2, 3],
            "facts": ["Reduces GPU memory bandwidth consumption by 50%", "Accelerates matrix multiplication using Tensor Cores while maintaining FP32 master weights for numerical stability"],
            "ans": "Mixed precision training halves memory storage and bandwidth requirements, speeding up computation on Tensor Cores while retaining FP32 master weights for numerical stability [Mixed Precision Training, p. 3]."
        },
        {
            "id": "Q065", "cat": "Optimization",
            "q": "How does DeepSpeed ZeRO (Zero Redundancy Optimizer) partition training states across data-parallel processes?",
            "docs": ["ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"],
            "pages": [1, 3],
            "facts": ["ZeRO-Stage 1: Partitions optimizer states", "ZeRO-Stage 2: Partitions gradients; ZeRO-Stage 3: Partitions model parameters across GPUs"],
            "ans": "ZeRO eliminates memory redundancy across data-parallel ranks by partitioning optimizer states (Stage 1), gradients (Stage 2), and model parameters (Stage 3) [ZeRO, p. 3]."
        },
        {
            "id": "Q066", "cat": "Optimization",
            "q": "What is Post-Training Quantization (PTQ) versus Quantization-Aware Training (QAT)?",
            "docs": ["A Survey of Quantization Methods for Efficient Neural Network Inference"],
            "pages": [3, 5],
            "facts": ["PTQ converts FP32 weights to INT8 using calibration data without re-training", "QAT simulates quantization errors in forward passes during training using fake quantization"],
            "ans": "PTQ quantizes weights and activations post-hoc without retraining, whereas QAT models quantization rounding noise during training for superior recovery of accuracy [Survey of Quantization Methods, p. 5]."
        },
        {
            "id": "Q067", "cat": "Optimization",
            "q": "How does Knowledge Distillation transfer competence from a large teacher to a compact student?",
            "docs": ["Distilling the Knowledge in a Neural Network"],
            "pages": [1, 2],
            "facts": ["Trains student model on soft probabilities produced by teacher model using temperature scaling", "Soft targets capture dark knowledge and inter-class correlations hidden by one-hot labels"],
            "ans": "Distillation trains a student network on high-temperature soft probability distributions from a teacher model, capturing dark knowledge and inter-class similarities [Distilling the Knowledge, p. 2]."
        },
        {
            "id": "Q068", "cat": "Optimization",
            "q": "Why does gradient checkpointing (activation checkpointing) trade computation for GPU memory?",
            "docs": ["Training Deep Nets with Sublinear Memory Cost"],
            "pages": [2, 3],
            "facts": ["Discards intermediate layer activations during the forward pass", "Recomputes activations on-the-fly during the backward pass, reducing memory from O(N) to O(sqrt(N))"],
            "ans": "Gradient checkpointing discards forward-pass activations and recomputes them during backpropagation, reducing memory footprint to O(sqrt(N)) at the cost of ~33% additional computation [Training Deep Nets with Sublinear Memory, p. 2]."
        },
        {
            "id": "Q069", "cat": "Optimization",
            "q": "What is the role of gradient clipping in stabilizing deep network training?",
            "docs": ["On the difficulty of training recurrent neural networks"],
            "pages": [3, 5],
            "facts": ["Rescales parameter gradients when their L2 norm exceeds a threshold: g <- g * (threshold / ||g||)", "Prevents gradient explosion in recurrent and deep architectures"],
            "ans": "Gradient clipping scales back update vectors when their L2 norm exceeds a defined threshold, preventing disastrous parameter destabilization from exploding gradients [On the difficulty of training RNNs, p. 5]."
        },
        {
            "id": "Q070", "cat": "Optimization",
            "q": "How does weight decay act as a regularizer during SGD and Adam training?",
            "docs": ["Decoupled Weight Decay Regularization"],
            "pages": [2, 3],
            "facts": ["Penalizes large weight magnitudes, shrinking parameters toward zero", "Prevents model capacity from overfitting to spurious sample noise"],
            "ans": "Weight decay shrinks parameter magnitudes continuously, penalizing complex over-parameterized representations and promoting generalizability [Decoupled Weight Decay, p. 2]."
        },
        {
            "id": "Q071", "cat": "Optimization",
            "q": "What is FlashDecoding and how does it accelerate generation with long prompts?",
            "docs": ["FlashDecoding: Fast Attention Decoding for Long Sequences"],
            "pages": [1, 2],
            "facts": ["Parallelizes attention reduction across the sequence dimension for each generation step", "Increases GPU core saturation during single-token decoding on long context"],
            "ans": "FlashDecoding splits key-value cache sequences into parallel chunks across GPU thread blocks, overcoming memory bandwidth saturation during long-context decoding [FlashDecoding, p. 2]."
        },
        {
            "id": "Q072", "cat": "Optimization",
            "q": "How does AWQ (Activation-aware Weight Quantization) protect salient weights during INT4 compression?",
            "docs": ["AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"],
            "pages": [1, 3],
            "facts": ["Observes that only top 1% of weights corresponding to large activation channels are critical", "Scales salient weight channels prior to per-channel quantization to preserve output accuracy"],
            "ans": "AWQ identifies that weights associated with high-magnitude activation channels are disproportionately critical and applies channel-wise scaling to safeguard them from quantization distortion [AWQ, p. 3]."
        },
        {
            "id": "Q073", "cat": "Optimization",
            "q": "What is speculative decoding and how does it achieve lossless latency reduction?",
            "docs": ["Fast Inference from Large Language Models via Speculative Decoding"],
            "pages": [1, 3],
            "facts": ["A smaller draft model generates K speculative candidate tokens quickly", "The target model verifies all K tokens in parallel in a single forward pass"],
            "ans": "Speculative decoding uses a lightweight draft model to generate candidate token sequences which are then verified in parallel by the target model in a single execution step [Speculative Decoding, p. 3]."
        },
        {
            "id": "Q074", "cat": "Optimization",
            "q": "What causes catastrophic forgetting during continued domain pre-training of LLMs?",
            "docs": ["Catastrophic Forgetting in Connectionist Networks", "Continual Pre-training for Large Language Models"],
            "pages": [1, 4],
            "facts": ["Updating weights on new domain data overwrites neural representations learned from general data", "Mitigated by mixing 10-20% general replay data during continual fine-tuning"],
            "ans": "Catastrophic forgetting occurs when gradient updates on a specialized corpus overwrite parameters supporting general linguistic reasoning; it is countered with replay buffers [Continual Pre-training for LLMs, p. 4]."
        },
        {
            "id": "Q075", "cat": "Optimization",
            "q": "How does cosine annealing with warm restarts modulate learning rates over training epochs?",
            "docs": ["SGDR: Stochastic Gradient Descent with Warm Restarts"],
            "pages": [2, 4],
            "facts": ["Decays learning rate following a half-cosine curve and abruptly resets it to initial maximum", "Enables escaping sharp local minima and exploring multiple diverse basins"],
            "ans": "Cosine annealing decreases the learning rate along a cosine curve before periodically resetting it to the maximum, enabling escape from sub-optimal local minima [SGDR, p. 4]."
        },
        {
            "id": "Q076", "cat": "Optimization",
            "q": "What is the difference between batch normalization and layer normalization in sequence models?",
            "docs": ["Layer Normalization"],
            "pages": [1, 3],
            "facts": ["BatchNorm normalizes across the batch dimension for each channel independently", "LayerNorm normalizes across all features for each individual sample independently"],
            "ans": "BatchNorm normalizes across the batch dimension, which fluctuates with dynamic sequence lengths; LayerNorm computes statistics across features within each individual token [Layer Normalization, p. 2]."
        },
        {
            "id": "Q077", "cat": "Optimization",
            "q": "How does early stopping based on validation loss prevent overfitting?",
            "docs": ["Early Stopping - But When?"],
            "pages": [2, 3],
            "facts": ["Monitors validation metric each epoch; halts training when validation loss fails to improve for patience epochs", "Saves model weights corresponding to peak generalization checkpoint"],
            "ans": "Early stopping halts training when validation loss stops improving for a specified patience interval, preserving the model checkpoint before generalization degrades [Early Stopping, p. 3]."
        },
        {
            "id": "Q078", "cat": "Optimization",
            "q": "What advantages does bfloat16 (Brain Floating Point) provide over IEEE float16 in deep learning?",
            "docs": ["Bfloat16: The Secret to High Performance on Modern Accelerators"],
            "pages": [1, 2],
            "facts": ["bfloat16 allocates 8 bits for exponent (matching FP32) and 7 bits for mantissa", "FP16 allocates 5 bits for exponent and 10 bits for mantissa, making it prone to overflow"],
            "ans": "bfloat16 shares the 8-bit dynamic range of FP32, eliminating underflow and overflow issues common to float16 during large gradient computations [Bfloat16 for Accelerators, p. 2]."
        },
        {
            "id": "Q079", "cat": "Optimization",
            "q": "How does label smoothing regularize classification models?",
            "docs": ["Rethinking the Inception Architecture for Computer Vision"],
            "pages": [5, 6],
            "facts": ["Replaces hard one-hot targets with a mixture of true label (1 - eps) and uniform distribution eps/K", "Prevents the model from becoming overconfident in its logit predictions"],
            "ans": "Label smoothing blends hard one-hot targets with a uniform distribution over all classes, discouraging extreme logit differentials and overconfidence [Rethinking Inception, p. 6]."
        },
        {
            "id": "Q080", "cat": "Optimization",
            "q": "What is prompt tuning and how does it compare to parameter fine-tuning?",
            "docs": ["The Power of Scale for Parameter-Efficient Prompt Tuning"],
            "pages": [1, 3],
            "facts": ["Prepends a sequence of continuous learnable virtual token vectors to frozen LLM inputs", "Achieves full model fine-tuning quality as model parameter scale increases beyond 10B"],
            "ans": "Prompt tuning optimizes continuous prefix prompt tokens while keeping all core model weights frozen, matching full fine-tuning performance at large model scales [The Power of Scale for Prompt Tuning, p. 3]."
        },

        # --- Category 5: Multi-Paper Comparison, Cross-Study Findings & Gaps (Q081 - Q105) ---
        {
            "id": "Q081", "cat": "Multi-Paper Analysis",
            "q": "Compare the model architectures used by Attention Is All You Need and Vision Transformer (ViT).",
            "docs": ["Attention Is All You Need", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"],
            "pages": [2, 3],
            "facts": ["Attention Is All You Need uses an encoder-decoder architecture for sequence-to-sequence translation", "ViT uses an encoder-only architecture over flattened image patches for image classification"],
            "ans": "While Attention Is All You Need uses an encoder-decoder design for seq2seq translation [Attention Is All You Need, p. 2], ViT adapts the encoder-only backbone to process 16x16 image patches for vision classification [An Image is Worth 16x16 Words, p. 3]."
        },
        {
            "id": "Q082", "cat": "Multi-Paper Analysis",
            "q": "How do the experimental results of ViT on ImageNet compare when trained from scratch versus pre-trained on JFT-300M?",
            "docs": ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"],
            "pages": [4, 5],
            "facts": ["When trained from scratch on ImageNet-1K, ViT achieves lower accuracy than ResNet baselines", "When pre-trained on JFT-300M, ViT substantially outperforms state-of-the-art CNNs"],
            "ans": "ViT underperforms ResNet baselines when trained on mid-sized ImageNet due to lack of inductive bias, but outperforms CNNs when pre-trained on massive JFT-300M [An Image is Worth 16x16 Words, p. 5]."
        },
        {
            "id": "Q083", "cat": "Multi-Paper Analysis",
            "q": "Compare the memory savings achieved by LoRA versus QLoRA.",
            "docs": ["LoRA: Low-Rank Adaptation of Large Language Models", "QLoRA: Efficient Finetuning of Quantized LLMs"],
            "pages": [3, 2],
            "facts": ["LoRA reduces trainable parameter memory by 10,000x but keeps base weights in 16-bit", "QLoRA compresses base weights to 4-bit NormalFloat, reducing base weight memory footprint by 4x"],
            "ans": "LoRA freezes 16-bit weights and introduces low-rank updates [LoRA, p. 3], whereas QLoRA compresses the base weights to 4-bit NF4, enabling 65B model fine-tuning on a single consumer GPU [QLoRA, p. 2]."
        },
        {
            "id": "Q084", "cat": "Multi-Paper Analysis",
            "q": "What contradictory conclusions do studies reach regarding the necessity of recurrence for sequence modeling?",
            "docs": ["Attention Is All You Need", "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition"],
            "pages": [1, 2],
            "facts": ["Attention Is All You Need argues recurrence is entirely unnecessary for sequence modeling", "DeepConvLSTM argues recurrence remains crucial for continuous temporal sensory streaming"],
            "ans": "Vaswani et al. demonstrate that recurrence can be entirely discarded in favor of attention [Attention Is All You Need, p. 1], whereas DeepConvLSTM demonstrates that recurrent LSTM layers remain essential for temporal activity continuity [DeepConvLSTM, p. 2]."
        },
        {
            "id": "Q085", "cat": "Multi-Paper Analysis",
            "q": "Compare the data requirements of CNNs and Vision Transformers as reported in literature.",
            "docs": ["Deep Residual Learning for Image Recognition", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"],
            "pages": [3, 4],
            "facts": ["CNNs generalize effectively on smaller datasets (ImageNet-1K) due to inductive bias", "ViT requires 100M+ images to exceed CNN performance"],
            "ans": "ResNet CNNs generalize reliably on moderate datasets like ImageNet [Deep Residual Learning, p. 3], whereas ViTs require hundred-million scale corpora to supersede convolutional accuracy [An Image is Worth 16x16 Words, p. 4]."
        },
        {
            "id": "Q086", "cat": "Multi-Paper Analysis",
            "q": "How do Dense Passage Retrieval (DPR) and BM25 compare on top-20 passage retrieval accuracy?",
            "docs": ["Dense Passage Retrieval for Open-Domain Question Answering"],
            "pages": [1, 5],
            "facts": ["DPR outperforms BM25 by 9-19% on top-20 passage retrieval accuracy across multiple benchmarks", "BM25 maintains an advantage on specific niche entities with rare spelling"],
            "ans": "DPR outperforms BM25 by 9-19% on top-20 passage retrieval across open-domain QA datasets, although BM25 excels on rare entities [Dense Passage Retrieval, p. 5]."
        },
        {
            "id": "Q087", "cat": "Multi-Paper Analysis",
            "q": "What are the common research gaps identified across modern retrieval-augmented generation systems?",
            "docs": ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "Lost in the Middle: How Language Models Use Long Contexts"],
            "pages": [5, 3],
            "facts": ["Sensitivity to context ordering and position of relevant facts", "Handling conflicting or contradictory evidence retrieved from multiple sources"],
            "ans": "Persistent research gaps include degradation when evidence is placed in mid-context [Lost in the Middle, p. 3] and brittle handling of contradictory source passages [RAG for Knowledge-Intensive NLP, p. 5]."
        },
        {
            "id": "Q088", "cat": "Multi-Paper Analysis",
            "q": "Compare the computational efficiency of FlashAttention versus standard PyTorch scaled dot-product attention.",
            "docs": ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "Attention Is All You Need"],
            "pages": [1, 4],
            "facts": ["Standard attention requires O(N^2) HBM memory writes and reads", "FlashAttention yields 2-4x wall-clock speedup and linear memory scaling with sequence length"],
            "ans": "Standard attention incurs costly O(N^2) memory transfers [Attention Is All You Need, p. 4], whereas FlashAttention achieves 2-4x speedups and reduces memory footprint to linear in sequence length [FlashAttention, p. 1]."
        },
        {
            "id": "Q089", "cat": "Multi-Paper Analysis",
            "q": "How do Swin Transformer and original ViT differ in computational scaling relative to image resolution?",
            "docs": ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"],
            "pages": [2, 3],
            "facts": ["ViT has quadratic computational complexity relative to image resolution", "Swin Transformer has linear computational complexity due to local window self-attention"],
            "ans": "ViT scales quadratically with image resolution [An Image is Worth 16x16 Words, p. 3], while Swin Transformer achieves linear complexity by computing attention within shifted local windows [Swin Transformer, p. 2]."
        },
        {
            "id": "Q090", "cat": "Multi-Paper Analysis",
            "q": "Compare the tokenization approaches between original Transformer (BPE/WordPiece) and character-level models.",
            "docs": ["Attention Is All You Need"],
            "pages": [5, 7],
            "facts": ["Uses byte-pair encoding (BPE) with shared vocabulary of 37,000 tokens", "Balances subword representation efficiency with open-vocabulary coverage"],
            "ans": "The Transformer uses byte-pair encoding with 37,000 subwords, avoiding out-of-vocabulary errors while avoiding the excessive sequence expansion of character models [Attention Is All You Need, p. 5]."
        },
        {
            "id": "Q091", "cat": "Multi-Paper Analysis",
            "q": "How does ZeRO Stage 3 parameter partitioning compare to Megatron-LM tensor model parallelism?",
            "docs": ["ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"],
            "pages": [3, 4],
            "facts": ["Megatron splits weight matrices across GPUs within each layer, requiring frequent all-reduce communication", "ZeRO-3 gathers parameters dynamically per forward/backward pass and discards them immediately"],
            "ans": "Megatron splits weight tensors across GPUs requiring high-bandwidth all-reduce steps, while ZeRO-3 broadcasts full parameters on-demand and discards them after execution [ZeRO, p. 4]."
        },
        {
            "id": "Q092", "cat": "Multi-Paper Analysis",
            "q": "What methodological differences explain performance discrepancies between wrist and waist IMU activity tracking?",
            "docs": ["Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [5, 6],
            "facts": ["Waist is centered on human biomechanical torso for locomotion", "Wrist moves independently during conversation or arm gestures causing false motion signals"],
            "ans": "Discrepancies arise because waist sensors track torso locomotion kinematics directly, while wrist sensors suffer from uncoupled arm gestures [Deep Learning for Sensor-based HAR, p. 6]."
        },
        {
            "id": "Q093", "cat": "Multi-Paper Analysis",
            "q": "Compare the parameter efficiency of prefix tuning, prompt tuning, and LoRA.",
            "docs": ["LoRA: Low-Rank Adaptation of Large Language Models", "The Power of Scale for Parameter-Efficient Prompt Tuning"],
            "pages": [2, 3],
            "facts": ["Prefix tuning optimizes virtual tokens across all attention layers", "LoRA optimizes rank-decomposition matrices across attention projection weights"],
            "ans": "Prefix tuning inserts continuous prompt activations across all transformer layers, while LoRA modifies linear projection weights with low-rank decomposition matrices [LoRA, p. 2]."
        },
        {
            "id": "Q094", "cat": "Multi-Paper Analysis",
            "q": "How do RAG and long-context LLMs compare in efficiency for question answering over 50+ documents?",
            "docs": ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "Lost in the Middle: How Language Models Use Long Contexts"],
            "pages": [2, 4],
            "facts": ["RAG retrieves only relevant chunks, minimizing token compute cost and latency", "Long-context models feed all pages into prompt, incurring quadratic or linear attention costs and middle-loss degradation"],
            "ans": "RAG selectively indexes and filters relevant chunks to limit inference cost, whereas brute-force long-context feeding suffers from high token latency and middle-position degradation [Lost in the Middle, p. 4]."
        },
        {
            "id": "Q095", "cat": "Multi-Paper Analysis",
            "q": "What conflicting findings exist regarding the effectiveness of post-training INT4 quantization on reasoning benchmarks?",
            "docs": ["AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", "A Survey of Quantization Methods for Efficient Neural Network Inference"],
            "pages": [1, 4],
            "facts": ["Standard round-to-nearest INT4 causes noticeable perplexity degradation on reasoning tasks", "Activation-aware methods like AWQ retain near-lossless reasoning capability"],
            "ans": "Naive INT4 quantization severely degrades reasoning benchmarks [Survey of Quantization, p. 4], whereas activation-aware quantization protects critical weights and preserves accuracy [AWQ, p. 1]."
        },
        {
            "id": "Q096", "cat": "Multi-Paper Analysis",
            "q": "Compare the pre-training objectives of BERT (MLM), GPT (causal LM), and T5 (span corruption).",
            "docs": ["Attention Is All You Need", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"],
            "pages": [2, 3],
            "facts": ["BERT masks random tokens and predicts bidirectionally", "GPT predicts next token autoregressively; T5 masks contiguous spans and reconstructs them via encoder-decoder"],
            "ans": "BERT optimizes masked language modeling bidirectionally, GPT trains autoregressively left-to-right, and T5 uses span corruption in an encoder-decoder framework [RAG for Knowledge-Intensive NLP, p. 2]."
        },
        {
            "id": "Q097", "cat": "Multi-Paper Analysis",
            "q": "How do DeepConvLSTM and TinyHAR compare in computational overhead for wearable deployment?",
            "docs": ["Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition", "TinyHAR: A Lightweight Deep Learning Model for Human Activity Recognition"],
            "pages": [3, 4],
            "facts": ["DeepConvLSTM requires sequential recurrence and substantial memory for hidden states", "TinyHAR uses depthwise separable convolutions to cut parameter count by 80%"],
            "ans": "DeepConvLSTM incurs high memory overhead from recurrent cell state propagation [DeepConvLSTM, p. 3], whereas TinyHAR replaces heavy layers with depthwise convolutions for edge microcontrollers [TinyHAR, p. 4]."
        },
        {
            "id": "Q098", "cat": "Multi-Paper Analysis",
            "q": "What are the limitations of standard BLEU metric as documented in translation and synthesis literature?",
            "docs": ["Attention Is All You Need"],
            "pages": [8],
            "facts": ["BLEU measures n-gram precision without accounting for semantic equivalence or fluency", "Penalizes valid paraphrases that do not match reference phrasing exactly"],
            "ans": "BLEU relies on superficial exact n-gram overlap, penalizing valid semantic paraphrases and failing to measure conceptual accuracy [Attention Is All You Need, p. 8]."
        },
        {
            "id": "Q099", "cat": "Multi-Paper Analysis",
            "q": "Compare the scaling laws of model size versus dataset size in deep learning.",
            "docs": ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"],
            "pages": [4, 6],
            "facts": ["Scaling model capacity yields diminishing returns unless accompanied by proportional training dataset scaling", "Large models with insufficient data overfit rapidly"],
            "ans": "Model performance scales smoothly with parameter count only when training data volume is expanded concurrently to prevent sample saturation [An Image is Worth 16x16 Words, p. 6]."
        },
        {
            "id": "Q100", "cat": "Multi-Paper Analysis",
            "q": "How do attention maps visualize model interpretability across research papers?",
            "docs": ["Attention Is All You Need", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"],
            "pages": [6, 4],
            "facts": ["Attention weights highlight which input words or image patches the model focuses on for each prediction", "Demonstrates syntactic dependencies in NLP and object localization in vision"],
            "ans": "Plotting multi-head attention weights reveals grammatical dependencies in text [Attention Is All You Need, p. 6] and semantic object localization in image patches [An Image is Worth 16x16 Words, p. 4]."
        },
        {
            "id": "Q101", "cat": "Multi-Paper Analysis",
            "q": "What is the role of temperature scaling in calibrating neural network confidence?",
            "docs": ["Distilling the Knowledge in a Neural Network"],
            "pages": [2],
            "facts": ["Dividing logits by temperature T > 1 smooths probability distribution across classes", "Exposes relative probabilities of incorrect classes without altering argmax ranking"],
            "ans": "Dividing logits by temperature T > 1 softens output probability distributions, exposing relative ranking distributions across non-top classes [Distilling the Knowledge, p. 2]."
        },
        {
            "id": "Q102", "cat": "Multi-Paper Analysis",
            "q": "Compare the gradient backpropagation stability of Deep Residual Networks versus early deep feedforward networks.",
            "docs": ["Deep Residual Learning for Image Recognition"],
            "pages": [1, 2],
            "facts": ["Identity shortcut connections allow gradients to flow directly through skip paths without vanishing", "Enabled successful training of networks with over 100-1000 layers"],
            "ans": "Residual connections provide direct identity shortcuts, allowing gradients to propagate backward through hundreds of layers without vanishing [Deep Residual Learning, p. 2]."
        },
        {
            "id": "Q103", "cat": "Multi-Paper Analysis",
            "q": "How does cross-entropy loss with teacher forcing operate during seq2seq training?",
            "docs": ["Attention Is All You Need"],
            "pages": [3, 7],
            "facts": ["Ground-truth target tokens are fed as inputs to the decoder regardless of model predictions", "Prevents error propagation during early training epochs"],
            "ans": "Teacher forcing supplies ground-truth target tokens as decoder input at step t, preventing early decoding errors from cascading across the sequence [Attention Is All You Need, p. 3]."
        },
        {
            "id": "Q104", "cat": "Multi-Paper Analysis",
            "q": "Compare the strengths and limitations of offline vector indexing versus online real-time incremental vector insertion.",
            "docs": ["FAISS: A Library for Efficient Similarity Search"],
            "pages": [3, 5],
            "facts": ["Offline batch indexing optimizes clustering centroids and quantization codebooks", "Online incremental indexing allows instant ingestion without full rebuilds but can lead to index fragmentation"],
            "ans": "Offline indexing produces optimal cluster partitions and compression, while incremental IndexIDMap insertion provides immediate availability with minor fragmentation [FAISS, p. 3]."
        },
        {
            "id": "Q105", "cat": "Multi-Paper Analysis",
            "q": "What strategies are effective for synthesizing research findings when indexed papers report conflicting experimental outcomes?",
            "docs": ["Seven Failure Points When Engineering a Retrieval Augmented Generation System", "Deep Learning for Sensor-based Human Activity Recognition: A Survey"],
            "pages": [3, 8],
            "facts": ["Dissect differences in benchmark datasets, sensor positions, metrics, and assumptions", "Attribute claims explicitly to respective papers with page citations rather than asserting a single consensus"],
            "ans": "Effective synthesis explicitly cites both divergent claims with their respective source pages and clarifies whether differences stem from datasets, sensors, or evaluation assumptions [Seven Failure Points, p. 3]."
        }
    ]

    dataset = []
    for q in raw_questions:
        dataset.append(EvaluationQuestion(
            question_id=q["id"],
            question=q["q"],
            category=q["cat"],
            expected_document_titles=q["docs"],
            expected_pages=q["pages"],
            expected_key_facts=q["facts"],
            reference_answer=q["ans"]
        ))
    return dataset


def save_evaluation_dataset_to_disk(file_path: Optional[Path] = None) -> Path:
    """Persist the 105 evaluation benchmark items to JSON on disk."""
    settings = get_settings()
    out_path = file_path or (settings.evaluation_dir / "benchmark_100_questions.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = get_curated_evaluation_dataset()
    data = [item.model_dump() for item in dataset]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out_path

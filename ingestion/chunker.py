"""Semantic and token-aware text chunker with precise cross-page provenance tracking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional
from ingestion.text_extractor import PageData


@dataclass
class TextChunk:
    chunk_id: str
    document_id: str
    document_title: str
    document_filename: str
    chunk_index: int
    text: str
    token_count: int
    start_page: int
    end_page: int
    section_heading: Optional[str] = None


class SemanticChunker:
    """Chunks multi-page extracted text while tracking page provenance and headings."""

    def __init__(
        self,
        chunk_size_tokens: int = 700,
        chunk_overlap_tokens: int = 100,
        min_chunk_tokens: int = 50
    ) -> None:
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using whitespace and punctuation heuristics (~1.3 tokens/word)."""
        words = len(text.split())
        return max(1, int(words * 1.3))

    def chunk_pages(
        self,
        pages: List[PageData],
        document_id: str,
        document_title: str,
        document_filename: str
    ) -> List[TextChunk]:
        """Split a sequence of pages into overlapping chunks with page provenance."""
        if not pages:
            return []

        # Step 1: Build a stream of atomic segments (sentences or paragraphs) tagged with page & heading
        segments: List[dict] = []
        for p in pages:
            if not p.clean_text.strip():
                continue

            active_heading = p.headings[0] if p.headings else None

            # Split page text into paragraphs/sentences
            paragraphs = re.split(r"\n\s*\n", p.clean_text)
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Further split long paragraphs by sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    tokens = self.estimate_tokens(sent)
                    segments.append({
                        "text": sent,
                        "tokens": tokens,
                        "page": p.page_number,
                        "heading": active_heading
                    })

        if not segments:
            return []

        chunks: List[TextChunk] = []
        chunk_idx = 0
        seg_idx = 0
        n_segs = len(segments)

        while seg_idx < n_segs:
            curr_text_parts: List[str] = []
            curr_tokens = 0
            start_page = segments[seg_idx]["page"]
            end_page = start_page
            active_heading = segments[seg_idx]["heading"]

            end_seg_idx = seg_idx
            while end_seg_idx < n_segs:
                seg = segments[end_seg_idx]
                # If adding this segment exceeds chunk_size_tokens (and we already have some text)
                if curr_tokens + seg["tokens"] > self.chunk_size_tokens and curr_text_parts:
                    break

                curr_text_parts.append(seg["text"])
                curr_tokens += seg["tokens"]
                end_page = seg["page"]
                if seg["heading"] and not active_heading:
                    active_heading = seg["heading"]
                end_seg_idx += 1

            combined_text = " ".join(curr_text_parts).strip()

            # Ensure minimum token threshold unless it is the last remaining segment
            if curr_tokens >= self.min_chunk_tokens or (end_seg_idx >= n_segs and chunks == []):
                chunk_id = f"{document_id[:8]}_p{start_page}-{end_page}_c{chunk_idx:04d}"
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_title=document_title,
                    document_filename=document_filename,
                    chunk_index=chunk_idx,
                    text=combined_text,
                    token_count=curr_tokens,
                    start_page=start_page,
                    end_page=end_page,
                    section_heading=active_heading
                ))
                chunk_idx += 1

            if end_seg_idx >= n_segs:
                break

            # Calculate overlap: step back until overlap tokens are reached
            step_back_tokens = 0
            back_idx = end_seg_idx - 1
            while back_idx > seg_idx and (step_back_tokens + segments[back_idx]["tokens"]) <= self.chunk_overlap_tokens:
                step_back_tokens += segments[back_idx]["tokens"]
                back_idx -= 1

            # Next chunk starts from back_idx + 1 (or at least advance by 1 to guarantee forward progress)
            next_seg_idx = max(seg_idx + 1, back_idx + 1)
            seg_idx = next_seg_idx

        return chunks

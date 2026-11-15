"""
rag_engine.py
Retrieval-Augmented Generation: combines semantic search results with
the Gemini API to produce cited, grounded, student-friendly answers.
Includes an intelligent local extractive mode when no API key is supplied.
"""

import os
import re
from typing import List, Tuple, Optional
import google.generativeai as genai
from pdf_processor import PageChunk

SYSTEM_PROMPT = """\
You are EduPolicy AI, an intelligent, student-friendly academic policy advisor.
Your job is to read official academic policy documents and explain the regulations in simple, accessible, plain English that any student can immediately understand.

STRICT INSTRUCTIONS:
1. ANSWER THE SPECIFIC QUESTION DIRECTLY: Address exactly what the student asked. Do not summarize the entire regulation or provide historical/administrative background if only one specific rule was asked.
2. CONCISE & STUDENT-FRIENDLY: Explain the rule in simple, crystal-clear, plain English. Never copy-paste lengthy paragraphs or regurgitate complex legalistic jargon verbatim.
3. BULLET FORMAT (2 TO 5 BULLETS): Format your core answer as 2 to 5 short, direct, actionable bullet points.
4. HIGHLIGHT CRITICAL DETAILS: Use **bold** markdown to emphasize essential keywords, minimum requirements, GPA/marks, deadlines, credit caps, eligibility criteria, penalties, and exceptions (e.g., **85% attendance**, **within 5 business days**, **CGPA below 2.00**, **FA grade**).
5. CITATION MANDATE: Always cite the source document name and original page number (e.g., [DocumentName, p. 3]) for the policies you cite.
6. ZERO HALLUCINATION: Rely ONLY on the provided context passages. If the provided documents do not contain the answer, explicitly state: "The provided academic policy documents do not contain information regarding this question."
7. NO REPETITION OR FLUFF: Avoid introductory filler phrases like "Based on the provided documents..." or long bureaucratic preambles. Jump directly into the concise policy explanation.
"""


def highlight_policy_terms(text: str) -> str:
    """Highlight numbers, percentages, GPAs, deadlines, and key regulatory terms."""
    patterns = [
        r'\b\d+(?:\.\d+)?%',                                   # percentages: 85%, 15%
        r'\b[0-4]\.\d{2}\b',                                  # GPAs: 2.00, 3.75, 4.00
        r'\b(?:CGPA|GPA|FA|LOA|XF)\b',                        # academic acronyms
        r'\b\d+\s+(?:calendar\s+days?|business\s+days?|working\s+days?|days?|weeks?|semesters?|credit\s+hours?|hours?)\b', # durations
        r'\b(?:Academic Probation|Academic Suspension|Academic Warning|Good Academic Standing)\b',
        r'\b(?:Dean\'s Honor Roll|Dean\'s List|Cum Laude|Magna Cum Laude|Summa Cum Laude)\b',
        r'\b(?:strictly prohibited|prohibited|mandatory|compulsory|automatic failure|automatic zero|unexcused absences)\b',
    ]
    for pat in patterns:
        text = re.sub(
            pat,
            lambda m: f"**{m.group(0)}**" if not m.group(0).startswith("**") else m.group(0),
            text,
            flags=re.IGNORECASE,
        )
    # Clean any accidental redundant markdown asterisks
    text = re.sub(r'\*{3,}', '**', text)
    return text


def extract_concise_policy_rules(text: str, query: str = "", max_bullets: int = 4) -> List[str]:
    """
    Extracts complete, student-friendly policy rule bullets from a chunk's text,
    filtering administrative noise and ranking by query relevance.
    """
    # Remove document headers and administrative banners
    clean_text = re.sub(r'OFFICIAL ACADEMIC REGULATIONS[^\n]*\n', '', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'SECTION \d+:[^\n]*\n', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'Office of the Provost[^\n]*', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'Page \d+', '', clean_text, flags=re.IGNORECASE)

    # Split into coherent regulatory clauses
    raw_clauses = re.split(
        r'(?:\n\s*(?=\d+\.\d+|\b[A-Z0-9\s]{3,}:)|(?<=\n)\s*[-•*]\s*)',
        clean_text,
    )

    candidates = []
    for c in raw_clauses:
        c = re.sub(r'\s+', ' ', c).strip()
        # Clean section numbering e.g. "1.1 Good Academic Standing: "
        c = re.sub(r'^\d+\.\d+\s+', '', c)
        if len(c) > 25 and not c.startswith("SECTION"):
            candidates.append(c)

    if not candidates:
        return ["Policy details available in source document."]

    # Score candidates against query keywords
    q_words = set(re.findall(r'\w+', query.lower())) if query else set()
    stop_words = {"what", "is", "the", "for", "a", "an", "and", "in", "to", "of", "how", "can", "i", "do", "are", "about"}
    meaningful_q_words = q_words - stop_words

    def score_candidate(cand: str) -> int:
        cand_lower = cand.lower()
        score = sum(3 for w in meaningful_q_words if w in cand_lower)
        if any(k in cand_lower for k in ["minimum", "maximum", "deadline", "penalty", "must", "required", "fail", "prohibited", "gpa"]):
            score += 1
        return score

    if meaningful_q_words:
        candidates.sort(key=score_candidate, reverse=True)

    selected = []
    for cand in candidates:
        # If very long, take the first 1-2 complete sentences
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', cand)
        if len(sents) > 1 and len(cand) > 180:
            cand = sents[0] + " " + sents[1]
        elif len(sents) > 1:
            cand = sents[0]

        cand = highlight_policy_terms(cand)
        if cand not in selected:
            selected.append(cand)
        if len(selected) >= max_bullets:
            break

    return selected


def _build_context_block(results: List[Tuple[PageChunk, float]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = []
    for idx, (chunk, score) in enumerate(results, start=1):
        lines.append(
            f"--- PASSAGE {idx} | Source: [{chunk.citation}] | Match Relevance: {score * 100:.1f}% ---\n"
            f"{chunk.text}\n"
        )
    return "\n".join(lines)


def _format_conversation_history(history: List[dict], max_turns: int = 3) -> str:
    """Format recent chat history for conversational context."""
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = ["Recent conversation context:"]
    for turn in recent:
        q = turn.get("q", "")
        a = turn.get("r", {}).get("answer", "")
        if q and a:
            truncated_a = a[:250] + "..." if len(a) > 250 else a
            lines.append(f"Student: {q}")
            lines.append(f"EduPolicy AI: {truncated_a}")
    return "\n".join(lines) + "\n\n"


class RAGEngine:
    """
    Wraps a SemanticSearchEngine and Gemini Generative AI to provide
    concise, student-friendly academic policy answers with source citations.
    """

    def __init__(
        self,
        search_engine,
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        top_k: int = 5,
    ):
        self.search_engine = search_engine
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.model_name = model_name
        self.top_k = top_k
        self.model = None

        if self.gemini_api_key:
            self._init_gemini()

    def _init_gemini(self):
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=SYSTEM_PROMPT,
            )
        except Exception:
            self.model = None

    def update_key(self, api_key: str, model_name: Optional[str] = None):
        """Update API key and reinitialize model."""
        self.gemini_api_key = api_key.strip()
        if model_name:
            self.model_name = model_name
        self._init_gemini()

    def answer(
        self,
        query: str,
        chat_history: Optional[List[dict]] = None,
        doc_filter: Optional[str] = None,
    ) -> dict:
        """
        Retrieve relevant chunks and generate a concise, cited, student-friendly answer.

        Returns a dictionary:
            - "answer"     : str (Concise bulleted answer text)
            - "sources"    : List[str] (Unique citations)
            - "passages"   : List[Tuple[PageChunk, float]]
            - "confidence" : float (Average match percentage of top 3)
            - "mode"       : "gemini" | "local_extractive"
        """
        results = self.search_engine.search(
            query, top_k=self.top_k, doc_filter=doc_filter
        )

        if not results:
            return {
                "answer": "The provided academic policy documents do not contain information regarding this question.",
                "sources": [],
                "passages": [],
                "confidence": 0.0,
                "mode": "none",
            }

        # Calculate average confidence of top 3
        top_scores = [score for _, score in results[:3]]
        avg_confidence = round(float(sum(top_scores) / len(top_scores)) * 100, 1)
        sources = list(dict.fromkeys(chunk.citation for chunk, _ in results))

        # -----------------------------------------------------------------
        # Mode 1: Gemini Generative AI (When API key configured)
        # -----------------------------------------------------------------
        if self.model:
            context_block = _build_context_block(results)
            history_context = _format_conversation_history(chat_history or [])

            full_prompt = (
                f"{history_context}"
                f"Official Academic Policy Context:\n\n"
                f"{context_block}\n\n"
                f"Student Question: {query}\n\n"
                "Instructions: Answer the specific question directly in 2 to 5 short, student-friendly bullet points. "
                "Highlight key numbers, percentages, GPAs, deadlines, and requirements in bold (**like this**). "
                "Cite your sources with [DocumentName, p. X] notation. Avoid lengthy copied paragraphs or legalistic jargon."
            )

            try:
                response = self.model.generate_content(full_prompt)
                answer_text = response.text.strip()
                return {
                    "answer": answer_text,
                    "sources": sources,
                    "passages": results,
                    "confidence": avg_confidence,
                    "mode": "gemini",
                }
            except Exception as e:
                err_msg = str(e)
                if "quota" in err_msg.lower() or "429" in err_msg:
                    notice = "⚠️ *Gemini API quota exceeded. Displaying concise policy explanation below:*\n\n"
                else:
                    notice = f"⚠️ *Gemini notice ({err_msg[:80]}). Displaying verified policy rules:*\n\n"
                extractive = self._generate_extractive_answer(results, query)
                return {
                    "answer": notice + extractive,
                    "sources": sources,
                    "passages": results,
                    "confidence": avg_confidence,
                    "mode": "local_extractive",
                }

        # -----------------------------------------------------------------
        # Mode 2: Local Extractive Synthesis (When no API key is supplied)
        # -----------------------------------------------------------------
        extractive = self._generate_extractive_answer(results, query)
        if "do not contain information regarding this question" in extractive:
            return {
                "answer": extractive,
                "sources": [],
                "passages": [],
                "confidence": 0.0,
                "mode": "none",
            }

        guidance = (
            "💡 *Student-Friendly Policy Explanation* (To enable conversational Gemini synthesis, add your Gemini API key in the sidebar):\n\n"
        )
        return {
            "answer": guidance + extractive,
            "sources": sources,
            "passages": results,
            "confidence": avg_confidence,
            "mode": "local_extractive",
        }

    def _generate_extractive_answer(
        self, results: List[Tuple[PageChunk, float]], query: str
    ) -> str:
        """
        Synthesizes 2 to 5 concise, student-friendly bullet points directly answering the query,
        avoiding raw paragraph dumps and highlighting key conditions.
        """
        if not results:
            return "The provided academic policy documents do not contain information regarding this question."

        # Filter out generic stop and meta words to check true topical relevance
        generic_meta_words = {
            "what", "is", "the", "for", "a", "an", "and", "in", "to", "of",
            "how", "can", "i", "do", "are", "about", "on", "my", "any", "this",
            "policy", "policies", "rule", "rules", "regulation", "regulations",
            "university", "academic", "handbook", "official", "documents", "document",
            "student", "students", "institution", "institutional"
        }
        top_score = results[0][1] if results else 0.0
        q_words = set(re.findall(r'\w+', query.lower())) - generic_meta_words
        combined_top_text = " ".join(c.text.lower() for c, _ in results[:2])
        keyword_overlap = [w for w in q_words if w in combined_top_text]

        # If similarity is low and no substantive topic words match
        if top_score < 0.32 or (q_words and not keyword_overlap):
            return "The provided academic policy documents do not contain information regarding this question."

        bullets = []
        seen_rules = set()

        # Extract top matching rules across the best 2-3 chunks
        for chunk, score in results[:3]:
            chunk_rules = extract_concise_policy_rules(chunk.text, query=query, max_bullets=2)
            for rule in chunk_rules:
                # Deduplicate similar statements
                simplified = re.sub(r'[^a-zA-Z0-9]', '', rule.lower())[:40]
                if simplified not in seen_rules:
                    seen_rules.add(simplified)
                    bullets.append(f"• {rule} `[{chunk.citation}]`")
                if len(bullets) >= 4:
                    break
            if len(bullets) >= 4:
                break

        if not bullets:
            return "The provided academic policy documents do not contain information regarding this question."

        lines = ["### 📋 **Key Policy Rules:**\n"]
        lines.extend(bullets)
        return "\n".join(lines)

# Hybrid AI Resume-to-JD Matcher & Skill Gap Analyzer

A 3-layer NLP & Generative AI candidate evaluation system combining deterministic keyword matching, $O(N+M)$ hash-set skill extraction, and structured-JSON LLM semantic reasoning.

## System Architecture (3-Layer Hybrid Scoring)
1. **Layer 1 — Deterministic ATS Similarity (20% Weight):** Vectorizes resume and job description text using `TfidfVectorizer(ngram_range=(1, 2))` and computes Cosine Similarity ($\frac{A \cdot B}{\|A\|\|B\|}$).
2. **Layer 2 — DSA Set-Based Skill Gap Engine (30% Weight):** Uses Python hash `set` intersection (`&`) and difference (`-`) for $O(N + M)$ exact keyword extraction and gap identification.
3. **Layer 3 — Generative AI Semantic Evaluator (50% Weight):** Uses the Google GenAI SDK with strict **Pydantic JSON Schema** enforcement and automatic multi-model fallback (`gemini-3.8-flash` $\rightarrow$ `gemini-4-hydrogen` $\rightarrow$ `gemini-3.5-flash-lite`) to detect transferable skills (e.g., *FastAPI* $\rightarrow$ *REST APIs*) and generate tailored resume bullets.

## Tech Stack
- **Language:** Python 3
- **NLP & ML:** Scikit-learn (`TfidfVectorizer`, `cosine_similarity`), Regular Expressions (`re`)
- **Generative AI:** Google GenAI SDK (`google-genai`), Pydantic (`BaseModel`)

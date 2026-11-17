import os
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

st.set_page_config(page_title="AI Resume & JD Matcher", layout="wide")
st.title("Smart AI Resume-to-JD Matcher & Skill Gap Analyzer")
st.write("Compare your resume against any job description using TF-IDF, Hash-Set Skill Matching, and Gemini AI.")

# Load API key safely from Streamlit Cloud Secrets
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

TECH_SKILLS = {
    "python", "c", "c++", "java", "sql", "sqlite", "fastapi", "streamlit",
    "pandas", "numpy", "scikit-learn", "git", "html", "css", "javascript",
    "docker", "rest apis", "machine learning", "deep learning", "llm apis",
    "generative ai", "gemini api", "data structures", "algorithms"
}

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def analyze_match(resume: str, jd: str):
    # 1. TF-IDF Cosine Similarity
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vec.fit_transform([clean_text(resume), clean_text(jd)])
    tfidf_score = round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 1)

    # 2. Hash-Set Skill Gap Analysis (O(N + M) time)
    res_pad, jd_pad = f" {clean_text(resume)} ", f" {clean_text(jd)} "
    jd_skills = {s for s in TECH_SKILLS if f" {s} " in jd_pad}
    res_skills = {s for s in TECH_SKILLS if f" {s} " in res_pad}
    matched = sorted(jd_skills & res_skills)
    missing = sorted(jd_skills - res_skills)
    skill_pct = round((len(matched) / len(jd_skills)) * 100, 1) if jd_skills else 0.0

    # 3. Gemini AI Semantic Feedback
    prompt = f"""
    Compare this Resume and Job Description briefly in 3 bullet points:
    1) Overall Fit Score (0-100%) and why
    2) Transferable skills found (e.g., FastAPI counts for REST APIs)
    3) Two stronger resume bullet points tailored to this JD.
    RESUME: {resume}
    JD: {jd}
    """
    ai_feedback = "Add your GEMINI_API_KEY in app secrets to enable AI feedback."
    for model_name in ["gemini-3.8-flash", "gemini-4-hydrogen", "gemini-3.5-flash-lite"]:
        try:
            res = client.models.generate_content(model=model_name, contents=prompt)
            ai_feedback = res.text
            break
        except Exception:
            continue

    return tfidf_score, skill_pct, matched, missing, ai_feedback

# --- User Interface ---
col1, col2 = st.columns(2)
with col1:
    resume_input = st.text_area("Paste Your Resume Text", height=220, placeholder="Paste your resume here...")
with col2:
    jd_input = st.text_area("Paste Job Description", height=220, placeholder="Paste the job description here...")

if st.button("Analyze Match & Skill Gaps", type="primary"):
    if not resume_input.strip() or not jd_input.strip():
        st.warning("Please paste both a Resume and a Job Description.")
    else:
        with st.spinner("Analyzing skills and generating AI feedback..."):
            tfidf_s, skill_s, matched, missing, feedback = analyze_match(resume_input, jd_input)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Exact Skill Match", f"{skill_s}%")
        m2.metric("TF-IDF Text Similarity", f"{tfidf_s}%")
        m3.metric("Skills Matched", f"{len(matched)} / {len(matched) + len(missing)}")

        st.success(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.error(f"**Missing Keywords:** {', '.join(missing) if missing else 'None'}")
        st.markdown("### AI Recruiter Feedback & Tailored Bullets")
        st.markdown(feedback)

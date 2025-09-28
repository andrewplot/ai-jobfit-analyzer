# app.py
import streamlit as st
import PyPDF2, io, re
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go

# ---------- Helpers ----------
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except Exception as e:
        return ""

def clean_text(text):
    text = text.replace("\n"," ").replace("\r"," ")
    text = re.sub(r'[^A-Za-z0-9\+\#\.\- ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def top_keywords(text, n=15):
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=2000)
    X = vec.fit_transform([text])
    feats = np.array(vec.get_feature_names_out())
    scores = X.toarray()[0]
    top_idx = scores.argsort()[::-1][:n]
    return [feats[i] for i in top_idx]

COMMON_SKILLS = [
 "Python","C++","C#","Java","JavaScript","SQL","NoSQL","TensorFlow","PyTorch","scikit-learn",
 "pandas","numpy","Docker","Kubernetes","AWS","GCP","Azure","REST","Flask","FastAPI","Streamlit",
 "Hugging Face","NLP","Computer Vision","Transformers","Spark","Kafka","Linux","Git","CI/CD",
 "Tableau","PowerBI","CUDA","ROS"
]

def match_skills(resume_text, skills_list=COMMON_SKILLS):
    r = resume_text.lower()
    present = [s for s in skills_list if s.lower() in r]
    missing = [s for s in skills_list if s not in present]
    return present, missing

@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def embedding_similarity_score(resume_text, jd_text, model):
    emb_r = model.encode(resume_text, convert_to_tensor=True)
    emb_j = model.encode(jd_text, convert_to_tensor=True)
    sim = float(util.cos_sim(emb_r, emb_j).item())
    score = max(0.0, min(100.0, sim * 100.0))
    return score

def keyword_coverage_score(top_kwds, resume_text):
    r = resume_text.lower()
    present_count = sum(1 for k in top_kwds if k.lower() in r)
    return (present_count / len(top_kwds)) * 100 if len(top_kwds)>0 else 100.0
    
def combined_score(semantic, skills_cov, keyword_cov, w_sem=0.6, w_sk=0.3, w_kw=0.1):
    return semantic*w_sem + skills_cov*w_sk + keyword_cov*w_kw

VERB = [""]

def recommendations_for_missing_skills(missing_skills):
    recs = []
    for s in missing_skills[:10]:
        recs.append(f"Add a concise resume bullet: 'Implemented a project using {s} to ... (quantify impact).'")
    return recs

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Resume-Job Matcher", layout="wide")
st.title("AI Resume ↔ Job Matcher")

model = load_model()

with st.sidebar:
    st.markdown("## Instructions")
    st.markdown("- Upload a PDF resume.")
    st.markdown("- Paste the job description text.")
    st.markdown("- Review scores, highlighted skills, and recommended bullets.")

col1, col2 = st.columns([1,2])

with col1:
    resume_file = st.file_uploader("Upload Resume (PDF)", type=['pdf'])
    jd_text = st.text_area("Paste Job Description here", height=300)
    top_k = st.number_input("Number of top keywords to extract", min_value=5, max_value=30, value=12)

if resume_file and jd_text:
    raw_resume = extract_text_from_pdf(resume_file)
    resume = clean_text(raw_resume)
    jd = clean_text(jd_text)

    # compute outputs
    top_kw = top_keywords(jd, n=top_k)
    present_skills, missing_skills = match_skills(resume)
    semantic = embedding_similarity_score(resume, jd, model)
    keyword_cov = keyword_coverage_score(top_kw, resume)
    # approximate JD skills count = number of common skills that appear in JD (helps skills coverage denominator)
    jd_skill_candidates = [s for s in COMMON_SKILLS if s.lower() in jd.lower()]
    skills_cov = (len([s for s in present_skills if s in jd_skill_candidates]) / max(1, len(jd_skill_candidates))) * 100 if len(jd_skill_candidates)>0 else 100.0

    overall = combined_score(semantic, skills_cov, keyword_cov)

    # Top display
    st.subheader("Overall fit")
    st.metric("Match score", f"{overall:.1f}%")

    # Subscores chart
    scores_df = pd.DataFrame({
        "Signal":["Semantic","SkillsCoverage","KeywordCoverage"],
        "Score":[semantic, skills_cov, keyword_cov]
    })
    fig = go.Figure([go.Bar(x=scores_df["Signal"], y=scores_df["Score"])])
    fig.update_layout(yaxis_range=[0,100], height=350, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Skills lists
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("#### Matched skills (from common list)")
        for s in present_skills:
            st.write("✅ " + s)
    with s2:
        st.markdown("#### Missing common skills (from list)")
        for s in missing_skills:
            st.write("❌ " + s)

    # Keywords lists
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("#### Top keywords extracted from job description")
        for k in top_kw:
            mark = "✅" if k.lower() in resume.lower() else "❌"
            st.write(f"{mark} {k}")
    with k2:
        st.markdown("#### Job description skill candidates (common skills found in JD)")
        for s in [s for s in COMMON_SKILLS if s.lower() in jd.lower()]:
            st.write("• " + s)

    # Recommendations
    st.subheader("Actionable recommendations")
    recs = recommendations_for_missing_skills(missing_skills)
    for r in recs:
        st.write("- " + r)
    for r in ["Tip: prefer short past-tense bullets. Quantify results when possible."]:
        st.info(r)

    # Downloadable short report
    report_text = f"""AI Resume-Job Match Report
Overall score: {overall:.1f}%
Semantic score: {semantic:.1f}%
Skills coverage: {skills_cov:.1f}%
Keyword coverage: {keyword_cov:.1f}%

Matched skills:
{', '.join(present_skills)}

Missing skills:
{', '.join(missing_skills)}

Top job keywords:
{', '.join(top_kw)}

Recommendations:
{chr(10).join(recs)}
"""
    st.download_button("Download report (plain text)", report_text, file_name="match_report.txt")

else:
    st.info("Upload a PDF resume and paste the job description to get results.")


#add lightweight LLM for output (input: match score + matched/missing skills; output: feedback + improvement suggestions)
#add NER-based keyword extraction rather than list
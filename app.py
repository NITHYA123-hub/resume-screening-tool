# ============================================================
# app.py - AI Resume Screening Tool (Streamlit)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, io, pickle, base64, time
from datetime import datetime

from parser import parse_resume, compute_ats_score, SKILLS_DB
from train_model import predict_category, clean_text

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="logo.png" if os.path.exists("logo.png") else ":briefcase:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ───────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
UPLOADS_DIR = os.path.join(BASE, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ─── Load CSS ────────────────────────────────────────────
def load_css():
    css_path = os.path.join(BASE, "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    # Extra Streamlit overrides
    st.markdown("""<style>
    .stApp { background: #0a0e1a; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1628 0%, #0a0e1a 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #f0f4ff; }
    .stMetric label { color: #94a3b8 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #f0f4ff !important; }
    div[data-testid="stExpander"] { background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04); border-radius: 8px;
        color: #94a3b8; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f8ef7, #8b5cf6);
        color: white !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f8ef7, #8b5cf6);
        color: white; border: none; border-radius: 10px;
        padding: 8px 24px; font-weight: 600;
    }
    .stButton > button:hover { transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79,142,247,0.35); }
    h1, h2, h3 { color: #f0f4ff !important; }
    p, li, span { color: #cbd5e1; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    </style>""", unsafe_allow_html=True)

load_css()

# ─── Session State ───────────────────────────────────────
if "resumes" not in st.session_state:
    st.session_state.resumes = []
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

# ─── Load ML Model ───────────────────────────────────────
@st.cache_resource
def load_ml_model():
    paths = [
        os.path.join(MODELS_DIR, "resume_classifier.pkl"),
        os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
        os.path.join(MODELS_DIR, "label_encoder.pkl"),
    ]
    if all(os.path.exists(p) for p in paths):
        with open(paths[0], "rb") as f: model = pickle.load(f)
        with open(paths[1], "rb") as f: vec = pickle.load(f)
        with open(paths[2], "rb") as f: enc = pickle.load(f)
        return model, vec, enc
    return None, None, None

model, vectorizer, encoder = load_ml_model()

# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <h1 style="font-size:1.6rem; background: linear-gradient(135deg,#4f8ef7,#8b5cf6);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        AI Resume Screener</h1>
        <p style="color:#94a3b8; font-size:0.85rem;">Intelligent Candidate Analysis</p>
    </div>""", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation", [
        "Dashboard",
        "Upload & Analyze",
        "Candidate Comparison",
        "Analytics",
        "Admin Panel",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown(f"""<div style="text-align:center; padding:10px; color:#64748b; font-size:0.75rem;">
        Model: {'Loaded' if model else 'Not found'}<br>
        Resumes: {len(st.session_state.resumes)}<br>
        {datetime.now().strftime('%b %d, %Y %H:%M')}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ═══════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown("""<h1 style="font-size:2.2rem;">
        <span style="background:linear-gradient(135deg,#4f8ef7,#8b5cf6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Dashboard</span></h1>""", unsafe_allow_html=True)

    resumes = st.session_state.resumes
    total = len(resumes)
    avg_ats = np.mean([r.get("ats", {}).get("total", 0) for r in resumes]) if resumes else 0
    high = sum(1 for r in resumes if r.get("ats", {}).get("total", 0) >= 70)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Resumes", total)
    c2.metric("Avg ATS Score", f"{avg_ats:.0f}/100")
    c3.metric("High Match", high)
    c4.metric("Categories", len(set(r.get("prediction", {}).get("category", "") for r in resumes)))

    if not resumes:
        st.info("No resumes uploaded yet. Go to **Upload & Analyze** to get started.")
        return

    col1, col2 = st.columns(2)
    with col1:
        scores = [r["ats"]["total"] for r in resumes]
        names = [r["parsed"]["name"] for r in resumes]
        fig = go.Figure(go.Bar(
            x=scores, y=names, orientation="h",
            marker=dict(color=scores, colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#10b981"]]),
        ))
        fig.update_layout(title="ATS Scores", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=max(300, len(resumes)*50), xaxis_range=[0,100])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cats = [r.get("prediction", {}).get("category", "Unknown") for r in resumes]
        cat_counts = pd.Series(cats).value_counts()
        fig2 = px.pie(values=cat_counts.values, names=cat_counts.index,
            color_discrete_sequence=["#4f8ef7","#8b5cf6","#06b6d4","#10b981","#f59e0b","#ef4444"])
        fig2.update_layout(title="Category Distribution", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # Top candidates table
    st.subheader("Top Candidates")
    rows = []
    for r in sorted(resumes, key=lambda x: x["ats"]["total"], reverse=True):
        rows.append({
            "Name": r["parsed"]["name"],
            "Email": r["parsed"]["email"],
            "Category": r.get("prediction", {}).get("category", "N/A"),
            "ATS Score": r["ats"]["total"],
            "Grade": r["ats"]["grade"],
            "Skills": len(r["parsed"]["skills"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# UPLOAD & ANALYZE PAGE
# ═══════════════════════════════════════════════════════════
def page_upload():
    st.markdown("""<h1 style="font-size:2.2rem;">
        <span style="background:linear-gradient(135deg,#4f8ef7,#8b5cf6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Upload & Analyze</span></h1>""", unsafe_allow_html=True)

    # Job Description
    with st.expander("Job Description (optional - improves matching)", expanded=False):
        jd = st.text_area("Paste the job description:", st.session_state.job_desc, height=120)
        st.session_state.job_desc = jd

    jd_keywords = [w.strip().lower() for w in jd.split() if len(w.strip()) > 2] if jd else []

    # Upload
    uploaded = st.file_uploader(
        "Upload resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("Analyze Resumes", type="primary"):
        progress = st.progress(0, text="Processing...")
        for i, f in enumerate(uploaded):
            progress.progress((i+1)/len(uploaded), text=f"Analyzing {f.name}...")
            file_bytes = f.read()
            parsed = parse_resume(file_bytes, f.name)
            ats = compute_ats_score(parsed, jd_keywords if jd_keywords else None)
            pred = {}
            if model:
                pred = predict_category(parsed["raw_text"], model, vectorizer, encoder)
            entry = {"parsed": parsed, "ats": ats, "prediction": pred, "timestamp": datetime.now().isoformat()}
            # Avoid duplicates
            existing = [r["parsed"]["filename"] for r in st.session_state.resumes]
            if f.name not in existing:
                st.session_state.resumes.append(entry)
            time.sleep(0.2)
        progress.empty()
        st.success(f"Analyzed {len(uploaded)} resume(s)!")
        st.rerun()

    # Show results
    if st.session_state.resumes:
        st.divider()
        for idx, r in enumerate(st.session_state.resumes):
            p = r["parsed"]
            ats = r["ats"]
            pred = r.get("prediction", {})

            with st.expander(f"{p['name']} | ATS: {ats['total']}/100 ({ats['grade']}) | {pred.get('category','N/A')}", expanded=(idx==0)):
                t1, t2, t3 = st.tabs(["Profile", "ATS Breakdown", "Category"])
                with t1:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Name:** {p['name']}")
                        st.markdown(f"**Email:** {p['email']}")
                        st.markdown(f"**Phone:** {p['phone']}")
                        st.markdown(f"**Experience:** {p['experience']}")
                    with c2:
                        st.markdown(f"**LinkedIn:** {p['linkedin']}")
                        st.markdown(f"**GitHub:** {p['github']}")
                        st.markdown(f"**File:** {p['filename']}")
                    if p["skills"]:
                        st.markdown("**Skills:**")
                        skill_html = " ".join(
                            f'<span style="display:inline-block;padding:4px 12px;margin:3px;'
                            f'border-radius:20px;font-size:0.8rem;font-weight:600;'
                            f'background:rgba(79,142,247,0.12);border:1px solid rgba(79,142,247,0.25);'
                            f'color:#4f8ef7;">{s}</span>' for s in p["skills"]
                        )
                        st.markdown(skill_html, unsafe_allow_html=True)
                    if p["education"]:
                        st.markdown("**Education:**")
                        for edu in p["education"]:
                            st.markdown(f"- {edu}")

                with t2:
                    for k, v in ats["breakdown"].items():
                        st.markdown(f"**{k}:** {v} pts")
                        st.progress(v / 30 if k != "Contact Info" else v / 20)
                    grade_colors = {"Excellent":"#10b981","Good":"#4f8ef7","Average":"#f59e0b","Poor":"#ef4444"}
                    color = grade_colors.get(ats["grade"], "#94a3b8")
                    st.markdown(f'<div style="text-align:center;font-size:2.5rem;font-weight:800;'
                        f'color:{color};margin:20px 0;">{ats["total"]}/100</div>'
                        f'<div style="text-align:center;color:{color};font-weight:600;">{ats["grade"]}</div>',
                        unsafe_allow_html=True)

                with t3:
                    if pred:
                        st.markdown(f"**Predicted Category:** {pred.get('category','N/A')}")
                        st.markdown(f"**Confidence:** {pred.get('confidence',0):.1f}%")
                        if pred.get("all_scores"):
                            scores_df = pd.DataFrame(
                                list(pred["all_scores"].items()), columns=["Category","Score"]
                            ).sort_values("Score", ascending=True)
                            fig = px.bar(scores_df, x="Score", y="Category", orientation="h",
                                color="Score", color_continuous_scale=["#1e293b","#4f8ef7","#8b5cf6"])
                            fig.update_layout(template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                height=250, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("ML model not loaded. Run `python train_model.py` first.")


# ═══════════════════════════════════════════════════════════
# COMPARISON PAGE
# ═══════════════════════════════════════════════════════════
def page_comparison():
    st.markdown("""<h1 style="font-size:2.2rem;">
        <span style="background:linear-gradient(135deg,#4f8ef7,#8b5cf6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Candidate Comparison</span></h1>""", unsafe_allow_html=True)

    resumes = st.session_state.resumes
    if len(resumes) < 2:
        st.info("Upload at least 2 resumes to compare candidates.")
        return

    names = [f"{r['parsed']['name']} ({r['parsed']['filename']})" for r in resumes]
    selected = st.multiselect("Select candidates to compare:", names, default=names[:min(3,len(names))])

    if len(selected) < 2:
        st.warning("Select at least 2 candidates.")
        return

    indices = [names.index(s) for s in selected]
    sel_resumes = [resumes[i] for i in indices]

    # Radar chart
    categories_radar = ["Contact", "Skills", "Education", "Job Match", "ATS Total"]
    fig = go.Figure()
    for r in sel_resumes:
        b = r["ats"]["breakdown"]
        vals = [
            b.get("Contact Info", 0) / 20 * 100,
            b.get("Skills Detected", 0) / 30 * 100,
            b.get("Education", 0) / 20 * 100,
            b.get("Job Match", 0) / 30 * 100,
            r["ats"]["total"],
        ]
        fig.add_trace(go.Scatterpolar(r=vals, theta=categories_radar,
            fill="toself", name=r["parsed"]["name"]))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0,100], showticklabels=True, gridcolor="rgba(255,255,255,0.1)"),
        angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=450,
        title="Candidate Comparison Radar")
    st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    comp_data = []
    for r in sel_resumes:
        comp_data.append({
            "Name": r["parsed"]["name"],
            "ATS Score": r["ats"]["total"],
            "Grade": r["ats"]["grade"],
            "Category": r.get("prediction",{}).get("category","N/A"),
            "Skills Count": len(r["parsed"]["skills"]),
            "Experience": r["parsed"]["experience"],
            "Email": r["parsed"]["email"],
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # Skill overlap
    st.subheader("Skill Analysis")
    all_skills_sets = [set(s.lower() for s in r["parsed"]["skills"]) for r in sel_resumes]
    if len(all_skills_sets) >= 2:
        common = all_skills_sets[0]
        for s in all_skills_sets[1:]:
            common = common & s
        if common:
            st.markdown("**Common Skills:** " + ", ".join(sorted(s.title() for s in common)))
        for i, r in enumerate(sel_resumes):
            unique = all_skills_sets[i] - common
            if unique:
                st.markdown(f"**Unique to {r['parsed']['name']}:** " + ", ".join(sorted(s.title() for s in unique)))


# ═══════════════════════════════════════════════════════════
# ANALYTICS PAGE
# ═══════════════════════════════════════════════════════════
def page_analytics():
    st.markdown("""<h1 style="font-size:2.2rem;">
        <span style="background:linear-gradient(135deg,#4f8ef7,#8b5cf6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Analytics</span></h1>""", unsafe_allow_html=True)

    resumes = st.session_state.resumes
    if not resumes:
        st.info("No data yet. Upload resumes first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        # Score distribution
        scores = [r["ats"]["total"] for r in resumes]
        fig = px.histogram(x=scores, nbins=10, labels={"x":"ATS Score","y":"Count"},
            color_discrete_sequence=["#4f8ef7"])
        fig.update_layout(title="ATS Score Distribution", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Grade breakdown
        grades = [r["ats"]["grade"] for r in resumes]
        grade_counts = pd.Series(grades).value_counts()
        colors_map = {"Excellent":"#10b981","Good":"#4f8ef7","Average":"#f59e0b","Poor":"#ef4444"}
        fig2 = px.pie(values=grade_counts.values, names=grade_counts.index,
            color=grade_counts.index, color_discrete_map=colors_map)
        fig2.update_layout(title="Grade Distribution", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # Top skills
    st.subheader("Most Common Skills")
    all_skills = []
    for r in resumes:
        all_skills.extend(r["parsed"]["skills"])
    if all_skills:
        skill_counts = pd.Series(all_skills).value_counts().head(15)
        fig3 = px.bar(x=skill_counts.values, y=skill_counts.index, orientation="h",
            labels={"x":"Count","y":"Skill"}, color=skill_counts.values,
            color_continuous_scale=["#1e293b","#4f8ef7","#8b5cf6"])
        fig3.update_layout(template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=400, showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig3, use_container_width=True)

    # Skills by category
    st.subheader("Skills by Category")
    cat_skill_data = []
    for cat_name, cat_skills in SKILLS_DB.items():
        for r in resumes:
            count = sum(1 for s in r["parsed"]["skills"] if s.lower() in cat_skills)
            if count:
                cat_skill_data.append({"Category": cat_name.replace("_"," ").title(), "Count": count})
    if cat_skill_data:
        df_cs = pd.DataFrame(cat_skill_data).groupby("Category").sum().reset_index()
        fig4 = px.bar(df_cs, x="Category", y="Count", color="Category",
            color_discrete_sequence=["#4f8ef7","#8b5cf6","#06b6d4","#10b981","#f59e0b","#ef4444"])
        fig4.update_layout(template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# ADMIN PAGE
# ═══════════════════════════════════════════════════════════
def page_admin():
    st.markdown("""<h1 style="font-size:2.2rem;">
        <span style="background:linear-gradient(135deg,#4f8ef7,#8b5cf6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Admin Panel</span></h1>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Status")
        if model:
            st.success("ML Model loaded successfully")
            report_path = os.path.join(MODELS_DIR, "training_report.txt")
            if os.path.exists(report_path):
                with open(report_path) as f:
                    st.code(f.read(), language="text")
        else:
            st.error("Model not found. Run: `python train_model.py`")

    with col2:
        st.subheader("Data Management")
        st.metric("Stored Resumes", len(st.session_state.resumes))
        if st.button("Clear All Resumes", type="secondary"):
            st.session_state.resumes = []
            st.rerun()
        if st.session_state.resumes:
            if st.button("Export to CSV"):
                rows = []
                for r in st.session_state.resumes:
                    rows.append({
                        "Name": r["parsed"]["name"],
                        "Email": r["parsed"]["email"],
                        "Phone": r["parsed"]["phone"],
                        "Skills": ", ".join(r["parsed"]["skills"]),
                        "Experience": r["parsed"]["experience"],
                        "Category": r.get("prediction",{}).get("category",""),
                        "ATS Score": r["ats"]["total"],
                        "Grade": r["ats"]["grade"],
                    })
                csv = pd.DataFrame(rows).to_csv(index=False)
                st.download_button("Download CSV", csv, "resume_analysis.csv", "text/csv")

    # Dataset info
    st.divider()
    st.subheader("Training Dataset")
    ds_path = os.path.join(BASE, "dataset", "resume_dataset.csv")
    if os.path.exists(ds_path):
        df = pd.read_csv(ds_path)
        c1, c2 = st.columns(2)
        c1.metric("Total Samples", len(df))
        c2.metric("Categories", df["Category"].nunique())
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# ─── Router ──────────────────────────────────────────────
if page == "Dashboard":
    page_dashboard()
elif page == "Upload & Analyze":
    page_upload()
elif page == "Candidate Comparison":
    page_comparison()
elif page == "Analytics":
    page_analytics()
elif page == "Admin Panel":
    page_admin()

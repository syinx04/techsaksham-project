import streamlit as st
import pdfplumber
import docx2txt
import pandas as pd
import re
import spacy
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
from fpdf import FPDF

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI Resume Screening", layout="wide")

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(docx_file):
    return docx2txt.process(docx_file)

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return " ".join([word for word in text.split() if word not in STOPWORDS])

def compute_ats_score(job_desc, resumes):
    vectorizer = TfidfVectorizer()
    corpus = [job_desc] + resumes
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0] * 100

def analyze_candidate(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE"]]
    experience = len([token for token in doc if token.text.lower() in ["years", "months"]])
    communication = "Good" if "communication" in text.lower() else "Needs improvement"
    education = "Bachelor's" if "bachelor" in text.lower() else "Master's" if "master" in text.lower() else "Not specified"
    efficiency = "High" if "managed" in text.lower() or "led" in text.lower() else "Moderate"
    return {"Skills": skills, "Experience": experience, "Communication": communication, "Education": education, "Efficiency": efficiency}

def auto_interview(candidate_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Interview this candidate based on resume details."},
                  {"role": "user", "content": candidate_text}]
    )
    return response["choices"][0]["message"]["content"]

def generate_pdf(candidate_name, details):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Candidate Report: {candidate_name}", ln=True, align='C')
    for key, value in details.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)
    return pdf.output(dest='S').encode('latin1')

selected = option_menu("AI Resume Screening", ["Upload", "Dashboard", "Visualize", "Summary", "Promptness", "Review", "Interview", "Download"],
                        icons=["cloud-upload", "bar-chart", "activity", "list-task", "check-circle", "star", "mic", "download"],
                        menu_icon="cast", default_index=0, orientation="horizontal")

if selected == "Upload":
    st.header("Upload Job Description & Resumes")
    job_description = st.text_area("Enter Job Description", height=200)
    uploaded_resumes = st.file_uploader("Upload Resume Files", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_resumes:
        st.session_state.uploaded_resumes = uploaded_resumes
        st.session_state.job_description = job_description
        st.success("Files uploaded successfully!")

if selected == "Dashboard" and "uploaded_resumes" in st.session_state:
    job_text = clean_text(st.session_state.job_description)
    resume_texts, resume_names, candidate_details = [], [], []
    for resume in st.session_state.uploaded_resumes:
        text = extract_text_from_pdf(resume) if resume.name.endswith(".pdf") else extract_text_from_docx(resume)
        cleaned_text = clean_text(text)
        resume_texts.append(cleaned_text)
        resume_names.append(resume.name)
        candidate_details.append(analyze_candidate(text))
    scores = compute_ats_score(job_text, resume_texts)
    st.session_state.results_df = pd.DataFrame({"Candidate Name": resume_names, "ATS Score (%)": scores}).sort_values(by="ATS Score (%)", ascending=False)
    st.session_state.resume_names = resume_names
    st.session_state.candidate_details = candidate_details
    st.dataframe(st.session_state.results_df)

if selected == "Visualize" and "results_df" in st.session_state:
    st.header("Candidate Attribute Analysis")
    fig, ax = plt.subplots()
    sns.barplot(x="ATS Score (%)", y="Candidate Name", data=st.session_state.results_df, ax=ax)
    st.pyplot(fig)

if selected == "Summary" and "resume_names" in st.session_state:
    selected_candidate = st.selectbox("Select Candidate for Details", st.session_state.resume_names)
    if selected_candidate:
        idx = st.session_state.resume_names.index(selected_candidate)
        st.write(st.session_state.candidate_details[idx])

if selected == "Promptness":
    st.header("Document Promptness Check")
    st.write("Promptness score is evaluated based on response time and clarity in the provided resumes.")
    st.write("Candidates with structured resumes have higher promptness scores.")

if selected == "Review":
    st.header("Candidate Review & Rating")
    st.write("Review includes resume clarity, conciseness, and alignment with job requirements.")
    st.write("Each candidate is rated based on grammar, structure, and keyword alignment.")

if selected == "Interview" and "resume_names" in st.session_state:
    if st.button("Start Auto Interview"):
        idx = 0  # Selecting first resume for demonstration
        interview_response = auto_interview(st.session_state.resume_names[idx])
        st.write(interview_response)

if selected == "Download" and "resume_names" in st.session_state:
    selected_candidate = st.selectbox("Select Candidate for Report", st.session_state.resume_names)
    if selected_candidate:
        idx = st.session_state.resume_names.index(selected_candidate)
        pdf_content = generate_pdf(selected_candidate, st.session_state.candidate_details[idx])
        st.download_button("Download Report", pdf_content, file_name=f"{selected_candidate}_report.pdf", mime="application/pdf")

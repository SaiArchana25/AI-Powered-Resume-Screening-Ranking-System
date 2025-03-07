import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#function to extract text from pdf
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
        return text

#function to rank resumes based on job description

def rank_resumes(job_description, resumes):
    #combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    #Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors [1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

#Streamlit app

st.title("AI Resume Screening & Candidate Ranking System")
#Job description
st.header("Job Description")
job_description = st.text_area("Enter job description")

#file uploader
st.header("Upload resumes")
uploaded_files = st.file_uploader("Upload PDF files",type=["pdf"],accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    #Rank resumes
    scores = rank_resumes(job_description, resumes)

    #Display scores
    results = pd.DataFrame({"Resume":[file.name for file in uploaded_files], "Scores": scores})
    results = results.sort_values(by="Scores", ascending=False)

    st.write(results)



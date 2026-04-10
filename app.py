import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load models
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

st.title("AI Resume Screening System")

# Input
job_desc = st.text_area("Enter Job Description")
uploaded_files = st.file_uploader("Upload Resumes (TXT only)", accept_multiple_files=True)

resumes = []
for file in uploaded_files:
    resumes.append(file.read().decode("utf-8"))

if st.button("Analyze"):
    if not job_desc or len(resumes) == 0:
        st.warning("Please provide job description and upload resumes")
    else:
        documents = [job_desc] + resumes
        tfidf_matrix = tfidf.transform(documents)
        
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        ranked = sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)
        
        st.subheader("Ranked Candidates")
        
        for i, (resume, score) in enumerate(ranked):
            st.write(f"### Rank {i+1}")
            st.write(f"Score: {score:.2f}")
            st.write(resume[:200] + "...")
            
            # Explanation
            feature_names = tfidf.get_feature_names_out()
            job_vec = tfidf.transform([job_desc]).toarray()[0]
            res_vec = tfidf.transform([resume]).toarray()[0]
            
            common = np.where((job_vec > 0) & (res_vec > 0))[0]
            keywords = [feature_names[i] for i in common[:5]]
            
            st.write("Matching Skills:", keywords)

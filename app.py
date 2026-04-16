import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="AI Answer Evaluation System",
    page_icon="🧠",
    layout="centered"
)

# ---------------- DARK THEME ----------------
st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

h1, h2, h3 {
    color: #ffffff !important;
}

label {
    color: #ffffff !important;
}

/* Input box */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #1e222a;
    color: #ffffff !important;
    border: 1px solid #333;
}

/* Button */
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}

.stButton > button:hover {
    background-color: #45a049;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🧠 AI Subjective Answer Evaluation</h1>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- INPUT ----------------
question = st.text_area("❓ Question")
model_answer = st.text_area("📘 Model Answer")
student_answer = st.text_area("✍️ Student Answer")

st.markdown("---")

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

# ---------------- EVALUATION ----------------
def evaluate_answer(model, student):

    model_clean = clean_text(model)
    student_clean = clean_text(student)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([model_clean, student_clean])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    score = round(similarity * 10, 2)

    if score > 7:
        feedback = "Excellent answer 👍"
    elif score > 4:
        feedback = "Good answer, but needs improvement ✍️"
    else:
        feedback = "Poor answer, revise concepts 📚"

    return similarity, score, feedback

# ---------------- BUTTON ----------------
if st.button("🚀 Evaluate Answer"):

    if model_answer and student_answer:

        similarity, score, feedback = evaluate_answer(model_answer, student_answer)

        st.markdown("## 📊 Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("📈 Similarity", f"{round(similarity*100,2)}%")

        with col2:
            st.metric("🏆 Score", f"{score}/10")

        st.progress(int(similarity * 100))

        st.markdown("### 📝 Feedback")
        st.success(feedback)

    else:
        st.error("Please fill all fields!")
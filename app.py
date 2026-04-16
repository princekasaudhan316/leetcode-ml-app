import sys
import os
sys.path.append(os.path.abspath("."))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.model import load_model
from src.recommend import recommend_questions

# ---------------------------
# LOAD
# ---------------------------

model = load_model()
student_df = pd.read_csv("data/student_data.csv")
questions_df = pd.read_csv("data/questions_data.csv")

st.set_page_config(page_title="Adaptive Learning", layout="wide")

st.title("🚀 Adaptive Question Recommendation System")

# ---------------------------
# INPUT
# ---------------------------

col1, col2 = st.columns(2)

with col1:
    student_id = st.number_input("Enter Student ID", min_value=1, step=1)

with col2:
    num_questions = st.slider("Number of Questions", 1, 10, 5)

# ---------------------------
# BUTTON
# ---------------------------

if st.button("Get Recommendations"):

    skill, weak, topics, recs = recommend_questions(
        student_id,
        student_df,
        questions_df,
        model,
        top_n=num_questions
    )

    # ---------------------------
    # SKILL DISPLAY
    # ---------------------------

    st.subheader("📊 Skill Score")

    st.progress(min(skill / 10, 1.0))
    st.write(f"**Skill Score:** {skill:.2f} / 10")

    if skill < 4:
        st.error("Level: Beginner")
    elif skill < 7:
        st.warning("Level: Intermediate")
    else:
        st.success("Level: Advanced")

    # ---------------------------
    # WEAK TOPICS
    # ---------------------------

    st.subheader("⚠️ Weak Topics")

    if weak:
        st.write(", ".join(weak))
    else:
        st.write("No weak topics detected")

    # ---------------------------
    # GRAPH
    # ---------------------------

    st.subheader("📈 Topic Performance")

    if not topics.empty:
        fig, ax = plt.subplots()

        ax.bar(topics["skill_id"], topics["correct"])
        ax.set_xlabel("Topic")
        ax.set_ylabel("Accuracy")

        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("No topic data available")

    # ---------------------------
    # RECOMMENDATIONS
    # ---------------------------

    st.subheader("📚 Recommended Questions")

    if recs.empty:
        st.error("No questions found. Try different input.")
    else:
        for _, row in recs.iterrows():
            st.markdown(f"""
            ### {row['title']}
            - Difficulty: {row['difficulty']}
            - 👍 {row.get('likes', 0)} | 👎 {row.get('dislikes', 0)}
            - 🔗 [Solve Problem]({row.get('url', '#')})
            """)
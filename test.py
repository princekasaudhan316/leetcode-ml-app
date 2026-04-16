import pandas as pd
from src.model import load_model
from src.recommend import recommend_questions

# Load data
student_df = pd.read_csv("data/student_data.csv")
questions_df = pd.read_csv("data/questions_data.csv")

# Load model
model = load_model()

# Test student
student_id = student_df["student_id"].iloc[0]

skill, recs = recommend_questions(
    student_id,
    student_df,
    questions_df,
    model
)

print("Skill Score:", skill)
print(recs[["title", "difficulty"]])
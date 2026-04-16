import pandas as pd

# ---------------------------
# SKILL PREDICTION
# ---------------------------

def predict_skill(student_id, student_df, model):

    df = student_df[student_df["student_id"] == student_id]

    if df.empty:
        return 0  # safety

    agg = df.agg({
        "correct": "mean",
        "time_taken": "mean",
        "problem_id": "count",
        "difficulty_score": "mean"
    }).to_frame().T

    agg.rename(columns={
        "correct": "avg_accuracy",
        "time_taken": "avg_time",
        "problem_id": "attempts",
        "difficulty_score": "avg_difficulty"
    }, inplace=True)

    X = agg[["avg_time", "attempts", "avg_difficulty"]]

    skill_score = model.predict(X)[0]

    return float(skill_score)


# ---------------------------
# TOPIC ANALYSIS
# ---------------------------

def analyze_topics(student_df, student_id, top_n=3):

    df = student_df.copy()

    # Handle multi-topic values like "44;45;181"
    df["skill_id"] = df["skill_id"].astype(str)
    df = df.assign(skill_id=df["skill_id"].str.split(";"))
    df = df.explode("skill_id")

    # Compute accuracy per topic
    topic_acc = df.groupby(["student_id", "skill_id"])["correct"].mean().reset_index()

    student_topics = topic_acc[topic_acc["student_id"] == student_id]

    if student_topics.empty:
        return [], pd.DataFrame()

    # Weak topics
    weak_topics = (
        student_topics
        .sort_values(by="correct")
        .head(top_n)["skill_id"]
        .tolist()
    )

    return weak_topics, student_topics
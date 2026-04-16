import pandas as pd

from src.preprocess import predict_skill, analyze_topics

# ---------------------------
# MAP SKILL → DIFFICULTY
# ---------------------------

def map_skill_to_difficulty(score):
    if score < 2:
        return "Easy"
    elif score < 5:
        return "Medium"
    else:
        return "Hard"


# ---------------------------
# MAIN RECOMMEND FUNCTION
# ---------------------------

def recommend_questions(student_id, student_df, questions_df, model, top_n=5):

    # 1. Predict skill
    skill_score = predict_skill(student_id, student_df, model)

    # 2. Analyze topics
    weak_topics, topic_performance = analyze_topics(student_df, student_id)

    # 3. Map difficulty
    difficulty = map_skill_to_difficulty(skill_score)

    # 4. Filter by difficulty
    filtered = questions_df[
    questions_df["difficulty"].str.lower() == difficulty.lower()
    ].copy()

    # 5. Topic matching (if weak topics exist)
    # 5. Topic matching (SAFE VERSION)
    if weak_topics:
    
        def topic_match(q_topics):
            return any(str(topic) in str(q_topics) for topic in weak_topics)

    # Try filtering
        filtered_temp = filtered[
            filtered["related_topics"].astype(str).apply(topic_match)
        ]

    # Only apply if it gives results
        if not filtered_temp.empty:
           filtered = filtered_temp

    # 6. Ranking logic
    filtered["score"] = (
        filtered.get("likes", 0) - filtered.get("dislikes", 0)
    )

    filtered = filtered.sort_values(by="score", ascending=False)

    # 7. Return everything
    return skill_score, weak_topics, topic_performance, filtered.head(top_n)
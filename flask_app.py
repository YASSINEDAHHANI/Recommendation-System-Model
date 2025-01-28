from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import ast 

app = Flask(__name__)

# Helper function to load and filter courses
def load_and_filter_courses(user_preferences):
    dataset_path = 'processed_coursera_courses1.csv'
    courses = pd.read_csv(dataset_path, encoding='ISO-8859-1')
    courses['Numeric Ratings'] = pd.to_numeric(courses['Ratings'], errors='coerce')

    def convert_reviews(review):
        if pd.isna(review):
            return 0
        if 'K' in str(review):
            return float(review.replace('K', '').replace('k', '')) * 1000
        return float(review)

    courses['Review counts'] = courses['Review counts'].apply(convert_reviews)
    courses['Level'] = courses['Level'].str.strip()

    enrolled_courses = user_preferences.get('enrolledCourses', [])
    completed_courses = user_preferences.get('completedCourses', [])
    return courses[~courses['Title'].isin(enrolled_courses + completed_courses)]

# Route: /recommend
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_preferences = request.get_json()
        courses = load_and_filter_courses(user_preferences)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(courses[['Level']])

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit(courses['Skills'])

        user_level = user_preferences['level']
        user_skills = ' '.join(user_preferences['skills'])

        if user_level not in encoder.categories_[0]:
            return jsonify({"error": f"Unknown level: {user_level}. Known levels are: {encoder.categories_[0]}"})

        user_level_vec = encoder.transform(pd.DataFrame([[user_level]], columns=['Level']))
        user_skills_vec = tfidf.transform([user_skills])
        user_vector = np.hstack([user_level_vec, user_skills_vec.toarray()])

        course_levels = encoder.transform(courses[['Level']])
        course_skills = tfidf.transform(courses['Skills'])
        course_vectors = np.hstack([course_levels, course_skills.toarray()])

        similarities = cosine_similarity(user_vector, course_vectors)

        courses['Similarity'] = similarities[0]
        courses['Final_Score'] = (
            courses['Similarity'] +
            0.5 * courses['Numeric Ratings'].fillna(0) +
            0.3 * np.log1p(courses['Review counts'])
        )

        recommended_courses = courses.sort_values(by='Final_Score', ascending=False)
        recommended_courses_list = recommended_courses[['Title', 'Level', 'Skills', 'Numeric Ratings', 'Review counts', 'Final_Score']].head(10).to_dict(orient='records')
        return jsonify(recommended_courses_list)

    except Exception as e:
        return jsonify({"error": "An error occurred during recommendation. Please check your input and try again."})

# Route: /top_courses_by_skill_and_level
@app.route('/top_courses_by_skill_and_level', methods=['POST'])
def top_courses_by_skill_and_level():
    try:

        user_preferences = request.get_json()
        user_skills = user_preferences['skills']  

        # Load courses dataset
        courses = load_and_filter_courses(user_preferences) 

        # One-hot encode levels
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(courses[['Level']])

        # TF-IDF vectorizer for skills
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit(courses['Skills'])

        # Prepare response dictionary
        top_courses_by_skill = {}

        # Iterate over each skill provided
        for skill in user_skills:
            # Filter courses that contain the skill
            skill_courses = courses[courses['Skills'].str.contains(skill, case=False, na=False)]

            if skill_courses.empty:
                top_courses_by_skill[skill] = {"beginner": None, "intermediate": None, "advanced": None}
                continue

            # Calculate similarity scores for the skill
            user_skill_vec = tfidf.transform([skill])
            skill_courses_skills = tfidf.transform(skill_courses['Skills'])

            skill_courses['Similarity'] = cosine_similarity(user_skill_vec, skill_courses_skills)[0]
            skill_courses['Final_Score'] = (
                skill_courses['Similarity'] +
                0.5 * skill_courses['Numeric Ratings'].fillna(0) +
                0.3 * np.log1p(skill_courses['Review counts'])
            )

            # Get top courses for each level
            top_courses = {"Beginner": None, "Intermediate": None, "Advanced": None}
            for level in ['beginner', 'intermediate', 'advanced']:
                level_courses = skill_courses[skill_courses['Level'].str.lower() == level]
                if not level_courses.empty:
                    top_course = level_courses.nlargest(1, 'Final_Score').iloc[0]
                    top_courses[level] = {
                        "Title": top_course['Title'],
                        "Level": top_course['Level'],
                        "Skills": top_course['Skills'],
                        "Numeric Ratings": top_course['Numeric Ratings'],
                        "Review counts": top_course['Review counts'],
                        "Final_Score": top_course['Final_Score']
                    }
                else:
                    top_courses[level] = None  

            # Add the top courses for this skill to the response
            top_courses_by_skill[skill] = top_courses

        return jsonify(top_courses_by_skill)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})



if __name__ == '__main__':
    app.run(debug=True, port=5001)




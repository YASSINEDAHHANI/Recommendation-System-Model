from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Define the route
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Load and process data
        dataset_path = 'processed_coursera_courses1.csv'  
        courses = pd.read_csv(dataset_path, encoding='ISO-8859-1')

        # Convert columns to appropriate data types
        courses['Numeric Ratings'] = pd.to_numeric(courses['Ratings'], errors='coerce')  # Handle non-numeric entries

        def convert_reviews(review):
            if pd.isna(review):
                return 0  
            if 'K' in str(review):
                return float(review.replace('K', '').replace('k', '')) * 1000
            return float(review)

        courses['Review counts'] = courses['Review counts'].apply(convert_reviews)
        courses['Level'] = courses['Level'].str.strip()

        # Handle OneHotEncoder and TF-IDF Vectorizer
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(courses[['Level']])

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit(courses['Skills'])

        # Get user preferences from the request
        user_preferences = request.get_json()

        # Validate user level
        if user_preferences['Level'] not in encoder.categories_[0]:
            return jsonify({"error": f"Unknown level: {user_preferences['Level']}. Known levels are: {encoder.categories_[0]}"})

        # Encode user preferences
        user_level = encoder.transform(pd.DataFrame([[user_preferences['Level']]], columns=['Level']))
        user_skills = tfidf.transform([user_preferences['Skills']])
        user_vector = np.hstack([user_level, user_skills.toarray()])

        # Encode course data
        course_levels = encoder.transform(courses[['Level']])
        course_skills = tfidf.transform(courses['Skills'])
        course_vectors = np.hstack([course_levels, course_skills.toarray()])

        # Compute cosine similarity
        similarities = cosine_similarity(user_vector, course_vectors)

        courses['Similarity'] = similarities[0]
        courses['Final_Score'] = (
            courses['Similarity'] +
            0.5 * courses['Numeric Ratings'].fillna(0) +  # Fill NaN with 0 for ratings
            0.3 * np.log1p(courses['Review counts'])  # Log transform review counts for normalization
        )

        # Sort courses by similarity and score
        recommended_courses = courses.sort_values(by='Similarity', ascending=False)

        # Prepare recommended courses list to return
        recommended_courses_list = recommended_courses[['Title', 'Level', 'Skills', 'Numeric Ratings', 'Review counts', 'Final_Score']].head(10).to_dict(orient='records')

        return jsonify(recommended_courses_list)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

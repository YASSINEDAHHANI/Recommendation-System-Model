import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
dataset_path = 'processed_coursera_courses1.csv'  
courses = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Convert ratings to numeric format
courses['Numeric Ratings'] = courses['Ratings'].astype(float)

# Convert review counts (e.g., "10K" -> 10000)
def convert_reviews(review):
    if pd.isna(review):
        return 0  
    if 'K' in str(review):
        return float(review.replace('K', '').replace('k', '')) * 1000
    return float(review)

courses['Review counts'] = courses['Review counts'].apply(convert_reviews)

# Clean up the level data
courses['Level'] = courses['Level'].str.strip()

# One-Hot Encoding for "Level"
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(courses[['Level']])

# TF-IDF Vectorizer for "Skills"
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(courses['Skills'])

# Function to calculate course recommendations
def get_course_recommendations(user_preferences):
    # Encoding user preferences
    if user_preferences['Level'] not in encoder.categories_[0]:
        print(f"Unknown level: {user_preferences['Level']}. Known levels are: {encoder.categories_[0]}")
        user_level = encoder.transform(pd.DataFrame([[encoder.categories_[0][0]]], columns=['Level']))
    else:
        user_level = encoder.transform(pd.DataFrame([[user_preferences['Level']]], columns=['Level']))

    user_skills = tfidf.transform([user_preferences['Skills']])

    # Combine user level and skills into a single vector
    user_vector = np.hstack([user_level, user_skills.toarray()])

    # Process the courses to create feature vectors
    course_levels = encoder.transform(courses[['Level']])
    course_skills = tfidf.transform(courses['Skills'])
    course_vectors = np.hstack([course_levels, course_skills.toarray()])

    # Compute cosine similarity between user preferences and courses
    similarities = cosine_similarity(user_vector, course_vectors)

    # Add similarity to courses
    courses['Similarity'] = similarities[0]

    # Calculate final score based on similarity, numeric ratings, and review counts
    courses['Final_Score'] = (
        courses['Similarity'] +
        0.5 * courses['Numeric Ratings'] +
        0.3 * np.log1p(courses['Review counts'])
    )

    # Return sorted recommendations
    recommended_courses = courses.sort_values(by='Similarity', ascending=False)
    return recommended_courses[['Title', 'Level', 'Skills', 'Numeric Ratings', 'Review counts', 'Final_Score']].head(10)

# Example user preferences
user_preferences = {
    "Level": "Beginner",
    "Skills": "Java"
}


# Get recommended courses based on the user preferences
recommended_courses = get_course_recommendations(user_preferences)

# Display recommended courses
print("\nRecommended Courses:")
print(recommended_courses)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Generate synthetic hotel data based on travel context
@st.cache_data
def generate_hotel_data():
    # Synthetic hotel data with location and price (aligned with flight cities)
    data = {
        'hotel_id': [101, 102, 103, 104, 105],
        'location': ['Rio de Janeiro', 'Sao Paulo', 'Brasilia', 'Recife', 'Florianopolis'],
        'price_range': [150, 200, 180, 130, 220],
        'rating': [4.2, 4.0, 3.8, 4.5, 4.1]
    }
    return pd.DataFrame(data)

# Create feature matrix for similarity
@st.cache_data
def create_feature_matrix(df):
    # One-hot encode locations for similarity
    location_encoded = pd.get_dummies(df['location'])
    features = pd.concat([location_encoded, df[['price_range', 'rating']]], axis=1)
    return features

# Compute similarity
@st.cache_data
def compute_similarity(matrix):
    return cosine_similarity(matrix)

# Get recommendations based on user preferences
def get_recommendations(preferred_location, preferred_price, matrix, similarity, top_n=3):
    # Create a user preference vector
    user_pref = pd.DataFrame(0, index=[0], columns=matrix.columns)
    if preferred_location in matrix.columns:
        user_pref[preferred_location] = 1
    user_pref['price_range'] = preferred_price
    user_pref['rating'] = 4.0  # Default preference for high ratings

    # Compute similarity with hotels
    sim_scores = cosine_similarity(matrix, user_pref)
    hotel_scores = pd.Series(sim_scores.flatten(), index=matrix.index)
    return hotel_scores.sort_values(ascending=False).head(top_n)

# Streamlit app
st.title("Travel Recommendation System")

# Load data
hotel_df = generate_hotel_data()
feature_matrix = create_feature_matrix(hotel_df)
similarity = compute_similarity(feature_matrix)

# Sidebar for user input
st.sidebar.header("User Preferences")
preferred_location = st.sidebar.selectbox("Preferred Location", options=hotel_df['location'].unique(), index=0)
preferred_price = st.sidebar.slider("Preferred Price Range ($)", min_value=100, max_value=300, value=150)
top_n = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=5, value=3)

# Get and display recommendations
if st.sidebar.button("Get Recommendations"):
    recommendations = get_recommendations(preferred_location, preferred_price, feature_matrix, similarity, top_n)
    if not recommendations.empty:
        st.header("Top Hotel Recommendations")
        rec_df = hotel_df.loc[recommendations.index][['hotel_id', 'location', 'price_range', 'rating']]
        rec_df['Similarity Score'] = recommendations.values
        st.table(rec_df)
    else:
        st.warning("No recommendations available based on your preferences.")

# Visualizations
st.header("Insights")
st.subheader("Price Distribution")
st.bar_chart(hotel_df['price_range'].value_counts().sort_index())

st.subheader("Rating Distribution")
st.bar_chart(hotel_df['rating'].value_counts().sort_index())

# Add interactivity for custom input
st.sidebar.header("Add New Hotel Preference")
new_location = st.sidebar.selectbox("New Hotel Location", options=[''] + list(hotel_df['location'].unique()), index=0)
new_price = st.sidebar.number_input("New Hotel Price Range ($)", min_value=100, max_value=300, value=150)
new_rating = st.sidebar.number_input("New Hotel Rating", min_value=0.0, max_value=5.0, value=4.0)
if st.sidebar.button("Add Hotel"):
    if new_location:
        new_hotel = pd.DataFrame({
            'hotel_id': [max(hotel_df['hotel_id']) + 1],
            'location': [new_location],
            'price_range': [new_price],
            'rating': [new_rating]
        })
        hotel_df = pd.concat([hotel_df, new_hotel], ignore_index=True)
        feature_matrix = create_feature_matrix(hotel_df)
        similarity = compute_similarity(feature_matrix)
        st.experimental_rerun()
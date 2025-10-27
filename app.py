from flask import Flask, render_template, request
import pandas as pd
from model import collaborative_filtering, content_based_filtering

app = Flask(__name__)

# Load data
users = pd.read_csv('users.csv')
content = pd.read_csv('content.csv')
interactions = pd.read_csv('interactions.csv')
browsing_history = pd.read_csv('browsing_history.csv')

@app.route('/')
def index():
    return render_template('index.html', content=content.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    user_id = int(request.form['user_id'])
    algorithm = request.form['algorithm']

    # Get content IDs the user has interacted with
    interacted_content_ids = interactions[interactions['user_id'] == user_id]['content_id'].unique()
    browsed_content_ids = browsing_history[browsing_history['user_id'] == user_id]['content_id'].unique()

    # Content the user has already interacted with
    interacted_content = content[content['content_id'].isin(interacted_content_ids) | content['content_id'].isin(browsed_content_ids)]

    # Generate recommendations based on the selected algorithm
    if algorithm == 'user-collaborative':
        recommendations = collaborative_filtering(user_id, interactions, content, is_user_based=True)
    elif algorithm == 'item-collaborative':
        recommendations = collaborative_filtering(user_id, interactions, content)
    elif algorithm == 'content-based':
        recommendations = content_based_filtering(user_id, browsing_history, content)
    elif algorithm == 'hybrid':
        collaborative_recommendations = collaborative_filtering(user_id, interactions, content)
        content_based_recommendations = content_based_filtering(user_id, browsing_history, content)
        recommendations = pd.concat([collaborative_recommendations, content_based_recommendations]).drop_duplicates(subset=['content_id'], keep='first').reset_index(drop=True)

    # Ensure 'content_id' column exists before proceeding
    if 'content_id' not in recommendations.columns:
        print("Warning: 'content_id' column is missing from recommendations.")
        return "Error: 'content_id' column is missing."

    # Exclude content the user has already interacted with
    recommended_content = recommendations[~recommendations['content_id'].isin(interacted_content_ids) &
                                          ~recommendations['content_id'].isin(browsed_content_ids)]

    # Prepare data for rendering
    interacted_content = interacted_content[['content_id', 'title', 'category', 'popularity']].copy()
    interacted_content['source'] = interacted_content['content_id'].apply(
        lambda x: 'Interacted' if x in interacted_content_ids else 'Browsed'
    )

    recommended_content = recommended_content[['content_id', 'title', 'category', 'popularity']].copy()
    recommended_content['source'] = 'Recommended'

    return render_template('recommendations.html', 
                       interacted_content=interacted_content.to_dict(orient='records'),
                       recommended_content=recommended_content.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)

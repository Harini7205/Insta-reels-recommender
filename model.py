import pandas as pd

def collaborative_filtering(user_id, interactions, content, is_user_based=False):
    # Find content that the target user has engaged with
    user_interactions = interactions[interactions['user_id'] == user_id]['content_id'].unique()
    
    if is_user_based:
        # Implement user-based collaborative filtering logic
        other_users = interactions[interactions['content_id'].isin(user_interactions)]['user_id'].unique()
        other_content = interactions[interactions['user_id'].isin(other_users)]['content_id'].unique()
    else:
        # Implement item-based collaborative filtering logic
        other_content = interactions[interactions['user_id'] != user_id]['content_id'].unique()

    # Exclude content the user has already interacted with
    recommendations = content[~content['content_id'].isin(user_interactions) & 
                              content['content_id'].isin(other_content)].copy()
    recommendations['source'] = 'Collaborative Filtering'

    # Ensure the necessary columns are present
    required_columns = ['content_id', 'title', 'category', 'popularity']
    missing_columns = [col for col in required_columns if col not in recommendations.columns]
    for col in missing_columns:
        recommendations[col] = None  # Handle missing columns gracefully by adding placeholders or default values
    
    # Return final recommendations limited to min_recommendations
    return recommendations[['content_id', 'title', 'category', 'popularity', 'source']]

def content_based_filtering(user_id, browsing_history, content):
    # User's browsing history of content
    user_history = browsing_history[browsing_history['user_id'] == user_id]['content_id'].unique()
    user_content = content[content['content_id'].isin(user_history)]

    # Recommend similar content by category
    if not user_content.empty:
        recommendations = content[content['category'].isin(user_content['category']) &
                                  ~content['content_id'].isin(user_history)].copy()
    else:
        recommendations = pd.DataFrame()  # Return empty if no content in user history

    # Ensure the necessary columns are present
    required_columns = ['content_id', 'title', 'category', 'popularity']
    missing_columns = [col for col in required_columns if col not in recommendations.columns]
    for col in missing_columns:
        recommendations[col] = None  # Handle missing columns gracefully by adding placeholders or default values

    recommendations['source'] = 'Content-Based Filtering'
    
    # Ensure we return the right columns
    return recommendations[['content_id', 'title', 'category', 'popularity', 'source']]

import pandas as pd
import random

# Read existing data from recsys.csv
df = pd.read_csv('./recsys.csv')

# Define the constraints and format
user_ids = ["A" + str(i) for i in range(10, 100)]  # A10 to A99
news_ids = ["B" + str(i) for i in range(10, 201)]  # B10 to B200
user_ages = ["초등학생", "중학생", "고등학생", "대학생"]
news_contents = [1000, 2000, 3000, 4000]

# Mapping for userAge and newsContents to ensure consistency for userId and newsId
user_age_map = {user_id: random.choice(user_ages) for user_id in user_ids}
news_content_map = {news_id: random.choice(news_contents) for news_id in news_ids}

# Generate the data
data = []

for _ in range(1000):
    user_id = random.choice(user_ids)
    news_id = random.choice(news_ids)
    
    data.append([user_id, user_age_map[user_id], news_id, news_content_map[news_id]])

# Convert to DataFrame
new_df = pd.DataFrame(data, columns=["userId", "userAge", "newsId", "newsContents"])

# Concatenate with the existing dataframe
combined_df = pd.concat([df, new_df], ignore_index=True)

# Save the combined dataframe back to recsys.csv
combined_df.to_csv('./recsys.csv', index=False)

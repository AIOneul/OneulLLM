import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split


# Load the CSV file into a DataFrame
df = pd.read_csv('./dataset/recsys.csv')
df = df.drop(columns=[' 1', ' 2', ' 3'])
df['score'] = 1
# Using userAge as 'user' and newsContents as 'item'
user_item_data = df[['userAge', 'newsContents', 'score']]
reader = Reader(rating_scale=(0, 5))  # Assuming the range of newsContents is from 1000 to 4000

# Load the data
data = Dataset.load_from_df(user_item_data, reader)

# Split data into training and test set (e.g., 80% for training and 20% for testing)
trainset, testset = train_test_split(data, test_size=0.2)

# Use Singular Value Decomposition (SVD) as the algorithm and fit on the training data
algo = SVD()
algo.fit(trainset)

# Predictions on the test set
predictions = algo.test(testset)

# Calculate and print RMSE for our predictions
print("Test Set RMSE: ", accuracy.rmse(predictions, verbose=True))

# Sample prediction for a userAge
user_age_sample = '대학생'
news_content_sample = 3000.0
prediction_sample = algo.predict(user_age_sample, news_content_sample)
print("Prediction for userAge:", user_age_sample, "with newsContents value:", news_content_sample, "is:", prediction_sample.est)
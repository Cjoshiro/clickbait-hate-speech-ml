#Data used: data/Instagram.csv
#Script description: perform LSA on the captions and hashtags of the instagram data

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

#Imports from src
from src.utils import clean_and_preprocess
from src.features import get_cleaned_hashtags_captions, build_lsa_features

# Read in data, clean
data = pd.read_csv(os.path.join(project_root, "data", "Instagram.csv"))
#data = data[["Likes","Comments","Clickbait","Hashtags","Captions"]]
data = data.dropna()

# Grab hashtags and captions, used in LSA
data.Hashtags = data.Hashtags.astype('str')
data.Captions = data.Captions.astype('str')
Hashtags = list(data["Hashtags"])
Captions = list(data["Captions"])

# Clean hashtags and captions
cleaned_Hashtags_, cleaned_Captions_ = get_cleaned_hashtags_captions(data)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Fit and transform the TF-IDF vectorizer on the combined text data
tfidf_matrix = vectorizer.fit_transform(cleaned_Captions_ + cleaned_Hashtags_)
# Perform Latent Semantic Analysis (LSA) using Truncated SVD
num_topics = 1  # Adjust the number of topics as needed
lsa = TruncatedSVD(n_components=num_topics, random_state=42)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# Create DataFrames to display the results for Captions and Hashtags
results_Captions = pd.DataFrame({'Captions': cleaned_Captions_})
for topic_idx in range(num_topics):
    results_Captions[f'Topic {topic_idx + 1}'] = lsa_matrix[:len(cleaned_Captions_), topic_idx]

results_Hashtags = pd.DataFrame({'Hashtags': cleaned_Hashtags_})
for topic_idx in range(num_topics):
    results_Hashtags[f'Topic {topic_idx + 1}'] = lsa_matrix[len(cleaned_Hashtags_):, topic_idx]

scores_df = data
#Sets all labels to zero for now, will update with LSA label
scores_df["LSA Feature"] = 0

#Updates labels based on LSA results using threshold of 0.001
label_list = []
for i in range(len(scores_df)):
  if results_Captions["Topic 1"][i] >= 0.001 or results_Hashtags["Topic 1"][i] >= 0.001:
    label_list.append(1)
    scores_df["LSA Feature"].iloc[i] = 1
  else:
    label_list.append(0)
    scores_df["LSA Feature"].iloc[i] = 0

scores_df = scores_df.drop(columns=["SearchedTag", "Hashtags", "Captions"])
print(scores_df)
#scores_df now has LSA feature
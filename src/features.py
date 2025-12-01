from src.utils import clean_and_preprocess

def get_cleaned_hashtags_captions(df):
    
    cleaned_Hashtags_ = []
    cleaned_Captions_ = []
    
    Hashtags = list(df["Hashtags"])
    Captions = list(df["Captions"])

    for i in range(len(df)):
        cleaned_Hashtags = clean_and_preprocess(Hashtags[i])
        cleaned_Captions = clean_and_preprocess(Captions[i])

        cleaned_Hashtags_.append(cleaned_Hashtags)
        cleaned_Captions_.append(cleaned_Captions)
        
    return cleaned_Hashtags_, cleaned_Captions_
    
def build_lsa_features(cleaned_Hashtags_, cleaned_Captions_):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    # Fit and transform the TF-IDF vectorizer on the combined text data
    tfidf_matrix = vectorizer.fit_transform(cleaned_Captions_ + cleaned_Hashtags_)
    # Perform Latent Semantic Analysis (LSA) using Truncated SVD
    num_topics = 1  # Adjust the number of topics as needed
    lsa = TruncatedSVD(n_components=num_topics, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    
    return lsa_matrix

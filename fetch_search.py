import re
import streamlit as st
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from fuzzywuzzy import fuzz
from sklearn.metrics import jaccard_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Import NLTK components after downloading data
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load your dataset
data = pd.read_csv('merged_data_fetch.csv')

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ('Home', 'Data and Preprocessing', 'Model'))

# Home Page
if page == 'Home':
    st.title("Home Page")
    st.write("Welcome! This application performs data preprocessing and allows you to search for similar offers using a model.")

    # Add a summary of the task and solution approach
    st.subheader("Task Summary:")
    st.write("The goal of this task is to build a tool for intelligent searching of offers via text input. "
             "Users can search for categories, brands, or retailers, and the tool returns relevant offers with similarity scores.")

    st.subheader("Solution Approach:")
    st.write("To achieve this, we performed the following steps:")
    st.markdown("1. **Data Preprocessing:** Cleaned and processed data, including filling missing values, converting to lowercase, removing punctuation, and handling common terms.")
    st.markdown("2. **Joining Data:** Merged 'retailer' and 'brands' datasets on 'BRAND' column, then joined the result with 'categories' dataset on 'BRAND_BELONGS_TO_CATEGORY' column.")
    st.markdown("3. **Modeling (Pending):** Implement a model to intelligently search and return relevant offers based on user input.")

    # Add a section about why you are interested in Fetch Rewards and your strengths
    st.subheader("Why Fetch Rewards?")
    st.write("I am particularly interested in Fetch Rewards because of its commitment to innovation in the consumer-engagement space. "
             "The hybrid-remote workplace and focus on rewarding shoppers align with my passion for leveraging data science to create meaningful experiences.")

    st.subheader("Why Me?")
    st.write("As a data scientist specializing in NLP and ML, I bring a unique set of skills and experiences:")
    st.markdown("- Proven track record in NLP project development, collaboration, and ML tools.")
    st.markdown("- Experience in extracting valuable information from multimedia using tools like Dexer.")
    st.markdown("- Strong analytical and predictive modeling skills, as demonstrated in previous roles.")

    st.subheader("Get Started:")
    st.write("Navigate to 'Data and Preprocessing' to explore the processed data and 'Model' to start searching for offers.")

# Data and Preprocessing Page
if page == 'Data and Preprocessing':
    st.title("Data and Preprocessing Page")
    
    st.write("For this task, I have used three datasets: 'brands', 'categories', and 'retailer'.")

    # Load additional datasets
    brands = pd.read_csv('brand_category.csv')
    categories = pd.read_csv('categories.csv')
    retailer = pd.read_csv('offer_retailer.csv')

    # Display option to show data
    show_data = st.checkbox("**Show Data**")
    
    if show_data:
        # Display shapes and info of the datasets
        st.subheader("Shapes of Datasets")
        st.text(f"Brands: {brands.shape}")
        st.text(f"Categories: {categories.shape}")
        st.text(f"Retailer: {retailer.shape}")

        # Display sample data
        st.subheader("Sample Data:")
        st.text("Brands:")
        st.dataframe(brands.head())
        st.text("Categories:")
        st.dataframe(categories.head())
        st.text("Retailer:")
        st.dataframe(retailer.head())

    # Ask user if they want to see preprocessing steps
    show_preprocessing = st.checkbox("**Show Preprocessing Steps**", value=False)
    
    # Preprocessing Data
    if show_preprocessing:
        # Fill missing values in 'RETAILER' column
        show_fillna_retailer = st.checkbox("Fill Missing Values in RETAILER column", value=False)
        if show_fillna_retailer:
            retailer['RETAILER'].fillna('Unknown', inplace=True)
            st.text("Filled missing values in the 'RETAILER' column.")

        # Convert to lowercase
        show_lowercase_conversion = st.checkbox("Convert to Lowercase", value=False)
        if show_lowercase_conversion:
            retailer = retailer.applymap(lambda x: x.lower() if type(x) == str else x)
            brands = brands.applymap(lambda x: x.lower() if type(x) == str else x)
            categories = categories.applymap(lambda x: x.lower() if type(x) == str else x)
            st.text("Converted all text to lowercase.")

        # Remove punctuation and handle common terms for PRODUCT_CATEGORY
        show_punctuation_removal_category = st.checkbox("Remove Punctuation and Handle Common Terms for PRODUCT_CATEGORY", value=False)
        if show_punctuation_removal_category:
            categories['PRODUCT_CATEGORY'] = categories['PRODUCT_CATEGORY'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
            categories['IS_CHILD_CATEGORY_TO'] = categories['IS_CHILD_CATEGORY_TO'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
            categories['PRODUCT_CATEGORY'] = categories['PRODUCT_CATEGORY'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.isalpha()]) if type(x) == str else x)
            categories['IS_CHILD_CATEGORY_TO'] = categories['IS_CHILD_CATEGORY_TO'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.isalpha()]) if type(x) == str else x)
            st.text("Removed punctuation and handled common terms for 'PRODUCT_CATEGORY'.")

        # Tokenization and Lemmatization for PRODUCT_CATEGORY
        show_lemmatization_category = st.checkbox("Tokenization and Lemmatization for PRODUCT_CATEGORY", value=False)
        if show_lemmatization_category:
            lemmatizer = WordNetLemmatizer()
            categories['PRODUCT_CATEGORY'] = categories['PRODUCT_CATEGORY'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]) if type(x) == str else x)
            categories['IS_CHILD_CATEGORY_TO'] = categories['IS_CHILD_CATEGORY_TO'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]) if type(x) == str else x)
            st.text("Performed tokenization and lemmatization for 'PRODUCT_CATEGORY'.")

        # Remove punctuation for BRAND
        show_punctuation_removal_brands = st.checkbox("Remove Punctuation for Brands", value=False)
        if show_punctuation_removal_brands:
            brands['BRAND'] = brands['BRAND'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
            brands['BRAND_BELONGS_TO_CATEGORY'] = brands['BRAND_BELONGS_TO_CATEGORY'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
            st.text("Removed punctuation for 'BRAND'.")

        # Remove noise and handle common terms for RETAILER
        show_common_terms_removal_retailer = st.checkbox("Remove Noise and Handle Common Terms for RETAILER", value=False)
        if show_common_terms_removal_retailer:
            retailer['RETAILER'] = retailer['RETAILER'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
            retailer['RETAILER'] = retailer['RETAILER'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.isalpha()]) if type(x) == str else x)
            st.text("Removed noise and handled common terms for 'RETAILER'.")

        # Remove noise and handle common terms for BRAND
        show_common_terms_removal_brands = st.checkbox("Remove Noise and Handle Common Terms for BRAND", value=False)
        if show_common_terms_removal_brands:
            retailer['BRAND'] = retailer['BRAND'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
            retailer['BRAND'] = retailer['BRAND'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.isalpha()]) if type(x) == str else x)
            st.text("Removed noise and handled common terms for 'BRAND'.")

        # Tokenization and Lemmatization for OFFER
        show_lemmatization_offer = st.checkbox("Tokenization and Lemmatization for OFFER", value=False)
        if show_lemmatization_offer:
            lemmatizer = WordNetLemmatizer()
            retailer['OFFER'] = retailer['OFFER'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]) if type(x) == str else x)
            st.text("Performed tokenization and lemmatization for 'OFFER'.")

        # Remove stopwords for OFFER
        show_stopwords_removal_offer = st.checkbox("Remove Stopwords for OFFER", value=False)
        if show_stopwords_removal_offer:
            stop_words = set(stopwords.words('english'))
            retailer['OFFER'] = retailer['OFFER'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]) if type(x) == str else x)
            st.text("Removed stopwords for 'OFFER'.")

        # Display the preprocessed data
        st.subheader("Preprocessed Data:")
        st.text("Preprocessed Retailer Data:")
        st.dataframe(retailer.head())
        st.text("Preprocessed Brands Data:")
        st.dataframe(brands.head())
        st.text("Preprocessed Categories Data:")
        st.dataframe(categories.head())

    # Checkbox to show joining data steps
    show_joining_data = st.checkbox("**Show Joining Data Steps**", value=False)

    # Joining Data
    if show_joining_data:
        st.subheader("Joining Data Steps:")
        
        # Joining on 'BRAND'
        merged_data = pd.merge(retailer, brands, on='BRAND', how='outer')
        st.text("1. Performed an outer join between 'retailer' and 'brands' datasets on the 'BRAND' column. This type of join includes all rows from both datasets, filling in missing values with NaN for rows that do not have corresponding 'BRAND' values in both datasets. The choice of an outer join was made to retain all available records, ensuring comprehensive information for the analysis.")

        # Joining on 'BRAND_BELONGS_TO_CATEGORY'
        merged_data = pd.merge(merged_data, categories, left_on='BRAND_BELONGS_TO_CATEGORY', right_on='PRODUCT_CATEGORY', how='inner')
        merged_data = merged_data.drop('BRAND_BELONGS_TO_CATEGORY', axis=1)
        st.text("2. Joined the above result with 'categories' dataset on 'BRAND_BELONGS_TO_CATEGORY' column.")
        
        # Display the merged data
        st.subheader("Merged Data:")
        st.dataframe(merged_data)


# Function to preprocess user input using fuzzy matching
def preprocess_user_input(user_input, target_column):
    # Check similarity using fuzzy matching
    return data[target_column].apply(lambda x: preprocess_text_fuzzy(str(x).lower() if pd.notna(x) else '', user_input.lower()))

# Function to preprocess text using fuzzy matching
def preprocess_text_fuzzy(text, target):
    # Check similarity using fuzzy matching
    if fuzz.ratio(text, target) >= 50:
        return target
    return text
    

# Model Page
if page == 'Model':
    st.title("Offer Similarity Analysis from Brands, Category, Retailer Search")

    # Add a selection bar for choosing the model
    selected_model = st.sidebar.radio("Select Model:", ("TF-IDF", "Neural Networks"))

    # TF-IDF Model
    if selected_model == "TF-IDF":
        # Select option from brand, category, or retailer
        option = st.selectbox("Select option:", ('Brand', 'Category', 'Retailer'))

        # Display search bar based on the selected option
        if option:
            search_query = st.text_input(f"Enter {option} for search:")

            # Apply model based on user input
            if option == 'Brand':
                input_column = 'BRAND'
            elif option == 'Category':
                input_column = 'PRODUCT_CATEGORY'
            elif option == 'Retailer':
                input_column = 'RETAILER'

            # Use only the 'OFFER' column for text similarity calculation
            data['combined_text'] = data['OFFER']

            # Fill NaN values in the 'combined_text' column with an empty string
            data['combined_text'] = data['combined_text'].fillna('')

            # Exclude rows where 'OFFER' is empty or NaN
            data = data.dropna(subset=['OFFER'])
            data = data[data['OFFER'] != '']

            # Create a TF-IDF vectorizer
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(data['combined_text'])

            # Transform the user input using the same vectorizer
            user_tfidf = vectorizer.transform([search_query])

            # Calculate Cosine similarity between user input and each offer
            cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

            # Calculate Jaccard similarity between user input and each offer
            jaccard_similarities = []
            for offer_text in data['combined_text']:
                offer_words = set(offer_text.split())
                user_words = set(search_query.split())
                jaccard = len(offer_words.intersection(user_words)) / len(offer_words.union(user_words))
                jaccard_similarities.append(jaccard)

            # Add the similarity scores to the data DataFrame
            data['Cosine Similarity'] = cosine_similarities
            data['Jaccard Similarity'] = jaccard_similarities

            # Sort offers based on both similarity scores
            sorted_data = data.sort_values(by=['Cosine Similarity', 'Jaccard Similarity'], ascending=False)

            # Filter results based on user input criteria
            filtered_data = sorted_data[sorted_data[input_column].str.lower().str.contains(search_query.lower(), na=False)]

            # Display results only if there is a search query
            if search_query:
                if not filtered_data.empty:
                    st.header('Top Similar Offers:')
                    st.dataframe()
                else:
                    st.info(f"No offers found for the given search query: '{search_query}'.")

                check_future_offers = st.checkbox("Check for offers in the future")

                if check_future_offers:
                    st.info("Checking for future offers... (This functionality is not yet implemented)")
            else:
                st.info("Enter a search query to view results.")

    elif selected_model == "Neural Networks":
        st.title("Neural Networks Model")
            
        option = st.selectbox("Select option:", ('Brand', 'Category', 'Retailer'))
            
        if option:
            search_query = st.text_input(f"Enter {option} for search:")
            
            if search_query:
                # Function to get relevant offers for a user input retailer
                def get_relevant_offers_for_retailer(user_input_retailer):
                    # Transform the user input using the tokenizer
                    user_input_sequence_retailer = tokenizer_retailer.texts_to_sequences([user_input_retailer])
                    user_input_sequence_retailer = pad_sequences(user_input_sequence_retailer, maxlen=X_retailer.shape[1])
    
                    # Repeat the user input sequence for each offer
                    user_input_sequence_retailer_broadcasted = np.repeat(user_input_sequence_retailer, X_offer.shape[0], axis=0)
    
                    # Predict the similarity scores for the user input
                    user_pred = model.predict([X_offer, user_input_sequence_retailer_broadcasted, X_brand])
    
                    # Take the absolute value of the predicted scores
                    user_pred_absolute = np.abs(user_pred)
    
                    # Inverse transform to get the original scale
                    user_pred_original = scaler.inverse_transform(user_pred_absolute).flatten()
    
                    # Create a DataFrame with the relevant information
                    result_df = pd.DataFrame({
                        'OFFER': merged_data['OFFER'],
                        'RETAILER': merged_data['RETAILER'],
                        'BRAND': merged_data['BRAND'],
                        'PRODUCT_CATEGORY': merged_data['PRODUCT_CATEGORY'],
                        'predicted_score': user_pred_original
                    })
    
                    # Filter offers for the specified retailer
                    result_df = result_df[result_df['RETAILER'].str.lower() == user_input_retailer.lower()]
    
                    # Sort offers based on predicted score
                    result_df = result_df.sort_values(by='predicted_score', ascending=False)
    
                    # Reset index
                    result_df.reset_index(drop=True, inplace=True)
    
                    return result_df.head(10)  # Adjust the number of rows as needed
    
                result_df = get_relevant_offers_for_retailer(search_query.lower())
    
                if not result_df.empty:
                    st.header('Top Similar Offers:')
                    st.dataframe(result_df[['BRAND', 'OFFER', 'predicted_score']])
                else:
                    st.info(f"No offers found for the given search query: '{search_query}'.")
    
                check_future_offers = st.checkbox("Check for offers in the future")
    
                if check_future_offers:
                    st.info("Checking for future offers... (This functionality is not yet implemented)")
            else:
                st.info("Enter a search query to view results.")

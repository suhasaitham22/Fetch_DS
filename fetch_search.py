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
        merged_data = pd.merge(retailer, brands, on='BRAND', how='inner')
        st.text("1. Performed an inner join between 'retailer' and 'brands' datasets on the 'BRAND' column. This type of join retains only the rows where there is a match in the 'BRAND' column between both datasets. The choice of an inner join was made to keep only the records that have corresponding 'BRAND' values in both datasets, ensuring that we have relevant information for the analysis.")

        # Joining on 'BRAND_BELONGS_TO_CATEGORY'
        merged_data = pd.merge(merged_data, categories, left_on='BRAND_BELONGS_TO_CATEGORY', right_on='PRODUCT_CATEGORY', how='inner')
        merged_data = merged_data.drop('BRAND_BELONGS_TO_CATEGORY', axis=1)
        st.text("2. Joined the above result with 'categories' dataset on 'BRAND_BELONGS_TO_CATEGORY' column.")
        
        # Display the merged data
        st.subheader("Merged Data:")
        st.dataframe(merged_data)

# Model Page
if page == 'Model':
    st.title("Model Page")

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

        # Normalize words (convert to lowercase, remove punctuation, etc.)
        data['combined_text'] = data['combined_text'].apply(lambda x: ' '.join(word_tokenize(x.lower()) if type(x) == str else x))
        search_query = ' '.join(word_tokenize(search_query.lower()))

        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data['combined_text'])

        # Transform the user input using the same vectorizer
        user_tfidf = vectorizer.transform([search_query])

        # Calculate cosine similarity between user input and each offer
        cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

        # Add the similarity scores to the data DataFrame
        data['Similarity Score'] = cosine_similarities

        # Sort offers based on similarity score
        sorted_data = data.sort_values(by='Similarity Score', ascending=False)

        # Filter results based on user input criteria
        filtered_data = sorted_data[sorted_data[input_column].str.lower().str.contains(search_query.lower(), na=False)]

        # Display results only if there is a search query
        if search_query:
            st.header('Top Similar Offers:')
            st.dataframe(filtered_data[[input_column, 'OFFER', 'Similarity Score']])
        else:
            st.info("Enter a search query to view results.")

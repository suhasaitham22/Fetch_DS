import streamlit as st
import pandas as pd
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Load your dataset
data = pd.read_csv('C:\\Users\\SUHAS\\Downloads\\Fetch Challenge\\merged_data_fetch.csv')
# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ('Home', 'Data and Preprocessing', 'Model'))


# Data and Preprocessing Page
if page == 'Data and Preprocessing':
    st.title("Data and Preprocessing Page")

    # Load additional datasets
    brands = pd.read_csv('C:\\Users\\SUHAS\\Downloads\\Fetch Challenge\\brand_category.csv')
    categories = pd.read_csv('C:\\Users\\SUHAS\\Downloads\\Fetch Challenge\\categories.csv')
    retailer = pd.read_csv('C:\\Users\\SUHAS\\Downloads\\Fetch Challenge\\offer_retailer.csv')

    # Display shapes and info of the datasets
    st.subheader("Shapes of Datasets")
    st.text(f"Brands: {brands.shape}")
    st.text(f"Categories: {categories.shape}")
    st.text(f"Retailer: {retailer.shape}")

    st.subheader("Info of Datasets")
    st.text("Brands:")
    st.text(brands.info())
    st.text("Categories:")
    st.text(categories.info())
    st.text("Retailer:")
    st.text(retailer.info())

    # Display sample data
    st.subheader("Sample Data:")
    st.text("Brands:")
    st.dataframe(brands.head())
    st.text("Categories:")
    st.dataframe(categories.head())
    st.text("Retailer:")
    st.dataframe(retailer.head())

    # Preprocessing Data

    # Fill missing values in 'RETAILER' column
    retailer['RETAILER'].fillna('Unknown', inplace=True)

    # Convert to lowercase
    retailer = retailer.applymap(lambda x: x.lower() if type(x) == str else x)
    brands = brands.applymap(lambda x: x.lower() if type(x) == str else x)
    categories = categories.applymap(lambda x: x.lower() if type(x) == str else x)

    # Remove punctuation and handle common terms
    categories['PRODUCT_CATEGORY'] = categories['PRODUCT_CATEGORY'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
    categories['IS_CHILD_CATEGORY_TO'] = categories['IS_CHILD_CATEGORY_TO'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
    categories['PRODUCT_CATEGORY'] = categories['PRODUCT_CATEGORY'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.isalpha()]) if type(x) == str else x)
    categories['IS_CHILD_CATEGORY_TO'] = categories['IS_CHILD_CATEGORY_TO'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.isalpha()]) if type(x) == str else x)

    # Tokenization and Lemmatization
    lemmatizer = WordNetLemmatizer()
    categories['PRODUCT_CATEGORY'] = categories['PRODUCT_CATEGORY'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]) if type(x) == str else x)
    categories['IS_CHILD_CATEGORY_TO'] = categories['IS_CHILD_CATEGORY_TO'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]) if type(x) == str else x)

    # Remove punctuation
    brands['BRAND'] = brands['BRAND'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
    brands['BRAND_BELONGS_TO_CATEGORY'] = brands['BRAND_BELONGS_TO_CATEGORY'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)

    # Remove noise and handle common terms for PRODUCT_CATEGORY
    categories['PRODUCT_CATEGORY'] = categories['PRODUCT_CATEGORY'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
    categories['PRODUCT_CATEGORY'] = categories['PRODUCT_CATEGORY'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.isalpha()]) if type(x) == str else x)

    # Remove noise and handle common terms for IS_CHILD_CATEGORY_TO
    categories['IS_CHILD_CATEGORY_TO'] = categories['IS_CHILD_CATEGORY_TO'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
    categories['IS_CHILD_CATEGORY_TO'] = categories['IS_CHILD_CATEGORY_TO'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.isalpha()]) if type(x) == str else x)

    # Tokenization and Lemmatization
    lemmatizer = WordNetLemmatizer()
    brands['BRAND'] = brands['BRAND'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]) if type(x) == str else x)
    brands['BRAND_BELONGS_TO_CATEGORY'] = brands['BRAND_BELONGS_TO_CATEGORY'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]) if type(x) == str else x)

    # Remove punctuation
    retailer['OFFER'] = retailer['OFFER'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
    retailer['RETAILER'] = retailer['RETAILER'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
    retailer['BRAND'] = retailer['BRAND'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)

    # Tokenization and Lemmatization
    lemmatizer = WordNetLemmatizer()
    retailer['OFFER'] = retailer['OFFER'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]) if type(x) == str else x)
    retailer['RETAILER'] = retailer['RETAILER'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]) if type(x) == str else x)
    retailer['BRAND'] = retailer['BRAND'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]) if type(x) == str else x)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    retailer['OFFER'] = retailer['OFFER'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]) if type(x) == str else x)

    # Remove noise and handle common terms for RETAILER
    retailer['RETAILER'] = retailer['RETAILER'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
    retailer['RETAILER'] = retailer['RETAILER'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.isalpha()]) if type(x) == str else x)

    # Remove noise and handle common terms for BRAND
    retailer['BRAND'] = retailer['BRAND'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
    retailer['BRAND'] = retailer['BRAND'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.isalpha()]) if type(x) == str else x)

    # Display the preprocessed data
    st.subheader("Preprocessed Data:")
    st.text("Preprocessed Retailer Data:")
    st.dataframe(retailer.head())
    st.text("Preprocessed Brands Data:")
    st.dataframe(brands.head())
    st.text("Preprocessed Categories Data:")
    st.dataframe(categories.head())

# Model Page
if page == 'Model':
    st.title("Model Page")

    # Select option from brand, category, or retailer
    option = st.selectbox("Select option:", ('Brand', 'Category', 'Retailer'))

    # Display search bar based on the selected option
    if option:
        search_query = st.text_input(f"Enter {option} for search:")
        
        # Filter data based on user input
        if option == 'Brand':
            filtered_data = data[data['BRAND'].str.lower().str.contains(search_query.lower())]
        elif option == 'Category':
            filtered_data = data[data['PRODUCT_CATEGORY'].str.lower().str.contains(search_query.lower())]
        elif option == 'Retailer':
            filtered_data = data[data['RETAILER'].str.lower().str.contains(search_query.lower())]

        # Display results
        st.header('Top Similar Offers:')
        st.dataframe(filtered_data)
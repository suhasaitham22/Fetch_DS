import streamlit as st
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

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
        # Display sample data
        st.subheader("Sample Data:")
        st.text("Brands:")
        st.dataframe(brands.head())
        st.text("Categories:")
        st.dataframe(categories.head())
        st.text("Retailer:")
        st.dataframe(retailer.head())

    # Ask user if they want to see preprocessing steps
    show_preprocessing = st.checkbox("Show Preprocessing Steps", value=False)
    
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
        show_joining_data = st.checkbox("Show Joining Data Steps", value=False)

        # Joining Data
        if show_joining_data:
            st.subheader("Joining Data Steps:")
            
            # Joining on 'BRAND'
            merged_data = pd.merge(retailer, brands, on='BRAND', how='inner')
            st.text("1. The inner join was used to merge the 'retailer' and 'brands' datasets based on the common 'BRAND' column. This type of join retains only the rows where there is a match in the 'BRAND' column between both datasets. The choice of an inner join was made to keep only the records that have corresponding 'BRAND' values in both datasets, ensuring that we have relevant information for the analysis.")

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

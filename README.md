# Offer Similarity Analysis

This repository contains code for an application that performs similarity analysis on offers based on brands, categories, and retailers. The application allows users to search for specific brands, categories, or retailers and retrieves similar offers using text similarity metrics.

## Streamlit App

[Link to the Streamlit App](https://fetchds-ubwbhytyb8xf45gvt5oghv.streamlit.app/)

## Overview

The repository includes the following files:

- `brand_category.csv`: Dataset containing brand-category mapping information.
- `categories.csv`: Dataset with details about product categories.
- `fetch_search.py`: Python file containing code for the offer similarity analysis application.
- `merged_data_fetch.csv`: Merged dataset after combining retailer, brand, and category information.
- `offer_retailer.csv`: Dataset comprising retailer and offer information.
- `requirements.txt`: File listing the required Python packages.

## Instructions

To run the application locally, follow these steps:
1. Clone this repository to your local machine.
2. Ensure all dataset files (`brand_category.csv`, `categories.csv`, `offer_retailer.csv`) are in the same directory as `fetch_search.py`.
3. Install the necessary Python packages listed in `requirements.txt`.
4. Run the `fetch_search.py` file using Python.
5. Access the application in your web browser.

## Usage

Once the application is running, you can navigate through different functionalities:
- Search for similar offers based on brands, categories, or retailers.
- Explore the merged dataset or individual datasets for brand, category, and retailer information.
- Check the code in `fetch_search.py` for the implementation details.

## Future Improvements

- Incorporating user authentication for personalized search experiences.
- Building a recommendation system for users based on their search history.

Feel free to explore the code and datasets or contribute to enhancing the functionality of the application.

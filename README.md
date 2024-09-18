# Predicting Home Prices Using the Ames Housing Dataset

This project aims to predict housing prices in Ames, Iowa using the Kaggle Ames Housing Dataset, which contains 2,919 housing sale records between 2006-2010 with 79 features for each house. The project implements a K-Nearest Neighbors (KNN) algorithm to estimate the price of a new house based on historical data. Below is an outline of the project:

## Dataset
- **Source**: Kaggle's Ames Housing Dataset
- **Size**: 2,919 observations and 79 features
- **Features**: Information about house size, quality, area, age, and other attributes related to housing prices in Ames, Iowa.

## Project Overview
This project involves the following steps:
1. **Data Preprocessing**:
   - Loaded the dataset as a Pandas DataFrame.
   - Cleaned and handled missing values.
   - Performed exploratory data analysis to understand key features.
   
2. **Key Feature Analysis**:
   - Analyzed the relationship between important features like `OverallQual`, `YearBuilt`, `TotalBsmtSF`, `GrLivArea`, and `SalePrice`.
   - Performed correlation analysis and generated descriptive statistics.
   
3. **Feature Engineering**:
   - Created new features such as `TotalArea` (sum of basement area and ground living area) and `AreaPerRoom` (average room size).
   
4. **Modeling**:
   - Implemented the K-Nearest Neighbors (KNN) algorithm to estimate housing prices.
   - Predicted the price of a new house (test dataset) by comparing it to 5 similar houses based on Euclidean distance.
   
5. **Results**:
   - Predicted price for the target house was **$121,080**.
   - Used the average price of the 5 nearest neighbors to generate the prediction.

## Files
- `HousingData_processed.csv`: Processed dataset with key features selected for modeling.
- `PredictionModel.ipynb`: Jupyter notebook containing all code for data processing, feature analysis, and model implementation.

## Requirements
- Python 3.x
- Libraries:
  - Pandas
  - NumPy
  - Matplotlib
  - scikit-learn

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/gksdusql94/ML_House.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter notebook and run all cells:
    ```bash
    jupyter notebook PredictionModel.ipynb
    ```

## Conclusion
This project demonstrates how to use machine learning techniques like K-Nearest Neighbors for regression tasks such as predicting housing prices. The preprocessing and feature engineering steps are crucial for improving model performance, and this project can be further extended by experimenting with more advanced algorithms like XGBoost or Random Forest.

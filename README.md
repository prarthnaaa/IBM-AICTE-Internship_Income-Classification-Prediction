# Income Classification Prediction App

This project was developed as part of the **EDUNET FOUNDATION - IBM SKILLSBUILD - ARTIFICIAL INTELLIGENCE - 6 WEEKS INTERNSHIP (June 2025 Batch)**.

It aims to build a machine learning model and deploy a web application that predicts whether an individual's income exceeds \$50K based on demographic and socio-economic features. It uses various classification algorithms, with XGBoost achieving the best performance.

## Project Overview

* **Problem**: Classify individuals as earning `<=50K` or `>50K` annually.
* **Dataset**: [Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult) from the UCI Machine Learning Repository.
* **Features**: Age, education, gender, occupation, marital status, etc.
* **Target**: Income (`<=50K`, `>50K`)

## Project Structure

```
📁 prarthnaaa/
├── app.py                                # Streamlit web app
├── income_predictor.ipynb                # Model training and evaluation notebook
├── dataset.csv                           # Cleaned input dataset
├── best_model.pkl                        # Trained XGBoost model
├── encoders.pkl                          # Label encoders for categorical features
├── model_test_dataset.csv                # Dataset with predictions
├── 2025-07-21T14-48_export.csv           # Streamlit export with predicted income
├── PrarthnaPuhan-IBMProject_PPT.pptx     # Project presentation
└── README.md                             # Project documentation
```

## Technologies Used

* Python (pandas, numpy, scikit-learn, xgboost, imbalanced-learn)
* Streamlit (for web deployment)
* Matplotlib / Seaborn (for visualization)

## Results

* **Best Model**: XGBoost with accuracy **89.19%**
* Other models tested: Random Forest, Logistic Regression, SVM, KNN, Naive Bayes, etc.

## Dataset Information

* **Source**: [UCI Adult Dataset](http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html)
* **Attributes**: 14 demographic features + 1 target
* **Target Classes**: `<=50K`, `>50K`

## How to Run

1. Clone the repository
2. Install dependencies
3. Run the app using:

```bash
streamlit run app.py
```

## Acknowledgements

* Internship: Edunet Foundation - IBM SkillsBuild - AI Internship (June 2025 Batch)
* Dataset: UCI Machine Learning Repository – Adult Income Dataset

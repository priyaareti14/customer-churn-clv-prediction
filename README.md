# Customer Churn Prediction and CLV Analysis

## Overview
This project analyzes 7,043 telecom customers to identify churn drivers, 
quantify revenue at risk, and prioritize high-risk customers for retention 
intervention using K-Means clustering, Logistic Regression, and Random Forest.

## Business Question
Which customers are most likely to churn, and what is the revenue at risk?

## Tools and Libraries
- R (tidyverse, ggplot2)
- K-Means Clustering (cluster)
- Random Forest (randomForest)
- Logistic Regression (base R glm)
- caTools (train/test split)

## Project Workflow
1. Data cleaning and feature engineering
2. K-Means clustering to segment customers into 4 profiles
3. CLV calculation (Monthly Charges x Tenure)
4. Churn prediction using Logistic Regression and Random Forest
5. Risk Score calculation (Churn Probability x Normalized CLV)
6. Identification of high-risk customers for retention targeting

## Models and Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 78.71% |
| Random Forest | 78.71% |

## Key Findings
- 26.5% churn rate representing $2.86M in annual revenue at risk
- 4 distinct customer segments identified via K-Means clustering
- Contract type was the strongest churn predictor — month-to-month 
  customers churned at 3x the rate of two-year contract holders
- 1,761 high-risk customers identified using a combined Risk Score 
  of churn probability and normalized CLV

## Dataset
IBM Telco Customer Churn, publicly available on Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

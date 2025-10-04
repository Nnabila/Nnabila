## Hi there 👋

# Autism Spectrum Disorder Detection Using Machine Learning Approach

## Overview
This project applies simple machine learning methods to help detect Autism Spectrum Disorder (ASD) based on behavioral and screening data.  
It aims to assist early identification and provide a supportive tool for healthcare professionals.

## Background
This work is inspired by my previous research paper:  
[Autism Spectrum Disorder Detection Using Machine Learning Approach (IEEE 2021)](https://ieeexplore.ieee.org/document/9579522)

## Features
- Uses supervised machine learning algorithms  
- Works with behavioral and screening questionnaire data  
- Provides basic accuracy and performance reports  

## How It Works
1. Input dataset (CSV file with behavioral and screening features)  
2. Choose a machine learning model (e.g., Decision Tree, SVM, Random Forest)  
3. Train and evaluate the model  
4. Display accuracy and confusion matrix  

## Example Code (Simplified)
```python
# example.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv("asd_data.csv")
X = data.drop("ASD", axis=1)
y = data["ASD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

<!--
**Nnabila/Nnabila** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

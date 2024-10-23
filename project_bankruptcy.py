# %% [markdown]
# This project works on `Company Bankruptcy Prediction` dataset with different versions of `Random Forest` to create prediction about possible bankruptcy status of companies. 
# 
# The best version could be used as a supporting tool for investors by helping them avoid companies predicted with bankruptcy status.

# %% [markdown]
# # 1. Data Preparation

# %%
# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# %%
# Read the dataset
df = pd.read_csv("datasets/bankruptcy.csv")
df.head()

# %%
# Dataset overview
df.info()

# %%
df.describe()

# %%
# Check nulls and duplicates
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
# Check class frequency
df["Bankrupt?"].value_counts(normalize=True)

# %%
# Check dataset size
df.shape

# %% [markdown]
# Observation:
# * The dataset has 6819 records and 96 columns. Columns includes financial attributes and their bankruptcy status in `Bankrupt?`. 
# * Datatype are `float64` or `int64` in similar range from 0 to 1. 
# * There is no null or duplicated values. 
# 
# Problem:
# * The target column `Bankrupt?` is imbalanced, with only 3% of total values is `1`, which indicates the bankrupt status.

# %% [markdown]
# # 2. Exploratory Data Analysis 
# 

# %%
# Check corelations between features using heatmap

# Create correlation matrix
corr_matrix = df.corr()

# Generate the heatmap
plt.figure(figsize=(30, 25))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# %%
# Set benchmark for features correlation significance with target
corr_features = corr_matrix["Bankrupt?"].abs() >= 0.15
corr_features.value_counts()

# %%
# Get list of names for features passing the benchmark 
feature_names = list(corr_features[corr_features].index)
feature_names

# %%
# Check corelations between selected features

# Create correlation matrix
mini_corr_matrix = df[feature_names].corr()

# Generate the heatmap
plt.figure(figsize=(30, 25))
sns.heatmap(mini_corr_matrix, annot = True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# %% [markdown]
# Observation:
# * Assume 0.15 is the benchmark for correlation significance, so only 22 attributes have at least weak correlation with the target. 
# * The rest attributes have correlation with target less than the benchmark, which means they have too weak or even no relation with the target.
# 
# Problem:
# * The choice based on correlation significance with the target only might not be the best choice. The reason is that correlation assume linear relationship between features.

# %%
# Create a figure with subplots
fig, axes = plt.subplots(len(feature_names), 2, figsize=(16, len(feature_names) * 4))

for i, feature in enumerate(feature_names):
    # Plot with original feature
    sns.boxenplot(x="Bankrupt?", y=feature, data=df, ax=axes[i, 0])
    axes[i, 0].set_title(f'Original {feature}')

    # Plot after removing outliers in feature
    q1, q9 = df[feature].quantile([0.1, 0.9])
    mask = df[feature].between(q1, q9)
    sns.boxplot(x="Bankrupt?", y=feature, data=df[mask], ax=axes[i, 1])
    axes[i, 1].set_title(f'{feature} without Outliers')

# Adjust layout
plt.tight_layout()
plt.show()

# %% [markdown]
# Problem:
# * The dataset has outliers.

# %% [markdown]
# Solutions for imbalanced dataset, feature selection and outliers:
# 
# Choose `Random Forest` model:
# * For imbalanced data: Diverse trees and result voting helps this model reducing bias towards majority class.
# * For feature selection: Random features selection and feature importance scores helps this model identify informative features and remove irrelevant ones.
# * For outliers: Result voting among trees makes this model less sensitive to outliers.
# 
# Use `Resampling` for data processing:
# * With high imbalance ratio, this technique is use to reduce majority class size or increase minority class size. As a result, the bias towards majority class is reduced.
# 
# Focus on `Recall` and `F1-score` evaluation metrics for class `1`:
# * Wrong investment due to fasle negative (incorrect prediction of company will not bankrupt when it actually does) is more costly. Therefore, false negative is aimed to be reduced, which means `Recall` is aimed to be increased.
# * Since precision-recall is a trade-off relationship, `F1-score` need is be used to ensure balance between them. This metric is also aimed to be increased.

# %% [markdown]
# # 3. Random Forest 

# %%
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.metrics import ConfusionMatrixDisplay, classification_report # Evaluation metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split # Data splitting
from imblearn.under_sampling import RandomUnderSampler # Resampling
from imblearn.over_sampling import RandomOverSampler 
from imblearn.over_sampling import SMOTE

# %%
# Split features and target
target = "Bankrupt?"
X = df.drop(columns=[target])
y = df[target]

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# A `Random Forest` is a combines multiple decision trees to improve the overall accuracy and stability of predictions.
# 
# How it works:
# * Bagging (Bootstrapping): The algorithm starts by creating multiple subsets of the original data with replacement. 
# * Building Decision Trees: For each bootstrap sample, a decision tree is built with random subset of features. 
# * Ensemble Voting: When making a prediction, each tree in the forest votes on the most likely outcome. For classification problems, the majority vote wins.
# 
# Advantages:
# 
# * High accuracy: Beter accuracy because it average out the errors of individual trees.
# * Robustness to overfitting: The randomness in the feature selection and tree building process reduces overfitting. 
# * Can handle high dimensional data
# * Interpretability: The overall behavior can be understood by examining the importance of different features in the predictions.

# %% [markdown]
# ## a. Baseline Model
# 

# %%
# Create a Random Forest classifier with default parameters
rf_baseline = RandomForestClassifier()

# Train the model on the training data
rf_baseline.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = rf_baseline.predict(X_test)

# Evaluate using evaluation scores
accuracy_base = accuracy_score(y_test, y_pred)
print("Baseline Accuracy:", accuracy_base)
recall_base = recall_score(y_test, y_pred)
print("Baseline Recall:", recall_base)
f1_base = f1_score(y_test, y_pred)
print("Baseline F1-score:", f1_base)

# %% [markdown]
# Accuracy alone is not enough for imbalanced data. Confusion matrix and classficiation report with baseline `Recall` and `F1-score` for class `1` also need to be considered.

# %%
# Evaluate using confusion matrix
ConfusionMatrixDisplay.from_estimator(rf_baseline, X_test, y_test)

# %%
# Evaluate using classification report
print(classification_report(y_test, y_pred))

# %% [markdown]
# ## b. Model With Undersampling

# %% [markdown]
# `Undersampling` is a technique used to address class imbalance. It works by reducing the number of data points in the majority class, bringing it closer to the size of the minority class. 

# %%
# Undersample the training data
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# Create and train the Random Forest model
rf_rus = RandomForestClassifier(random_state=42)
rf_rus.fit(X_train_resampled, y_train_resampled)

# Predict the labels for the test data
y_pred = rf_rus.predict(X_test)

# Evaluate using evaluation scores
accuracy_rus = accuracy_score(y_test, y_pred)
print("Undersampling Accuracy:", accuracy_rus)
recall_rus = recall_score(y_test, y_pred)
print("Undersampling Recall:", recall_rus)
f1_rus = f1_score(y_test, y_pred)
print("Undersampling F1-score:", f1_rus)

# %%
# Evaluate using confusion matrix
ConfusionMatrixDisplay.from_estimator(rf_rus, X_test, y_test)

# %%
# Evaluate using classification report
print(classification_report(y_test, y_pred))

# %% [markdown]
# In this technique, `Recall` achieve much higher rate with the trade-off with `Precision`, leading to lower `F1-score` than the baseline for class `1`. It means investors will be too safe if they based on this prediction version, and they will loose the chance to invest in many companies that is well-operarting but predicted as dealing with bankruptcy. The overall accuracy also drops significantly compared to the default model. 
# 
# `Undersampling` is not a good solution for the imbalanced dataset and `Random Forest` model in this task. 

# %% [markdown]
# ## c. Model With Oversampling

# %% [markdown]
# `Oversampling` is another technique used to address class imbalance. It works by increasing the number of data points in the minority class, bringing it closer to the size of the majority class. 

# %%
# Oversample the training data
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Create and train the Random Forest model
rf_ros = RandomForestClassifier(random_state=42)
rf_ros.fit(X_train_resampled, y_train_resampled)

# Predict the labels for the test data
y_pred = rf_ros.predict(X_test)

# Evaluate using evaluation scores
accuracy_ros = accuracy_score(y_test, y_pred)
print("Oversampling Accuracy:", accuracy_ros)
recall_ros = recall_score(y_test, y_pred)
print("Oversampling Recall:", recall_ros)
f1_ros = f1_score(y_test, y_pred)
print("Oversampling F1-score:", f1_ros)

# %%
# Evaluate using confusion matrix
ConfusionMatrixDisplay.from_estimator(rf_ros, X_test, y_test)

# %%
# Evaluate using classification report
print(classification_report(y_test, y_pred))

# %% [markdown]
# This technique get increases in both `Recall` and `F1-score` for class `1`, which is the aim been looking for. This helps to reduce the risk of wrong investment to bankrupt companies predicted as normal. 
# 
# `Oversampling` for `Random Forest` is the better version compared to baseline model and model with `Undersampling`.
# 

# %% [markdown]
# ## d. Model With SMOTE

# %% [markdown]
# `SMOTE` (Synthetic Minority Oversampling Technique) is a technique used to address class imbalance in datasets by creating synthetic examples of the minority class. It identitfies the minority class, selects nearest neighbors, and generates synthetic examples for each minority class example with their neighbors.

# %%
# Oversample the training data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create and train the Random Forest model
rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_resampled, y_train_resampled)

# Predict the labels for the test data
y_pred = rf_smote.predict(X_test)

# Evaluate using evaluation scores
accuracy_smote = accuracy_score(y_test, y_pred)
print("SMOTE Accuracy:", accuracy_smote)
recall_smote = recall_score(y_test, y_pred)
print("SMOTE Recall:", recall_smote)
f1_smote = f1_score(y_test, y_pred)
print("SMOTE F1-score:", f1_smote)

# %% [markdown]
# # 5. Feature Selection 

# %% [markdown]
# `Random Forest` offers `Feature Importance Score` to support better interpretation in feature selection. This method could use `Gini Importance` as measurement.
# 
# `Gini Importance` is a measure of how much a feature contributes to the decrease in Gini impurity across all trees in a random forest. A lower Gini impurity indicates a more pure node, where most data points belong to the same class.
# 
# The best model `Random Forest` with `SMOTE` resampling technique will apply feature selection to show most relevant features. 

# %%
# Find important features
features = X_test.columns
importances = rf_smote.feature_importances_

# Create dataframe of sorted features based on importance
feature_importances_df = pd.DataFrame({
    "Feature": features,
    "Gini Importance": importances
})

feature_importances_df = feature_importances_df.sort_values(by="Gini Importance", ascending=False)
feature_importances_df.head()

# %%
# Visualize top 10 important features
plt.figure(figsize=(15, 8))  
sns.barplot(x="Gini Importance", y="Feature", data=feature_importances_df.head(10), palette="viridis")

plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.show()

# %% [markdown]
# # 6. Conclusion 

# %%
# Create dataframe to compare performance
accuracy = [accuracy_base, accuracy_rus, accuracy_ros, accuracy_smote]
recall = [recall_base, recall_rus, recall_ros, recall_smote]
f1 = [f1_base, f1_rus, f1_ros, f1_smote]
model = ['RF Baseline', 'RF with Undersampling', 'RF with Oversampling',
       'RF with SMOTE']

compare = pd.DataFrame({'Model': model, 'Accuracy': accuracy, 'Recall': recall, 'F1-score': f1})
compare

# %%
# Create figure with size
plt.figure(figsize=(15, 8))  

# Create a point plot using Seaborn
sns.pointplot(x='Model', y='Accuracy', data=compare, color='#F72585', label='Accuracy')
sns.pointplot(x='Model', y='Recall', data=compare, color='#7209B7', label='Recall')
sns.pointplot(x='Model', y='F1-score', data=compare, color='#3A0CA3', label='F1-score')

# Add labels and title
plt.xlabel('Model')
plt.ylabel('Evaluation Metric')
plt.title('Evaluation Metrics for Different Models')

# Show the legend
plt.legend()

# Show the plot
plt.show()

# %% [markdown]
# We have used `Random Forest` as the based algorithm for company bankruptcy prediction task because it could solve problems with imbalanced class, feature selection and outliers in the dataset provided. The model is combined with `SMOTE` data processing technique to achieve better result for `Recall` and `F1-score` as evaluation metrics besides `Accuracy`. Moreover, `Feature Importance Score` is applied to increase interpretability in the feature selection stage of the model.
# 
# There could be some possbile limitations for `Random Forest` with `SMOTE` data processing. `SMOTE`'s computational cost can be a burden, especially for large datasets, limiting its use in real-time applications or those with limited resources. Additionally, finding the optimal parameter combination for both `SMOTE` and `Random Forest` can be challenging and time-consuming.



# %%

# **CS412 TERM PROJECT**
# **Influencer Category Classification Task:**
* **Round 1:** Logistic Regression 
* **Round 2:** SVM with GridSearchCV

## **Repository Overview**
###### .
###### ├── README.md
###### ├── ROUND 1
###### │   ├── CS412_TERM_PROJECT_ROUND_1_(Classification).ipynb
###### │   └── CS412_TERM_PROJECT_ROUND_1_(Regression).ipynb
###### ├── ROUND 2
###### │   ├── CS412_TERM_PROJECT_ROUND_2_(Classification).ipynb
###### │   └── CS412_TERM_PROJECT_ROUND_2_(Regression).ipynb
###### ├── ROUND 3
###### │   ├── CS412_TERM_PROJECT_ROUND_3_(Classification).ipynb
###### │   └── CS412_TERM_PROJECT_ROUND_3_(Regression).ipynb
###### └── desktop.ini


## **Classification Task: Initial Steps For All Rounds**

### **1. Import Dependencies**
- Ensure all required libraries and dependencies are imported for data manipulation, preprocessing, and model development.

### **2. Load Turkish Stopwords Corpus**
- Load and utilize Turkish stopwords to preprocess the textual data effectively.

### **3. Influencer Category Classification Workflow**
- **Read Data**: Load the dataset for classification.
- **Preprocess Data**: Clean and standardize the text to enhance model performance.
- **Prepare Model**: Set up the classification model with appropriate configurations.
- **Predict Test Data**: Make predictions on the test dataset.
- **Save Outputs**: Export predictions to an output file for analysis.

### **4. TF-IDF**
- **1. Preprocessing**:
  - Cleans and standardizes text using methods such as case folding.
  - Handles Turkish-specific text challenges to improve model performance.
- **2. Aggregation**:
  - Combines all posts of a user into a single textual entry for efficient analysis.
- **3. Vectorization**:
  - Transforms text into numerical vectors using the TF-IDF method, which assigns importance to words based on their frequency in the text.
- **4. Separation of Train and Test Data**:
  - Ensures the vectorizer is fitted only on the training data to avoid data leakage.

---

### **5. Split Data into Training and Test Sets**
- The dataset is split into training and test subsets for model development and evaluation.

---

## **Classification Task Round 1 Approach: Logistic Regression**

### Overview
This section outlines the step-by-step approach used in Round 1 of a classification task, leveraging Logistic Regression with L1 regularization. 
### Steps
### 1. Import Libraries
The following libraries are utilized in this approach:
* LogisticRegression from sklearn: Implements the logistic regression model.
* StratifiedKFold, GridSearchCV, and cross_val_score: Enable cross-validation and hyperparameter tuning.
* accuracy_score and classification_report: Evaluate model performance.
* numpy (np): Provides numerical operations, including creating logarithmic grids.
________________________________________
### 2. Define the Logistic Regression Model
```
logreg = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    max_iter=1000,
    multi_class='ovr'
)
```
### Key Details:
* Model Type: Logistic regression with L1 regularization to perform feature selection.
* Solver: liblinear, suitable for small datasets and supports L1 regularization.
* Multi-class Approach: One-vs-Rest (OVR).
* Maximum Iterations: Set to 1000 to ensure convergence.
________________________________________
### 3. Set Up Cross-Validation
```
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
### Cross-Validation Setup:
* Stratified K-Folds: Maintains class distribution across folds.
* Number of Splits: 5.
* Shuffle: Randomizes data order to reduce biases.
* Random State: Ensures reproducibility.
________________________________________
### 4. Hyperparameter Tuning with Grid Search
```
param_grid = {'C': np.logspace(-4, 4, 20)}
grid_search = GridSearchCV(logreg, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)
```
### Key Points:
* C Parameter: Controls regularization strength. Smaller values enforce stronger regularization.
* Search Grid: 20 logarithmically spaced values between 10−410^{-4}10−4 and 10410^4104.
* GridSearchCV:
    - Performs 5-fold cross-validation for each value of CCC.
    - Optimizes for accuracy (scoring='accuracy').
    - Uses parallel computation (n_jobs=-1).
________________________________________
### 5. Retrieve and Train with the Best C
```
best_C = grid_search.best_params_['C']
modelLogReg = LogisticRegression(penalty='l1', C=best_C, solver='liblinear', max_iter=1000, multi_class='ovr')
modelLogReg.fit(x_train, y_train)
```
* Retrieves the optimal CCC value from the grid search.
* Trains the logistic regression model on the entire training dataset.
________________________________________
### 6. Feature Selection
```
non_zero_coefficients = np.any(modelLogReg.coef_ != 0, axis=0)
selected_feature_count = np.sum(non_zero_coefficients)
```
* Identifies features with non-zero coefficients (not eliminated by regularization).
* Counts the selected features.
________________________________________
### 7. Adjust Regularization Based on Feature Count
```
if selected_feature_count < 2000:
    print("Selected features are less than 2000, try increasing the C value.")
elif selected_feature_count > 2000:
    print("Selected features are more than 2000, try decreasing the C value.")
else:
    print("Approximately 2000 features are selected.")
```
* Adjusts the CCC value based on the selected feature count to balance feature selection.
________________________________________
### 8. Filter Data Based on Selected Features
```
selected_features_logistic = x_train.columns[non_zero_coefficients]
x_train_selected = x_train[selected_features_logistic]
x_val_selected = x_val[selected_features_logistic]
```
* Filters the training and validation datasets to retain only the selected features.
________________________________________
### 9. Re-Train the Model
```
modelLogReg.fit(x_train_selected, y_train)
```
* Retrains the logistic regression model using the reduced feature set for improved focus and efficiency.
________________________________________
### 10. Evaluate the Model
```
train_accuracy = accuracy_score(y_train, modelLogReg.predict(x_train_selected))
cv_accuracy = cross_val_score(modelLogReg, x_train_selected, y_train, cv=5, scoring='accuracy').mean()
```
* Train Accuracy: Assesses model fit on the training data.
* Cross-Validation Accuracy: Evaluates generalization performance.
________________________________________
### 11. Validation Predictions
```
y_pred = modelLogReg.predict(x_val_selected)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred))
```
* Validation Accuracy: Measures model performance on unseen data.
* Classification Report: Provides detailed metrics like precision, recall, and F1-score.

### **Summary**
The Round 1 approach combines logistic regression with L1 regularization, hyperparameter tuning, and iterative refinement to achieve both high predictive performance and effective feature selection.
### Key Steps:
* Define the Model: Logistic regression with L1 regularization for feature selection.
* Tune Hyperparameters: Use GridSearchCV to find the optimal CCC value.
* Feature Selection: Analyze coefficients to identify relevant features.
* Filter Data: Reduce datasets to retain only selected features.
* Retrain the Model: Train the logistic regression model on the refined dataset.
* Evaluate Performance: Use training, cross-validation, and validation accuracies.
* Print Selected Features: Provide insights into important predictors.

### **Experimental Findings of Round 1:**

- Train Accuracy: 0.9343
- Cross-validation Accuracy: 0.6141
- Validation Accuracy: 0.6266

* Classification Report:

| Class                | Precision | Recall | F1-Score | Support |
|----------------------|-----------|--------|----------|---------|
| art                  | 0.43      | 0.32   | 0.36     | 38      |
| entertainment        | 0.43      | 0.40   | 0.41     | 65      |
| fashion              | 0.65      | 0.62   | 0.63     | 60      |
| food                 | 0.82      | 0.91   | 0.86     | 102     |
| gaming               | 0.00      | 0.00   | 0.00     | 3       |
| health and lifestyle | 0.52      | 0.64   | 0.58     | 100     |
| mom and children     | 0.60      | 0.40   | 0.48     | 30      |
| sports               | 0.81      | 0.57   | 0.67     | 23      |
| tech                 | 0.68      | 0.71   | 0.70     | 69      |
| travel               | 0.66      | 0.64   | 0.65     | 59      |

| Validation Accuracy           | 0.63  |
|--------------------|-------|
| Macro avg          | 0.56  | 0.52  | 0.53 |
| Weighted avg       | 0.62  | 0.63  | 0.62 |

* Confusion matrix is present in Round 1 Classification file.

#### Selected Features:
* ['abdullah', 'ad', 'adam', 'adana', 'adapazarı', 'aday', 'adlı', 'adres', 'adresini', 'advertising', ..., 'şarkıları', 'şehir', 'şehit', 'şekilde', 'şifa', 'şimdi', 'şubemiz', 'şık', 'şıklık', 'şıklığı']


## **Classification Task Round 2 Approach: Classification Pipeline with SMOTE, PCA, and SVC**

The Round 2 approach demonstrates 
a robust classification pipeline using SMOTE (Synthetic Minority Over-sampling Technique) for class imbalance, PCA (Principal Component Analysis) for dimensionality reduction, and LinearSVC (Linear Support Vector Classifier) for efficient classification. This approach also includes hyperparameter tuning using GridSearchCV and model evaluation with accuracy and classification reports.

### Steps
### 1. Importing Required Libraries
The following libraries are imported for the implementation:

* pandas and numpy for data manipulation.
* StandardScaler and LabelEncoder from sklearn.preprocessing for data preprocessing.
* PCA from sklearn.decomposition for dimensionality reduction.
* LinearSVC and Pipeline from sklearn.svm and sklearn.pipeline for model building.
* accuracy_score and classification_report from sklearn.metrics for model evaluation.
* train_test_split and GridSearchCV from sklearn.model_selection for splitting data and hyperparameter tuning.
* SMOTE from imblearn.over_sampling for addressing class imbalance.
* OneVsRestClassifier from sklearn.multiclass for handling multi-class classification.
### 2. Load Dataset

* x_train and x_val: Feature matrices for training and validation data.
* y_train and y_val: Target labels for training and validation data.
### 3. Label Encoding
Label encoding is applied to convert categorical labels (y_train and y_val) into numerical values.
```
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
```
### 4. Handle Class Imbalance with SMOTE
SMOTE is used to create synthetic samples for the minority class in the training set to balance the class distribution. The balanced dataset is then assigned to x_train_balanced and y_train_balanced.
```
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train_encoded)
```
5. Define the Pipeline with PCA + LinearSVC
A pipeline is defined, consisting of:

* StandardScaler: Standardizes the feature set to have a mean of 0 and a standard deviation of 1.
* PCA: Reduces the dimensionality of the feature set, keeping only the top 100 principal components.
* LinearSVC: A linear support vector classifier wrapped in a OneVsRestClassifier to handle multi-class classification.
```
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100, random_state=42)),
    ('svm', OneVsRestClassifier(LinearSVC(random_state=42, max_iter=10000)))
])
```
### 6. Hyperparameter Grid for GridSearchCV
We define a hyperparameter grid to tune:

* The regularization parameter C of the LinearSVC.
* The number of principal components n_components in PCA.
```
param_grid = {
    'svm__estimator__C': [0.1, 1, 10],
    'pca__n_components': [50, 100, 150]
}
```
### 7. Perform Grid Search with Cross-Validation
A GridSearchCV is performed to find the best combination of hyperparameters by evaluating the model across 5-fold cross-validation.
```
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train_balanced, y_train_balanced)
```
### 8. Get the Best Estimator and Evaluate on Validation Set
After performing the grid search, the best pipeline is retrieved, and predictions are made on the validation set (x_val).
```
best_pipeline = grid_search.best_estimator_
y_val_pred = best_pipeline.predict(x_val)
y_val_pred = label_encoder.inverse_transform(y_val_pred)
```
### 9. Evaluate the Model
The performance of the model is evaluated on the validation set using:

* Accuracy: The percentage of correctly classified instances.
* Classification Report: A detailed report showing precision, recall, and F1-score for each class.
```
accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))
```
This approach builds a classification model using LinearSVC with dimensionality reduction via PCA, class balancing via SMOTE, and hyperparameter tuning with GridSearchCV. The final model is evaluated on the validation set using accuracy and a detailed classification report.

### **Experimental Findings of Round 2:**
The key findings are as follows:

### Best Parameters Found by GridSearchCV:

* pca__n_components: 150
* svm__estimator__C: 0.1
Best Cross-Validation Score:
* The best score achieved during the cross-validation process was approximately 0.7973.

* Validation Accuracy:
The accuracy on the validation set was 0.6485, meaning the model correctly predicted approximately 64.85% of the instances.

* Classification Report
The model's performance was evaluated using the classification report, which includes precision, recall, and F1-score for each class. Below are the detailed metrics for each category:

| Class                | Precision | Recall | F1-Score | Support |
|----------------------|-----------|--------|----------|---------|
| art                  | 0.37      | 0.39   | 0.38     | 38      |
| entertainment        | 0.48      | 0.45   | 0.46     | 65      |
| fashion              | 0.59      | 0.63   | 0.61     | 60      |
| food                 | 0.90      | 0.82   | 0.86     | 102     |
| gaming               | 0.67      | 0.67   | 0.67     | 3       |
| health and lifestyle | 0.70      | 0.64   | 0.67     | 100     |
| mom and children     | 0.45      | 0.50   | 0.48     | 30      |
| sports               | 0.65      | 0.57   | 0.60     | 23      |
| tech                 | 0.63      | 0.80   | 0.71     | 69      |
| travel               | 0.73      | 0.69   | 0.71     | 59      |

* Accuracy: 0.65 (Overall accuracy on the validation set)
* Macro Average: Precision: 0.62, Recall: 0.62, F1-Score: 0.62
* Weighted Average: Precision: 0.66, Recall: 0.65, F1-Score: 0.65
* Observations:
* The highest performing class in terms of F1-score was food (0.86), reflecting a strong ability of the model to predict this category accurately.
* Gaming had the lowest number of instances (3), which may have impacted its classification performance.
* The overall macro average and weighted average F1-scores are 0.62 and 0.65.

## **Classification Task Round 3 Approach: Logistic Regression**

The Round 1 approach was used again, therefore the implementation details remain the same.

**Experimental Findings of Round 3:**

* **Train Accuracy:** 0.9972627737226277  
* **Cross-validation Accuracy:** 0.6090263259171425  
* **Validation Accuracy:** 0.6138433515482696  

### Classification Report:

| Class                  | Precision | Recall | F1-Score | Support |
|------------------------|-----------|--------|----------|---------|
| art                    | 0.34      | 0.37   | 0.35     | 38      |
| entertainment           | 0.41      | 0.43   | 0.42     | 65      |
| fashion                 | 0.72      | 0.52   | 0.60     | 60      |
| food                    | 0.87      | 0.88   | 0.87     | 102     |
| gaming                  | 0.00      | 0.00   | 0.00     | 3       |
| health and lifestyle    | 0.56      | 0.66   | 0.61     | 100     |
| mom and children        | 0.47      | 0.30   | 0.37     | 30      |
| sports                  | 0.55      | 0.52   | 0.53     | 23      |
| tech                    | 0.65      | 0.75   | 0.70     | 69      |
| travel                  | 0.65      | 0.59   | 0.62     | 59      |

* **Accuracy** 0.61  
* **Macro avg**: 0.52  0.50  0.51  549  
* **Weighted avg**: 0.61  0.61  0.61  549

### Confusion Matrix

|                | art | entertainment | fashion | food | gaming | health and lifestyle | mom and children | sports | tech | travel |
|----------------|-----|---------------|---------|------|--------|-----------------------|------------------|--------|------|--------|
| **art**        | 14  | 8             | 1       | 2    | 0      | 7                     | 1                | 0      | 2    | 3      |
| **entertainment** | 6  | 28            | 5       | 4    | 0      | 10                    | 2                | 2      | 5    | 5      |
| **fashion**    | 6   | 5             | 31      | 3    | 0      | 9                     | 1                | 1      | 5    | 0      |
| **food**       | 1   | 4             | 1       | 90   | 0      | 2                     | 0                | 0      | 0    | 4      |
| **gaming**     | 1   | 0             | 0       | 0    | 1      | 0                     | 0                | 0      | 1    | 0      |
| **health and lifestyle** | 4 | 6      | 1       | 1    | 0      | 66                    | 7                | 3      | 6    | 6      |
| **mom and children** | 5 | 4          | 2       | 0    | 0      | 6                     | 9                | 0      | 2    | 2      |
| **sports**     | 1   | 2             | 0       | 0    | 1      | 0                     | 3                | 12     | 3    | 1      |
| **tech**       | 0   | 3             | 2       | 1    | 0      | 8                     | 0                | 3      | 52   | 0      |
| **travel**     | 3   | 9             | 0       | 2    | 0      | 5                     | 1                | 4      | 4    | 35     |


# **Like Count Prediction Regression Task:**

* **Round 1:** Neural Networks
* **Round 2:** XGBoost

## **Regression Task: Initial Steps For All Rounds**

### **1. Install Dependencies**
To run the script, ensure you have the following Python package installed:
```
!pip install xgboost==1.7.6 
```
### **2. Processing the gzipped JSON Lines Dataset**

**Import Required Libraries:**
```
import gzip
import json
import random
```
* gzip: To read compressed .gz files.
* json: To parse JSON data.
* random: For shuffling data to ensure randomness before splitting.

**Define File Path and Initialize Data Dictionaries:**
```
data_path = "training-dataset.jsonl.gz"
train_data = {"profiles": {}, "posts": {}}
test_data = {"profiles": {}, "posts": {}}
```
* data_path: Path to the gzipped dataset file.
* train_data and test_data: Dictionaries to store profiles and posts for training and testing datasets.
**Load and Parse Data from Gzipped File:**
```
data_entries = []

with gzip.open(data_path, "rt") as fh:
    for line in fh:
        sample = json.loads(line)
        data_entries.append(sample)
```
* gzip.open(data_path, "rt"): Opens the gzipped file for reading as text.
* json.loads(line): Converts each line (JSON string) into a Python dictionary and appends it to data_entries.
**Shuffle the Data for Randomness:**
```
random.shuffle(data_entries)
```
random.shuffle(data_entries): Shuffles the list data_entries in place, ensuring that the data is randomized before splitting.

**Calculate Train-Test Split Size:**
```
train_ratio = 0.8
train_size = int(len(data_entries) * train_ratio)
```
* train_ratio = 0.8: Defines 80% of the data to be used for training.
* train_size: Calculates the number of samples in the training set.
**Split Data into Training and Testing Sets:**

* Loop through data_entries: For each sample, the profile and posts are extracted.
* Split based on index (i): If i is less than train_size, the sample is added to the training set; otherwise, it goes to the testing set.
* Count posts: Tracks the number of posts in both training and testing datasets (train_post_count, test_post_count).

### **3. Columns of train_data and test_data**

* Identify and display all the unique columns (keys) present in the posts of both the training and testing datasets. 
* The columns are extracted from the posts section and printed for both datasets.

### **4. Feature Extraction**

* For each post in the dataset, the following features are extracted:
```
feature = [
    len(caption),  # Caption length
    caption.count("#"),  # Number of hashtags
    caption.count("@"),  # Number of mentions
    post.get("comments_count", 0) or 0,  # Number of comments
    1 if post.get("media_type") == "IMAGE" else 0  # Media type
]
```
**Time-based Features:**
* Extract the hour, weekday, and a weekend indicator from the post's timestamp:
```
timestamp = post.get("timestamp")
if timestamp:
    dt = datetime.fromisoformat(timestamp)
    feature.append(dt.hour)  # Hour of posting
    feature.append(dt.weekday())  # Day of the week
    feature.append(1 if dt.weekday() >= 5 else 0) 
```
**Target Extraction:**

* **Like Count:** The number of likes on the post is used as the target variable:
```
like_count = post.get("like_count", None)
if like_count is not None:
    features.append(feature)
    targets.append(int(like_count))  # Ensure target is an integer
```
**Handling Outliers:**

* Outliers in the like count are removed using the Interquartile Range (IQR) method by filtering based on the calculated lower and upper bounds:
```
q1 = np.percentile(train_targets, 25)
q3 = np.percentile(train_targets, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
filtered_indices = (train_targets >= lower_bound) & (train_targets <= upper_bound)
train_features = train_features[filtered_indices]
train_targets = train_targets[filtered_indices]
```
**Feature Scaling:**

* The hour and weekday features (time-based features) are scaled using MinMaxScaler to ensure they are in the range [0, 1]:
```
scaler = MinMaxScaler()
train_features[:, 5:7] = scaler.fit_transform(train_features[:, 5:7])  # Scale hour and weekday
test_features[:, 5:7] = scaler.transform(test_features[:, 5:7])
```

## **Regression Task Round 1 Approach: Neural Networks**

The Round 1 neural network approach aims to predict the like count of social media posts using regression. The model leverages features extracted from the data and applies a series of steps including data preprocessing, model creation, training, and evaluation using standard metrics such as R² and Mean Squared Error (MSE). Below are the key steps involved:

### Steps

### 1. Train/Validation/Test Split + Scaling:

* The data is split into training, validation, and test sets using train_test_split.
* The features are scaled using StandardScaler to ensure all values are within a comparable range.
* The target variable (like count) is log-transformed with np.log1p to stabilize variance.
```
X_train, X_val, y_train, y_val = train_test_split(train_features, train_targets, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
y_train_log = np.log1p(y_train)
```
### 2. Build the Model with Regularization:

* A neural network model is created using keras.Sequential, with two dense layers of 64 units each, ReLU activation, L2 regularization, batch normalization, and dropout to prevent overfitting.
```
def create_regression_model(input_dim):
    l2_reg = regularizers.l2(1e-3)
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='linear')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(loss='mse', optimizer=optimizer, metrics=[])
    return model
```
### 3. EarlyStopping:

* The EarlyStopping callback is used to monitor the validation loss and stop training if the loss does not improve after a certain number of epochs (patience=20).
```
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
```
### 4. Train the Model:

* The model is trained for up to 300 epochs, using a batch size of 512, with validation data to monitor performance.
```
history = nn_model.fit(X_train_scaled, y_train_log, validation_data=(X_val_scaled, y_val_log), epochs=300, batch_size=512, callbacks=[early_stop])
```
### 5. Evaluate on Validation Set:

* After training, the model’s performance is evaluated on the validation set using the R² score and the log-transformed MSE.
```
val_pred_log = nn_model.predict(X_val_scaled).ravel()
val_pred_log_clipped = np.clip(val_pred_log, 0.0, 15.0)
val_pred = np.round(np.expm1(val_pred_log_clipped)).astype(int)
val_r2 = r2_score(y_val, val_pred)
```
### 6. Evaluate on Test Set:

* The final model is tested on the test set, applying the same transformations (log transformation and clipping) as on the validation set.
```
test_pred_log = nn_model.predict(X_test_scaled).ravel()
test_pred_log_clipped = np.clip(test_pred_log, 0.0, 15.0)
test_pred = np.round(np.expm1(test_pred_log_clipped)).astype(int)
test_r2 = r2_score(test_targets, test_pred)
```
**Experimental Findings of Round 1**
```
Validation Log MSE (NN): 1.099877690017771
Validation R² (NN): 0.07106787611229204
1142/1142 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step
Test Log MSE (NN): 3.701033419929791
Test R² (NN): -0.28522922226832437

Sample Predictions (NN):
Actual: 20, Predicted: 49
Actual: 56, Predicted: 57
Actual: 21, Predicted: 24
Actual: 18, Predicted: 31
Actual: 41, Predicted: 235
Actual: 141, Predicted: 245
Actual: 285, Predicted: 26
Actual: 664, Predicted: 217
Actual: 1376, Predicted: 220
Actual: 54, Predicted: 243
```

## **Regression Task Round 2 Approach: XGBoost**


### 1. Train/Validation/Test Split

* The dataset is divided into training, validation, and test sets to train the model and evaluate its performance on unseen data.
```
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train_features,  # Feature matrix
    train_targets,   # Target values
    test_size=0.2,   # 20% of data for validation
    random_state=42  # Ensures reproducibility
)
```
### 2. Transform the Target

* To handle skewness in the like counts, the logarithmic transformation log1p (natural log of 1 + value) is applied. This transformation helps stabilize variance and improves model performance.
```
import numpy as np

y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)
y_test_log  = np.log1p(test_targets)  
```
### 3. Create the XGBoost Model
* The XGBoost regressor is initialized with hyperparameters like n_estimators, learning_rate, and max_depth. These control the number of trees, learning speed, and tree depth, respectively.
```
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,      # Number of trees
    learning_rate=0.05,     # Step size for weight updates
    max_depth=6,            # Maximum tree depth
    subsample=0.8,          # Percentage of samples per tree
    colsample_bytree=0.8,   # Fraction of features for tree construction
    random_state=42         # Ensures reproducibility
)
```
### 4. Train with Early Stopping
* The model is trained on the training set, with early stopping enabled using the validation set. Early stopping halts training if the validation performance stops improving for a specified number of rounds.
```
xgb_model.fit(
    X_train,           # Training features
    y_train_log,       # Transformed training targets
    eval_set=[(X_val, y_val_log)],  # Validation data
    early_stopping_rounds=25,       # Stop after 25 rounds of no improvement
    verbose=True       # Prints training progress
)
```
### 5. Evaluate on Validation Data
* The model's performance is evaluated on the validation set. Predictions are first converted back from the log scale using expm1. Metrics such as Log MSE and R² are calculated.
```
from sklearn.metrics import r2_score

val_pred_log = xgb_model.predict(X_val)
val_pred     = np.round(np.expm1(val_pred_log)).astype(int)

val_log_mse = np.mean((np.log1p(y_val) - np.log1p(val_pred))**2)
val_r2      = r2_score(y_val, val_pred)

print("XGBoost Validation Log MSE:", val_log_mse)
print("XGBoost Validation R²:", val_r2)
```
### 6. Evaluate on Test Data
* Similar to the validation evaluation, predictions on the test set are converted back from the log scale, and performance metrics are computed.
```
test_pred_log = xgb_model.predict(test_features)
test_pred     = np.round(np.expm1(test_pred_log)).astype(int)

test_log_mse = np.mean((y_test_log - np.log1p(test_pred))**2)
test_r2      = r2_score(test_targets, test_pred)

print("XGBoost Test Log MSE:", test_log_mse)
print("XGBoost Test R²:", test_r2)
```
** Experimental Findings of Round 2***
```
XGBoost Validation Log MSE: 0.9800216427810836
XGBoost Validation R²: 0.3519317271561272
XGBoost Test Log MSE: 3.1723084765265406
XGBoost Test R²: -0.023006178225395857

Sample Predictions (XGB):
Actual: 14, Predicted: 59
Actual: 25, Predicted: 30
Actual: 100, Predicted: 64
Actual: 28, Predicted: 24
Actual: 38, Predicted: 36
Actual: 41, Predicted: 31
Actual: 19, Predicted: 21
Actual: 107, Predicted: 84
Actual: 73, Predicted: 53
Actual: 98, Predicted: 107
```

## **Regression Task Round 3 Approach: XGBoost**

The approach for regression task did not change since Round 2, the implementation details remain the same.

**Experimental Findings of Round 3**

```
XGBoost Validation Log MSE: 0.9649677251301894
XGBoost Validation R²: 0.3650225715208254
XGBoost Test Log MSE: 2.823053486113462
XGBoost Test R²: -0.00962724833433981

Sample Predictions (XGB):
Actual: 114, Predicted: 364
Actual: 68, Predicted: 127
Actual: 122, Predicted: 45
Actual: 100, Predicted: 66
Actual: 94, Predicted: 23
Actual: 70, Predicted: 28
Actual: 144, Predicted: 20
Actual: 113, Predicted: 64
Actual: 86, Predicted: 69
Actual: 59, Predicted: 34
```

### **Team Contributions**
* Bilge Kağan Yılmaz 30895: Regression task
* Mehmet Barış Baştuğ 30617: Regression task
* Bora Başkan 27747: Classification task
* Shahd Sherif 30531: Classification task
* Nazlıcan Özen: Classification task, GitHub repository management, READme author. 

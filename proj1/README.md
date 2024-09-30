# Team 11 -- Maurice LeBlank -- develop an authorship analysis model

## Goals and the Task
In computational linguistics and stylometry, authorship attribution is a classic problem where a model is trained to determine the author of a text based on linguistic features. For this assignment, you will work with a binary classification problem where you must develop a statistical model that can attribute the authorship of a crime novel to a specific author.

You are provided with a dataset containing novels from various crime authors, including a set of novels from a target author. The task is to build a binary classifier that predicts whether a given novel (or a passage from a novel) was written by the target author.

You will be given the independent test set a day before the presentation to perform an attribution on it and report results in class.

## Steps

### Dataset Preparation
- From Project Guttenberg, collect a dataset containing novels/excerpts from your assigned author and other authors assigned to another project, labeled as either the target author or other authors.
- Preprocess the data (cleaning, tokenization, etc.) as appropriate to extract meaningful features.

**Note:** If the dataset contains long texts, consider splitting the novels into smaller chunks for better training results.

### Feature Engineering
- Extract textual features from the novels. Possible features include:
  - N-grams (unigrams, bigrams, trigrams).
  - TF-IDF (Term Frequency-Inverse Document Frequency) representations.
  - Stylometric features like sentence length, word length, punctuation frequency, or part-of-speech (POS) tags.
  - Word embeddings (optional): You may use pre-trained word embeddings (e.g., GloVe, Word2Vec) for advanced feature representations.

### Model Development
- Choose an appropriate statistical model for binary classification. Possible models include:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
  - Any other pre-transformer classifier of your choice.
- Split the data into training and testing sets (e.g., 80% training, 20% testing).
- Train your model using the training set.

### Model Evaluation
- Evaluate your model on the test set using performance metrics such as:
  - Precision
  - Recall
  - F1 Score
  - AUC-ROC Curve
- Discuss which metrics are most appropriate for this problem and why.
- Tune your model and report on results.

### Analysis and Interpretation
- Analyze feature importance and discuss which features (e.g., certain n-grams, stylistic markers) were most helpful in distinguishing the target author’s novels from those of other authors.
- Discuss potential limitations of your model (e.g., overfitting, limited generalizability to other authors or genres).
- Suggest improvements or alternative approaches.

___ 
## Optional Challenges
- **Cross-Domain Generalization:** Test the performance of your model on novels from a different genre (e.g., science fiction, mystery) and report the results.
- **Multi-Class Classification:** Extend the task to a multi-class classification problem where the model predicts the exact author from a set of authors, not just the binary target author vs. others.


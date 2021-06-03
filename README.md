# Stats-170-project

## Running a Demo

## Scripts Introduction

### EDA Folder

- **Fact Check Tool API example.ipynb**

  This file contains the codes for feeding news to the Google Fact Check Tool and getting the fact check results.

- **dataset visualization.ipynb**

  This file contains the codes for visualizing the context (words distribution and word cloud) in our two datasets.

### data_preprocessing Folder

- **factcheck_prerocessing.ipynb**

  In this file, we loaded the raw data of the Fact Check dataset from a CSV file. We extracted the useful information from its DataFeedElement column (all data in this column is stored as JSON dictionary) and formed a new dataset. We detect the language of all contexts in the text column. We only kept the English data and stored them in the local PostgreSQL server. Then, we visualized some features of the dataset to understand our data better.

- **redditcomment_prerocessing.ipynb**

  In this file, we loaded the raw data of the Reddit Comment dataset from a CSV file. We transformed the data frame into 1NF and split it into two tables (news table & comments table). We stored each table in the local PostgreSQL server separately. Then, we visualized some features of the dataset to understand our data better.
  

### Main Directory
- **setup.sql**

  The script for setting up our local PostgreSQL database and defining all tables.

- **Project.ipynb**

  A Jupyter Notebook to show a demo on predicting 0/1 labels for sample data by our models. 
  
- **config.py**

  All configurations (e.g., local file names/batch size/epochs) we will use to train and validate our models are saved in this file. 
  
 - **preprocess.py**
    
    This script contains the functions for loading data from the database, embedding words, and splitting data into three sets (classifier training/voting testing/validation). Our self-defined K-fold data splitting function is also in this script, where we split the data into 6 folds and use a parameter called k_index to control which fold will be used as the validation data.
  
- **model.py**
  
  In this script, we build training and predicting functions for all classifiers. We also make some helper functions here for training and validating models.


- **train_bert.py**

  This is the first script we need to run for training models. It loads functions from **preprocess.py** to get data-splitting sets and pickles them into local txt files under the **tempfile** folder. Then, it uses the classifier training sets (of the news data and the comments data) to train the BertForSequenceClassification model. The BFSC models will be stored as a .h5 file under the **modelfile** folder. The script will also use the BFSC models to predict labels for all three sets (classifier training/voting testing/validation) and pickles the results into local txt files under the **tempfile** folder.

- **train_clf.py**

  This is the second script we need to run for training models. It unpickles the classifier training set of the news data from local files and trains the Random Forest, Naive Bayes, and Logistic Regression models. All the model files will be saved as .joblib files under the **modelfile** folder.

- **WMVE.py**

  This is the script where we implement the weighted majority voting algorithm. It contains the functions for training votes, getting comments voting prediction (the way to do so is explained in our final report), and generating the weighted voting results. 

- **evaluation.py**

  This script contains several helper functions for printing model performance's evaluation message.

- **validation.py**

  This is the only file we need to run for evaluating models. It loads the voting testing set, validation set, the model files (except the BFSC model), and the BFSC predict labels from local files. 
  
  First of all, it uses the RF, NB, LR models to predict labels for the voting testing set. It also gets the comments voting prediction for news in this set. All these predictions and the BFSC prediction, which is loaded from local, will be combined and used to calculate the voting weights. The weights will be pickled into a txt file under the **tempfile** folder.
  
  Then, the script repeats similar steps to get all models' predictions for the validation set. These predictions will be combined and used for generating the weighted voting prediction result. 
  
  (All predictions mentioned here will be compared to the true label and print message for evaluating the models' performance.)

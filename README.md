# Stats-170-project

A weighted voting model to do binary classification on the authenticity of the news.

## Running a Demo

Running our whole project is very time-consuming, so we provide a demo with a small sample of data to show how to predict the authenticity of news with our models. To run the demo, please follow the steps:

- **Step 1**: Clone or dowload this repository. 
- **Step 2**: Make sure the following files exist:
  -  Main directory: pred_comments.csv, pred_news.csv, pred_relationship.csv
  -  modelfile folder: model_fores.joblib, model_lr.joblib, model_nb.joblib
  -  tempfile folder: bert_pred.txt, comment_pred.txt, pred_data.txt, voting_weight.txt
  
  If any of these files missing, please download from the link and unzip the file to the corresponding folder (require UCI email): https://drive.google.com/file/d/1rzDj499cWGJUJve3tCWj-jAPd1lZ_cBb/view?usp=sharing
- **Step 3**: Run **Project.ipynb**. 

※Note: we also provide a **Project.html** file which shows all the outputs of **Project.ipynb**.

## Running the Project

  Here we introduce the complete process of training models, evaluating models, and predicting new data by the program. Before running the code, all the data have been cleaned and done with topic modeling. Both the data and the topic modeling results have been stored in the PostgreSQL server. 
  
  Also, notice that we split the data into 6 folds and use a parameter called k_index in **Config.py** to control the fold used as the validation set. We use different k_index to run on multiple computers for doing cross-validation.
  
  **Training Model**
  
  - **Step 1:** Run **train_bert.py**. 
  
    Load data from the PostgreSQL server and split it into the classifier training set, voting testing set, and validation set. All three sets will be saved locally. Then, train and save the BFSC model. 
    
  - **Step 2:** Run **train_clf.py**. 
    
    Load the classifier training set from local. Train and save the Random Forest, Naive Bayes, and Logistic Regression models.

  **Evaluating Model**
  
  - Run **validation.py**.
  
    Load models, the voting testing set, and the validation set from local. Use all models to predict labels for the voting testing set and use these predictions to calculate voting weights. Then, use all models to predict labels for the validation set and compare them with the true labels. Use these predictions and the voting weights to gain the final voting result and compare them with the true labels.

  **Predicting on New Data**
  
  - **Step 1:** Divide the data into a news table and a comments table and save them as CSV files. The two tables should contain the following columns:
  
    - News Table: text, title, author, topic, perception
    
    - Comment Table: text, title, comment_author, comment_text, comment_score, comment_subreddit

  - **Step 2:** Run **prediction.py**.
  
    Load the two tables and embedded the contexts. Load models and voting weights from local. Use models to predict labels for the data. Then, use these predictions and the voting weights to gain the final voting result.

## Scripts Introduction

### EDA Folder

- **Fact Check Tool API example.ipynb**

  This file contains the codes for feeding news to the Google Fact Check Tool and getting the fact check results.

- **dataset visualization.ipynb**

  This file contains the codes for visualizing the context (words distribution and word cloud) in our two datasets.

### data_preprocessing Folder

- **factcheck_preprocessing.ipynb**

  We loaded the raw data of the Fact Check dataset from a CSV file. We extracted the useful information from its DataFeedElement column (all data in this column is stored as JSON dictionary) and formed a new dataset. We detect the language of all contexts in the text column. We only kept the English data and stored them in a CSV file.

- **factcheck_label.ipynb**

  We loaded the CSV file generated in **factcheck_preprocessing.ipynb**. We explore the labels of the Fact Check dataset and manually convert them into binary 0/1 labels. Then, we stored the data in the local PostgreSQL server and visualized some dataset features to understand our data better.

- **redditcomment_prerocessing.ipynb**

  We loaded the raw data of the Reddit Comment dataset from a CSV file. We transformed the data frame into 1NF and split it into two tables (news table & comments table). We stored each table in the local PostgreSQL server separately. Then, we visualized some features of the dataset to understand our data better.
  
- **Topic Modeling.ipynb**

  We loaded the Fact Check dataset and the Reddit Comment dataset from local PostgreSQL server and combined their news in this file. Then, we embedded these news contexts and running an LDA model to do topic modeling. We assigned the news into 15 models and did some visualization. The topic modeling result was also stored in the local PostgreSQL server.

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

  This is the script where we implement the weighted majority voting algorithm. It contains the functions for calculating voting weights, getting comments voting prediction (the way to do so is explained in our final report), and generating the weighted voting results. 

- **evaluation.py**

  This script contains several helper functions for printing model performance evaluation message.

- **validation.py**

  This is the only file we need to run for evaluating models. It loads the voting testing set, validation set, the model files (except the BFSC model), and the BFSC predict labels from local files. 
  
  First of all, it uses the RF, NB, LR models to predict labels for the voting testing set. It also gets the comments voting prediction for news in this set. All these predictions and the BFSC prediction, which is loaded from local, will be combined and used to calculate the voting weights. The weights will be pickled into a txt file under the **tempfile** folder.
  
  Then, the script repeats similar steps to get all models' predictions for the validation set. These predictions will be combined and used for generating the weighted voting prediction result. 
  
  (All predictions mentioned here will be compared to the true labels to evaluate the models' performance. The evaluation message will be printed when running this script.)

- **prediction.py**
  
  This file contains all the functions for predicting news' authenticity. Before running this file, all models and the voting weights should have been trained and save under the modelfile folder; the news should be divided into a news table and a comment table saving in the main directory. The file loaded all the models and used each of them to get a prediction. Then, it loaded the voting weights and used the weights and all predictions to get the voting results.
  
  Our demo also imported functions from this file to do classification on the sample data.

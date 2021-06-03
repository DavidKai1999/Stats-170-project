# Stats-170-project


setup.sql
The script for setting up the database and defining all tables.

factcheck_prerocessing.ipynb
Importing the raw data of Fact Check dataset and proceed with language detecting and data visualization, then store data into the database.

redditcomment_prerocessing.ipynb
Importing the raw data of Reddit Comment dataset. Changing the data frame into 1NF and doing data visualization, then storing data into the database.

preprocess.py
Importing data from the database and splitting them into classifier training, voting testing, and validation set.

model.py
Building all classifiers and their helper functions.

train_bert.py
Using the classifier training set to train BertForSequence Classification model.

train_clf.py
Using the classifier training set to train other classifiers (RF, NB, LR).

WMVE.py
Implementing the weighted majority voting algorithm.

evaluation.py
Helper functions for evaluating the models’ performance.

validation.py
Using the voting testing set to train the voting weights. Use the validation set  to evaluate the performance of all models.


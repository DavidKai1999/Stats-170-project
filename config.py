
from pathlib import Path

seed_val = 1 # random seed for training

# Cross Validation
n_split = 5 # total splits of k-folds
k_index = 1 # get a specific index in k-folds result; value should be in range(0,n_split)

# BertForSequenceClassification
MAX_LEN = 128 # input_ids max_length of news
batch_size = 8
epochs = 1
checkpoint = 3
n_jobs = 2 # num of labels

# Save Models
save_news_model = Path('.\\newsbertmodel.h5') # save news bert model
save_comment_model = Path('.\\commentbertmodel.h5') # save comment bert model

forest_news_model = 'model_forest.joblib'
nb_news_model = 'model_nb.joblib'
lr_news_model = 'model_lr.joblib'

forest_comment_model = 'model_forest_c.joblib'
nb_comment_model = 'model_nb_c.joblib'
lr_comment_model = 'model_lr_c.joblib'
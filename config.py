
from pathlib import Path
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

seed_val = 1 # random seed for training

# Cross Validation
n_split = 6 # total splits of k-folds
k_index = 4 # get a specific index in k-folds result; value should be in range(0,n_split)

# BertForSequenceClassification
MAX_LEN = 128 # input_ids max_length of news
batch_size = 16
epochs = 10
checkpoint = 3
n_jobs = 2 # num of labels

# Save Models
save_news_model = Path('.\\modelfile\\newsbertmodel.h5') # save news bert model
save_comment_model = Path('.\\modelfile\\commentbertmodel.h5') # save comment bert model

forest_news_model = '.\\modelfile\\model_forest.joblib'
nb_news_model = '.\\modelfile\\model_nb.joblib'
lr_news_model = '.\\modelfile\\model_lr.joblib'

forest_comment_model = '.\\modelfile\\model_forest_c.joblib'
nb_comment_model = '.\\modelfile\\model_nb_c.joblib'
lr_comment_model = '.\\modelfile\\model_lr_c.joblib'
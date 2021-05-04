
from pathlib import Path

seed_val = 1 # random seed for training

MAX_LEN = 512 #input_ids max_length
batch_size = 8

epochs = 2
checkpoint = 1

n_jobs = 2 # num of labels


save_news_model = Path('.\\newsbertmodel.h5') # save news bert model
save_comment_model = Path('.\\commentbertmodel.h5') # save comment bert model
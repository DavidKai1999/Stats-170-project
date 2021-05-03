from pathlib import Path

MAX_LEN = 100 #input_ids max_length
batch_size = 10
epochs = 1
checkpoint = 5

save_model = Path('./my_checkpoint/model.h5') # save model
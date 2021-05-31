
from WMVE import *
import pickle
from config import *
from evaluation import *


def bertpredict_withbatch(dataloader,model):

    # Running the model on GPU.
    #model.cuda()

    # Running on GPU if available, otherwise on CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set the seed value all over the place to make this reproducible.
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    result = []

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time() # Validation start time

    # Put the model in evaluation mode
    model.eval()

    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in dataloader:

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        result.extend(np.argmax(logits, axis=1).flatten().tolist())

        # Track the number of batches
        nb_eval_steps += 1

    return result

X_train, Y_train, X_val, Y_val, \
X_test, Y_test, test_dataloader,\
train_dataloader,val_dataloader = get_news_data()

print('Saving...')
with open(".\\tempfile\\news_data.txt", "wb") as fp:  # Pickling
    pickle.dump([X_train, Y_train, X_test, Y_test, X_val, Y_val], fp)


#bertpretrain(train_dataloader, val_dataloader,'news')

bertnews = torch.load(save_news_model)
bert_news_train = bertpredict_withbatch(train_dataloader,bertnews)
bert_news_test = bertpredict_withbatch(test_dataloader,bertnews)
bert_news_val = bertpredict_withbatch(val_dataloader,bertnews)

with open(".\\tempfile\\news_train_pred.txt", "wb") as fp:  # Pickling
    pickle.dump(bert_news_train, fp)
with open(".\\tempfile\\news_test_pred.txt", "wb") as fp:  # Pickling
    pickle.dump(bert_news_test, fp)
with open(".\\tempfile\\news_val_pred.txt", "wb") as fp:  # Pickling
    pickle.dump(bert_news_val, fp)

# train bert model, model save in 'news/comments + bertmodel.h5'
X_train_c, Y_train_c, X_val_c, Y_val_c,\
X_test_c, Y_test_c, test_dataloader_c,\
train_dataloader_c,val_dataloader_c = get_comments_data()

print('Saving...')
with open(".\\tempfile\\comments_data.txt", "wb") as fp:  # Pickling
    pickle.dump([X_train_c, Y_train_c, X_test_c, Y_test_c, X_val_c, Y_val_c], fp)

bertpretrain(train_dataloader_c, val_dataloader_c,'comment')

bertcomment = torch.load(save_comment_model)
bert_comments_train = bertpredict_withbatch(train_dataloader_c,bertcomment)
bert_comments_test = bertpredict_withbatch(test_dataloader_c,bertcomment)
bert_comments_val = bertpredict_withbatch(val_dataloader_c,bertcomment)

with open(".\\tempfile\\comments_train_pred.txt", "wb") as fp:  # Pickling
    pickle.dump(bert_comments_train, fp)
with open(".\\tempfile\\comments_test_pred.txt", "wb") as fp:  # Pickling
    pickle.dump(bert_comments_test, fp)
with open(".\\tempfile\\comments_val_pred.txt", "wb") as fp:  # Pickling
    pickle.dump(bert_comments_val, fp)

print('Done!')
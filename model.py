

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import torch


from transformers import BertForSequenceClassification, AdamW, BertConfig

from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import numpy as np
from joblib import dump, load


# parameters for training
from config import *


def bertpretrain(train_dataloader, validation_dataloader,mode,MAX_LEN=MAX_LEN):
    if mode == 'news':
        if save_news_model.exists():
            model = torch.load(save_news_model)
        else:
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=n_jobs,
                output_attentions=False,
                output_hidden_states=False,
            )

    elif mode == 'comment':
        if save_comment_model.exists():
            model = torch.load(save_comment_model)
        else:
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=n_jobs,
                output_attentions=False,
                output_hidden_states=False,
            )

        # Running the model on GPU.
    # model.cuda()

    # Running on GPU if available, otherwise on CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = AdamW(model.parameters(),
                      lr = 2e-5,
                      eps = 1e-8
                      )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # Set the seed value all over the place to make this reproducible.
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    accuracy_values = []

    # For each epoch...
    for epoch in range(0, epochs):

        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        model.train() # switch to training mode

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # Pull the loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time() # Validation start time

        # Put the model in evaluation mode
        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

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

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        accuracy_values.append(eval_accuracy / nb_eval_steps)
        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

        if epoch % checkpoint == 0:
            print("Saving checkpoint...")
            if mode == 'news':
                torch.save(model,save_news_model)
            elif mode == 'comment':
                torch.save(model,save_comment_model)

    print("Saving model...")
    if mode == 'news':
        torch.save(model, save_news_model)
    elif mode == 'comment':
        torch.save(model, save_comment_model)

    plot_acc_loss(accuracy_values,loss_values)

    print("")
    print("Training complete!")

def bertpredict(model,inputs,masks):
    bert_pred = model(inputs,
                      token_type_ids=None,
                      attention_mask=masks)[0].detach().cpu().numpy()
    bertpred_class = np.argmax(bert_pred,axis=1).flatten()
    return bertpred_class

def foresttrain(X_train, Y_train, model_name):
    '''
    Train the random forest classifier and save to local.
    :param X_train:
    :param Y_train:
    '''
    forest = RandomForestClassifier(n_estimators=10, n_jobs=3) #n_job is n of CPU cores assigned
    forest.fit(X_train, Y_train)
    dump(forest, model_name)

def forest_predict(X_val, model_name):
    '''
    comparing the return result with the Y_val
    e.g.
    from sklearn.metrics import accuracy_score
    accuracy_score(Y_val, forest_prediction)
    '''
    forest = load(model_name)
    forest_prediction = forest.predict(X_val)
    return forest_prediction

def lrtrain(X_train, Y_train, model_name):
    lr = LogisticRegression(n_jobs=5)
    lr.fit(X_train, Y_train)
    dump(lr,model_name)

def lr_predict(X_val, model_name):
    lr= load(model_name)
    lr_prediction = lr.predict(X_val)
    return lr_prediction

def nbtrain(X_train, Y_train, model_name):
    '''
    Train the Naive Bayes classifer and save to local.
    :param X_train:
    :param Y_train:
    :return:
    '''
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    dump(nb, model_name)

def nb_predict(X_val, model_name):
    '''comparing the return result with the Y_val'''
    nb = load(model_name)
    nb_prediction = nb.predict(X_val)
    return nb_prediction


# ========================================
#             Helper Functions
# ========================================

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def plot_acc_loss(acc,loss):
    x1 = range(0, epochs)
    x2 = range(0, epochs)
    y1 = acc
    y2 = loss
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()

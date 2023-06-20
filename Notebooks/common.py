
# Functions and Classes for working with text classification

import json
import random
import time
import datetime
import re
import os

from nltk.tokenize import sent_tokenize

import torch
from tqdm import tqdm 
from torchmetrics import F1Score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn


# next sentence prediction dataset, where 50% are negative samples
def make_nsp_dataset(df, tokenizer, max_len):
    '''
    input_ the text samples column of a Pandas dataframe object
    
    '''
    sentences = [] # the sentences in lists, according to original order
    bag = [] # for randomly drawing negative samples
    for a in df:
        b = sent_tokenize(a)
        sentences.append(b)
        bag.extend(b)
    bag_size = len(bag)

    
    sentence_a = []
    sentence_b = []
    label = []

    #this is modified only slightly from the source above.
    for a in sentences:
        num_sentences = len(a)
        if num_sentences > 1:
            start = random.randint(0, num_sentences-2)
            # 50/50 whether is IsNextSentence or NotNextSentence
            if random.random() >= 0.5:
                # this is IsNextSentence
                sentence_a.append(a[start])
                sentence_b.append(a[start+1])
                label.append(0)
            else:
                index = random.randint(0, bag_size-1)
                # this is NotNextSentence
                sentence_a.append(a[start])
                sentence_b.append(bag[index])
                label.append(1)
    
    nsp_inputs = tokenizer(sentence_a, 
          sentence_b, 
          return_tensors='pt',
          max_length=max_len, 
          truncation=True, 
          padding='max_length')
    
    nsp_inputs['next_sentence_label'] = torch.LongTensor([label]).T
    nsp_inputs['labels'] = nsp_inputs.input_ids.detach().clone()
    
    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(nsp_inputs.input_ids.shape)
    # create mask array
    mask_arr = (rand < 0.15) * (nsp_inputs.input_ids != 101) * \
            (nsp_inputs.input_ids != 102) * (nsp_inputs.input_ids != 0)

    selection = []
    for i in range(nsp_inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
        
    for i in range(nsp_inputs.input_ids.shape[0]):
        nsp_inputs.input_ids[i, selection[i]] = 103
    
    return nsp_inputs

# for classical BERT pretraining 
class PretrainDataset(torch.utils.data.Dataset):
    '''
    initialize with nsp_inputs
    '''
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
    

# next sentence prediction and masked language model pretraining loop for BERT
def nsp_mlm_pretrain(model, dataloader, device, optimizer, n_epochs = 2):
    losses = []
    for epoch in range(n_epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            next_sentence_label=next_sentence_label,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optimizer.step()
        # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            losses.append((epoch, loss.item()))
    return losses



def train_test_val_split(df, train_split = 0.82, val_split = 0.08, test_split = 0.10, plot=True):
    total = train_split + val_split + test_split
    assert total <= 1.0 and total >= 0.999, "Split values should add to 1.0"
    
    data_size = df.shape[0]
    split = int(np.floor((val_split+test_split)*data_size))

    test_val_idxs = np.random.randint(0, data_size, split)
    train_idxs = [a for a in range(data_size) if a not in test_val_idxs]
    test_idxs = test_val_idxs[:int(np.floor(len(test_val_idxs)/2))]
    val_idxs = test_val_idxs[int(np.floor(len(test_val_idxs)/2)):]

    ds_train = df.iloc[train_idxs]
    ds_test = df.iloc[test_idxs]
    ds_val = df.iloc[val_idxs]

    x = ds_train['Target'].value_counts()
    y = ds_val['Target'].value_counts()
    z = ds_test['Target'].value_counts()
    
    if plot:
        fig, axs = plt.subplots(3)
        fig.suptitle('Categorical Distribution of Data')

        axs[0].bar(x.axes[0], height=x, color='violet')
        axs[1].bar(y.axes[0], height=y, color='limegreen')
        axs[2].bar(z.axes[0], height=z, color='lightblue')
        fig.legend(labels=['train', 'val', 'test'])
        
    return ds_train, ds_val, ds_test

class ClassificationDataset(torch.utils.data.Dataset):
    '''
    wrapper for the train-val-test data for compatibility
    with BERT

    Pandas df must have 'Target' and 'Text' as column names

    Tokenizer is from the BERT family
    '''
    def __init__(self, df, tokenizer, target_idx, max_seq_len):

        #self.tokenizer = tokenizer
        #self.target_idx = target_idx
        #self.max_seq_len = max_seq_len

        self.labels = [target_idx[a] for a in df['Target']]
        self.texts = [tokenizer.encode_plus(text,
                                add_special_tokens=True,
                                max_length=max_seq_len,
                                truncation=True,
                                padding='max_length',
                                return_attention_mask = True,
                                return_token_type_ids = True,
                                return_tensors='pt') 
                      for text in df.loc[:,'Text']]

    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        
        return batch_texts, batch_y


# fine-tuning BERT classifier

# helper functions (these are not mine, they need citation)
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

#validation loop function, adapted from same source as above

def BERT_fine_tune_validation(model, dataloader, device, metric = []): 

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()
    y_true = torch.tensor([]) # target values
    y_pred = torch.tensor([]) # argmax of each sample
    preds = torch.tensor([]) # logits of all samples
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in dataloader:

        b_input_ids = batch[0]['input_ids'].squeeze(1).to(device)
        b_input_mask = batch[0]['attention_mask'].squeeze(1).to(device)
        b_token_type_ids = batch[0]['token_type_ids'].squeeze(1).to(device)
        b_labels = batch[1].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, 
                                   token_type_ids = b_token_type_ids, 
                                   attention_mask = b_input_mask, 
                                   labels = b_labels.long())
        total_eval_loss += outputs.loss

        logits = outputs.logits.detach().cpu()
        label_ids = b_labels.to('cpu').numpy()

        
        pred = logits.argmax(dim=1).unsqueeze(0)
        preds = torch.cat((preds, logits), 0)

        y_pred = torch.cat((y_pred, pred), -1)
        #label_ids = label_ids.cat(b_labels.to('cpu').numpy())
        y_true = torch.cat((y_true, b_labels.detach().cpu()), -1)
        #label_ids = torch.cat(label_ids, b_labels)
        total_eval_accuracy += flat_accuracy(logits.numpy(), label_ids)

    stats = {}

    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(dataloader)
    
    y_pred = y_pred.int().flatten()
    y_true = y_true.int().flatten()
    val_f1 = metric(y_pred, y_true)

    validation_time = format_time(time.time() - t0)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation F1 Score: {0:.3f}".format(val_f1))
    print("  Validation took: {:}".format(validation_time))
    
    stats['avg_val_loss'] = avg_val_loss.item()
    stats['val_time'] = validation_time
    stats['val_f1'] = val_f1
    stats['avg_flat_acc'] = avg_val_accuracy
    stats['logits'] = preds
    stats['y_pred'] = y_pred
    stats['y_true'] = y_true

    return avg_val_loss, stats


def BERT_fine_tune_train(classification_model, 
                    train_dataloader, 
                    validation_dataloader,
                    device, 
                    metric,
                    optimizer, 
                    scheduler,
                    epochs = 2,
                    lr = 1e-5,
                    n_warmup = 0,
                    save_path = None
                    ):
    
    stats = []
    total_t0 = time.time()

    f1 = metric
    
    for j in range(0, epochs):
        train_stats = {}
        print("")
        print('======== Epoch {:} / {:} ========'.format(j + 1, epochs))
        print('Training...')
        t0 = time.time()
        
        # Reset loss
        total_train_loss = 0
        step_loss = 0

        classification_model.to(device)
        classification_model.train()

        for step, batch in enumerate(train_dataloader):
            
            if step%100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                ls = step_loss / 100
                print('  Batch {:>5,} / {:>5,}  -  Avg Batch Loss: {:.4f}  -  Elapsed: {:}'.format(step, len(train_dataloader), ls, elapsed))
                step_loss = 0

            b_input_ids = batch[0]['input_ids'].squeeze(1).to(device)
            b_input_mask = batch[0]['attention_mask'].squeeze(1).to(device)
            b_token_type_ids = batch[0]['token_type_ids'].squeeze(1).to(device)
            b_labels = batch[1].to(device)

            classification_model.zero_grad()

            outputs = classification_model(b_input_ids, 
                                 token_type_ids=b_token_type_ids, 
                                 attention_mask = b_input_mask, 
                                 labels = b_labels.long())

            step_loss += outputs.loss.item()
            total_train_loss += outputs.loss.item()

            outputs.loss.backward()

            torch.nn.utils.clip_grad_norm_(classification_model.parameters(), 1.0)

            optimizer.step()




        avg_train_loss = total_train_loss / len(train_dataloader)
        train_stats['avg_train_loss'] = avg_train_loss
        training_time = format_time(time.time() - t0)
        train_stats['training_time'] = training_time

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        val_loss, val_stats = BERT_fine_tune_validation(classification_model, 
                                          validation_dataloader,
                                        device,
                                         metric)
        scheduler.step(val_loss)

        # save checkpoint if improvement
        if j == 0:
            best_loss = val_loss
        else:
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({'epoch': j,
                            'model_state_dict': classification_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_loss},
                            save_path)

        stats.append(
            {
                'epoch': j + 1,
                'val_stats': val_stats,
                'train_stats': train_stats
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
    return stats


def plot_confusion_matrix (y_true, y_pred, idx_target):

    targets = [idx_target[a] for a in range(len(idx_target.keys()))]
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, 
                            index = targets,
                            columns = targets)
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)


def get_weights(df, target_idx, n_classes, beta=0.999, to_tensor = True):
    '''
    get weights for weighted cross-entropy loss.
    '''
    samples_per_class = df['Target'].value_counts()
    n_classes = len(target_idx.keys())
    effective_num = 1.0 - np.power(beta, samples_per_class)
    class_weights = (1.0 - beta) / effective_num
    class_weights = class_weights / np.sum(class_weights)*n_classes
    # IMPORTANT: connect the weights to the target IDS
    weights = [class_weights[a] for a in target_idx]
    
    if to_tensor:
        weights = torch.tensor(weights).to(torch.float32)
    print(class_weights)
    
    return weights


import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
from transformers import (AutoModel,
                          AutoConfig)

class CustomModel(nn.Module):
    
    def __init__(self, num_labels, loss_fn, tokenizer, dropout = 0.10): #checkpoint, num_labels): 
        super(CustomModel,self).__init__() 
        self.num_labels = num_labels 
        #Load Model
        self.config = AutoConfig.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_config(self.config) # 
        self.model.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(dropout) 
        self.linear = nn.Linear(512, 1)
        self.layer_norm = nn.LayerNorm([768, 1])
        self.classifier = nn.Linear(768, num_labels) # load and initialize weights

        self.loss_fn = loss_fn

    def forward(self, input_ids=None, token_type_ids = None, attention_mask=None,labels=None):
        #Extract outputs from the MLM model
        outputs = self.model(input_ids=input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask)
        #Get Last hidden state
        sequence_output = self.dropout(outputs['last_hidden_state'])#[-1])
        # condense the 512 tokens into 1 dimension
        lin = self.linear(sequence_output.transpose(-2,-1)) 
        # layer norm
        normed = self.layer_norm(lin)
        # condense the features into 6 class values 
        logits = self.classifier(normed.transpose(-2,-1)).squeeze(1)
        
        # calculate losses
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.long())

        return SequenceClassifierOutput(loss=loss, 
                                        logits=logits, 
                                        hidden_states=outputs.hidden_states, 
                                        attentions=outputs.attentions)

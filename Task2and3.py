# coding: utf-8


###### Submitted for CE807 Assignment 2 by:
## Student Information
# Name: Vivan Raaj Rajalingam  Registration Number: 1704827


## This code is modified from the original implementation at https://github.com/gkchai/NamedEntityRecognition



# importing required packages
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Activation, Dense, Dropout, TimeDistributed, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras import callbacks
from keras.utils.vis_utils import plot_model
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
import itertools

# fix random seed for reproducibility
np.random.seed(31415)


################################ 
# TASK 2
################################ 

#### Loading Data & Data Preprocessing

# limit sentences length to a maximum of 64 words.
max_len = 64


# load Training Data which is the silver standard Wikiner Corpus Dataset

# initialize new lists
X, Y = [], []

# load training data
with open('wikiner.txt', 'r') as f:
    content = f.read()

# sentences are separated by \n\n and ignore last line
sentences = content.split("\n\n")[:-1]
for sentence in sentences:
    tokens = sentence.split("\n")
    # initialize new lists
    x, y = [], []
    for token in tokens:
        # initialize new tuple
        tuple = token.split(" ")
        # append tuple to list
        x.append(tuple[0])
        y.append(tuple[1])

    # ignore sentences with more than max_len words or less than 1 word
    if (len(x) > max_len) or (len(x) <= 1):
        continue

    X.append(x)
    Y.append(y)



# load evaluation Data which is gold standard Wikigold Corpus Dataset

# initialize new lists
wikigold_X, wikigold_Y = [], []

# load testing data
with open('wikigold.conll.txt', 'r') as g:
    ingrediant = g.read()

# sentences are separated by \n\n
wiki_sentences = ingrediant.split("\n\n")[:-1]
for wiki_sentence in wiki_sentences:
    wiki_tokens = wiki_sentence.split("\n")
    
    # initialize new lists
    wikigold_x, wikigold_y = [], []
    for wiki_token in wiki_tokens:
        # initialize new tuple
        tuple = wiki_token.split(" ")
        # append tuple to list
        wikigold_x.append(tuple[0])
        wikigold_y.append(tuple[1])

    # ignore sentences with more than max_len words or less than 1 word
    if (len(wikigold_x) > max_len) or (len(wikigold_x) <= 1):
        continue

    wikigold_X.append(wikigold_x)
    wikigold_Y.append(wikigold_y)


# Get counts of occurences of named entity in the training dataset 
d = Counter(x for sublist in Y for x in sublist)
print(d)


# replace named entities of labels b-loc, b-org, b-misc and b-per 
# with corresponding I labels
for x in Y:
    for n, i in enumerate(x):
        if i == 'B-LOC':
            x[n] = 'I-LOC'
        if i == 'B-MISC':
            x[n] = 'I-MISC'
        if i == 'B-ORG':
            x[n] = 'I-ORG'
        if i == 'B-PER':
            x[n] = 'I-PER'


# Get counts of occurences of named entity again after preprocessing steps
d1 = Counter(x for sublist in Y for x in sublist)
print(d1)


# Check of Wikigold dataset contains same label types of named entities
d2 = Counter(x for sublist in wikigold_Y for x in sublist)
print(d2)


# function to one hot encode the labels
def encode(arr, num_labels):
    one_hot = []
    for z in arr:
        # create a new numpy array with same shape as labels
        temp = np.zeros(num_labels, dtype=np.int32)
        temp[z] = 1
        one_hot.append(temp)
    return one_hot


# check for unique labels in the labels array
# and sorted for label 
entities = sorted(set(itertools.chain(*Y)))
print(entities)


# combine the list that contains the individual words in all the datasets
full_words_final = X + wikigold_X
print(len(full_words_final))


# only collect the unique words that appear in the datasets
full_words_final_final = set(itertools.chain(*full_words_final))
print(len(full_words_final_final))


#create dictionaries that map the words in the vocabulary to integers. 
#Then we can convert each of our reviews into integers so they can be passed into the network.

# reserve index 0 for padding/masking
idx2word = dict((i+1,v) for i,v in enumerate(full_words_final_final))
word2idx = dict((v, i+1) for i,v in enumerate(full_words_final_final))

idx2entity = dict((i+1,v) for i,v in sorted(enumerate(entities)))
entity2idx = dict((v, i+1) for i,v in sorted(enumerate(entities)))

# add 1 on top of label counts
num_entities = len(entity2idx) + 1
num_words = len(word2idx) + 1

print('Training & Testing Word Counts = {0}, Entity Count = {1}'.format(num_words, num_entities))


# Training Dataset
# index encoder
X_enc = list(map(lambda x: [word2idx[wx] for wx in x], X))
Y_enc = list(map(lambda y: [entity2idx[wy] for wy in y], Y))


# Testing
# index encoder
wikigold_X_enc = list(map(lambda wikigold_x: [word2idx[wx] for wx in wikigold_x],wikigold_X))
wikigold_Y_enc = list(map(lambda wikigold_y: [entity2idx[wy] for wy in wikigold_y], wikigold_Y))


# one-hot encoder
Y_oh_enc = list(map(lambda y: encode(y, num_labels=num_entities), Y_enc))


# testing
wikigold_Y_oh_enc = list(map(lambda wikigold_y: encode(wikigold_y, num_labels=num_entities), wikigold_Y_enc))


#As maximum review length too many steps for RNN. Let's truncate to 64 steps. 
#For reviews shorter than 64 steps, we'll pad with 0s.
# training dataset
X_all = pad_sequences(X_enc, max_len) 
Y_all = pad_sequences(Y_oh_enc, max_len)


# testing dataset
Wikigold_X_all = pad_sequences(wikigold_X_enc, max_len) 
Wikigold_y_all = pad_sequences(wikigold_Y_oh_enc, max_len) 


# construct training and test sets using sklearn's train test split function
# we will allocate 33% for testing dataset
X_train, X_valid, Y_train, Y_valid = train_test_split(X_all, Y_all, test_size=0.33, random_state=42)
print ('Training and testing shapes:', X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)


########### Training

# Creating Callbacks which is used in the Keras fit function
# ModelCheckpoints is used to save the model after every epoch
# EarlyStopping is used to stop training when the validation loss has not improved after 2 epochs

cbks = [callbacks.ModelCheckpoint(filepath='./checkpoint_model_bilstm_with_dropout.h5', monitor='val_loss', save_best_only=True),
            callbacks.EarlyStopping(monitor='val_loss', patience=2)]


# define training hyperparameters

# embedding layer size
embedding_size = 32
# num of units in LSTM cell
num_cells = 100
batch_size = 32
num_epochs = 5

# if using bi_directional LSTM layer, set to true
use_bidirectional = True


# construct the NN model
model = Sequential()

# embed into vector space of dimension embedding_size
# input value 0 is a special "padding" value that should be masked out
# initialize with random vectors
model.add(Embedding(len(word2idx)+1, embedding_size, input_length=max_len, mask_zero=True))


# add LSTM layer; return all sequences for the output
if use_bidirectional:
    model.add(Bidirectional(LSTM(num_cells, return_sequences=True)))
else:
    model.add(LSTM(num_cells, return_sequences=True))


model.add(Dropout(0.2))

# applies fully-connected operation at every timestep
model.add(TimeDistributed(Dense(len(entity2idx)+1)))
# add softmax classifer at output
model.add(Activation('softmax'))

# use categorical cross entropy loss function and adam optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy')
print (model.summary())


# train the model
model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_valid, Y_valid),callbacks=cbks)


#to visualize the training graphs
#run "tensorboard --logdir='./logs' "  from the command terminal


######## Evaluating Model Performance on Wikiner Dataset

# load the saved model
# returns a compiled model
model = load_model('checkpoint_model_bilstm_with_dropout.h5')


# visualize model architecture
plot_model(model, to_file='model_plot_bilstm_with_dropout.png', show_shapes=True, show_layer_names=True)


# run prediction on training data
Y_test_pred = model.predict_classes(X_valid)


# function to convert predicted values and actual values to appropiate format 
# to generate confusion matrix first remove the zero masked inputs and outputs
def clean(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    # flatten to single array with class labels
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr


y_g_u, y_p_u = clean(Y_valid.argmax(2),Y_test_pred)


# prints confusion matrix for each labels
print ('\nconfusion matrix:\n')
print (confusion_matrix(y_g_u, y_p_u))

# prints metrics for each labels
precision, recall, fscore, support  = precision_recall_fscore_support(y_g_u, y_p_u)
print('\nclass | precision,recall,fscore,support\n')
for tag, i in entity2idx.items():
    print('{0} | {1:1.2f}\t{2:1.2f}\t{3:1.5f}\t{4}'.format(tag, precision[i-1], recall[i-1], fscore[i-1], support[i-1]))


# prints overall f1 score on wikiner dataset
# use averaging method of 'macro' to ensure equal weightage to each label
# irregadless of how big its support are

print(f1_score(y_g_u, y_p_u, average = 'macro'))


################################ 
# TASK 3
################################ 

#### Testing on WIkigold Dataset

# to predict wikigold data:
Wikigold_test_pred = model.predict_classes(Wikigold_X_all)


wikigold_y_g_u,wikigold_y_p_u  = clean(Wikigold_y_all.argmax(2),Wikigold_test_pred)


# prints confusion matrix for each labels
print ('\nconfusion matrix:\n')
print (confusion_matrix(wikigold_y_g_u, wikigold_y_p_u))

# prints metrics for each labels
precision, recall, fscore, support  = precision_recall_fscore_support(wikigold_y_g_u, wikigold_y_p_u)
print('\nlabel | precision,recall,fscore,support')
for tag, i in entity2idx.items():
    print('{0} | {1:1.2f}\t{2:1.2f}\t{3:1.5f}\t{4}'.format(tag, precision[i-1], recall[i-1], fscore[i-1], support[i-1]))


# prints overall f1 score on wikigold dataset
# use averaging method of 'macro' to ensure equal weightage to each label
# irregadless of how big its support are

print(f1_score(wikigold_y_g_u, wikigold_y_p_u, average='macro'))


# take the 9th predicted ouput
allo = Wikigold_test_pred[12]


# convert array to list
allo = [allo]


# take the 9th actual value
ground = [Wikigold_y_all[12].argmax(1)]


# visualize output and compare to actual sentence
y_vis_g,y_vis_u = clean(ground,allo)
y_vis_u_l = [idx2entity.get(val) for val in y_vis_u]
print("\nInput sentence: {}".format(' '.join(wikigold_X[12])))
print("\nPredict entities: {}".format(' '.join(y_vis_u_l)))
print("\nCorrect entities: {}".format(' '.join(wikigold_Y[12])))
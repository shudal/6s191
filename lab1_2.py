import tensorflow as tf
import mitdeeplearning as mdl

import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm


songs = mdl.lab1.load_training_data()
example_song = songs[0]
print("\n Example song: ")
print(example_song)
# mdl.lab1.play_song(example_song)

songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))
print("Threre are", len(vocab), "unique characters in the dataset")

char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)
print('{')
for char,_ in zip(char2idx,range(20)):
        print('    {:4s}: {:3d}'.format(repr(char),char2idx[char]))
print('    ...\n}')

def vectorize_string(string):
    ans = []
    for i in range(0,len(string)):
        c = string[i]
        ans.append(char2idx[c])
    ans = np.array(ans)
    return ans

vectorized_songs = vectorize_string(songs_joined)
print(vectorized_songs)
assert isinstance(vectorized_songs,np.ndarray)

def get_batch(vectorized_songs,seq_length,batch_size):
    n = vectorized_songs.shape[0] - 1
    idx = np.random.choice(n-seq_length,batch_size)
    # print(idx)
    input_batch=[]
    output_batch=[]
    for id in idx:
        #print(id)
        input_batch.append(vectorized_songs[id:id+seq_length])
        output_batch.append(vectorized_songs[id+1:id+seq_length+1])
    x_batch =np.reshape(input_batch,[batch_size,seq_length])
    y_batch =np.reshape(output_batch,[batch_size,seq_length])
    return x_batch,y_batch

test_args = (vectorized_songs,10,2)
print(test_args)
if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_next_step(get_batch, test_args):
   print("======\n[FAIL] could not pass tests")
else:
   print("======\n[PASS] passed all tests!")

x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
    print("Step {:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
      )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    # Layer 1: Embedding layer to transform indices into dense vectors
    #   of a fixed embedding size
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

    # Layer 2: LSTM with `rnn_units` number of units.
    # TODO: Call the LSTM function defined above to add this layer.
    LSTM(rnn_units),

    # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
    #   into the vocabulary size.
    # TODO: Add the Dense layer.
    tf.keras.layers.Dense(vocab_size,activation=tf.sigmoid)
    ])

    return model

# Build a simple model with default hyperparameters. You will get the
#   chance to change these later.
model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)
model.summary()
print(len(vocab))
x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")
#print(pred)
sampled_indices = tf.random.categorical(pred[0], num_samples=1)
#print(sampled_indices)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
#print(sampled_indices)
print("Input: \n", repr("".join(idx2char[x[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))
### Defining the loss function ###

'''TODO: define the loss function to compute and return the loss between
    the true labels and predictions (logits). Set the argument from_logits=True.'''
def compute_loss(labels, logits):
  loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True) # TODO
  return loss

'''TODO: compute the loss using the true next characters from the example batch 
    and the predictions from the untrained model several cells above'''
example_batch_loss = compute_loss(y, pred) # TODO

print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())
print("done")

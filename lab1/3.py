from lab1_2_2 import *
import tensorflow as tf
import mitdeeplearning as mdl

import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm


def generate_text(model, start_string, idx2char,char2idx,generation_length=1000):
    # Evaluation step (generating ABC text using the learned RNN model)

    '''TODO: convert the start string to numbers (vectorize)'''
    input_eval = vectorize_string(start_string,char2idx)
    print(input_eval)
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        '''TODO: evaluate the inputs and generate the next character predictions'''
        predictions = model(input_eval)

        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        '''TODO: use a multinomial distribution to sample'''
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # Pass the prediction along with the previous hidden state
        #   as the next inputs to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        '''TODO: add the predicted character to the generated text!'''
        # Hint: consider what format the prediction is in vs. the output
        text_generated.append("" + idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

if __name__ == '__main__':
    songs = mdl.lab1.load_training_data()
    songs_joined = "\n\n".join(songs)
    vocab = sorted(set(songs_joined))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    print('{')
    for char,_ in zip(char2idx,range(20)):
            print('    {:4s}: {:3d}'.format(repr(char),char2idx[char]))
    print('    ...\n}')

    ### Hyperparameter setting and optimization ###

    # Optimization parameters:
    num_training_iterations = 2000  # Increase this to train longer
    batch_size = 4  # Experiment between 1 and 64
    seq_length = 100  # Experiment between 50 and 500
    learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

    # Model parameters:
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024  # Experiment between 1 and 2048

    # Checkpoint location:
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

    ### Define optimizer and training operation ###

    '''TODO: instantiate a new model for training using the `build_model`
      function and the hyperparameters created above.'''
    batch_size=1
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    model.summary()

    '''TODO: Use the model and the function defined above to generate ABC format text of length 1000!
        As you may notice, ABC files start with "X" - this may be a good start string.'''

    while True:
        generated_text = generate_text(model, "X", idx2char, char2idx,1000)  # TODO
        # generated_text = generate_text('''TODO''', start_string="X", generation_length=1000)
        print(generated_text)
        generated_songs = mdl.lab1.extract_song_snippet(generated_text)
        if len(generated_songs) > 0:
            print(generated_text)
            break
    for i, song in enumerate(generated_songs):
        # Synthesize the waveform from a song
        waveform = mdl.lab1.play_song(song)

        # If its a valid song (correct syntax), lets play it!
        if waveform:
            print("Generated song", i)
            ipythondisplay.display(waveform)

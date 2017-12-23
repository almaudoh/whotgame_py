import tensorflow as tf
import numpy as np
import card
from card import Card
import gamestate as gs


# Normalizes the contents of the file into a feature array of doubles
def normalize(file):
  y = []
  x = []
  for line in file.split('\n'):
    move, top, called, mode, hand = line.split(',', 4)
    called_range = [0.] * 5
    if called in ["star", "cross", "circle", "square", "triangle"]:
      called_range[card.shape_strings[called]] = 1.
    y.append(gs.cardspace([Card(move)]))
    x.append(gs.cardspace([Card(top)]) + called_range + [0. if mode == 'Normal' else 1.] + gs.cardspace(cards(hand)))

  return [np.array(x), np.array(y)]


# Compresses a string of card specs (e.g. Circle 10,Cross 5,Whot 30) into a list of card point positions.
def cards(cardspecs):
  cards = []
  for cardspec in cardspecs.split(','):
    cards.append(Card(cardspec))
  return cards


def get_data_batch(dataset, batch_size, batch_num):
  num_batches = len(dataset) // batch_size
  offset = (batch_num % num_batches) * batch_size
  return dataset[offset : min(offset + batch_size, len(dataset)), :]


def get_state(top, mode, hand):
  state = gs.GameState()
  state.hand = cards(hand)
  state.topcard = Card(top)
  state.mode = mode
  return state


def top_matches(probability, number):
  # Find the highest probabilities.
  highest = np.argsort(-probability, axis=1)[0][0:number]
  return [card.from_cardpoint(cardpoint) for cardpoint in highest]


# Constants defining our neural network
input_size = 246 # top card space=120, called shape space=5, mode=1, hand=120
hidden_1_size = 1000
hidden_2_size = 880
hidden_3_size = 600
hidden_4_size = 300
output_classes = 120
batch_size = 100
learning_rate = 1e-1

tf.reset_default_graph()

# Input layer
inputs = tf.placeholder(tf.float32, [None, input_size])
W1 = tf.get_variable('W1', shape=[input_size, hidden_1_size], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.sigmoid(tf.matmul(inputs, W1))
W2 = tf.get_variable("W2", shape=[hidden_1_size, hidden_2_size], initializer=tf.contrib.layers.xavier_initializer())
layer2 = tf.nn.relu(tf.matmul(layer1, W2))
W3 = tf.get_variable("W3", shape=[hidden_2_size, hidden_3_size], initializer=tf.contrib.layers.xavier_initializer())
layer3 = tf.nn.relu(tf.matmul(layer2, W3))
W4 = tf.get_variable("W4", shape=[hidden_3_size, output_classes], initializer=tf.contrib.layers.xavier_initializer())
output = tf.nn.softmax(tf.matmul(layer3, W4))

labels = tf.placeholder(tf.float32, [None, output_classes])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Load the training data from the file and normalize it.
file = open("saved_moves.txt", "read")
dataset = normalize(file.read())
epochs = 10000

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# testing wth a state
state = get_state("Square 10", "Normal", "Cross 1,Square 3,Triangle 4,Star 8")
probability = sess.run(output, feed_dict={inputs: state.normalize()})
matches = top_matches(probability, 5)
print matches[0], matches[1], matches[2], matches[3], matches[4]

for epoch in range(epochs):
  batch_x = get_data_batch(dataset[0], 100, epoch)
  batch_y = get_data_batch(dataset[1], 100, epoch)
  _, current_loss, prediction = sess.run([train_step, loss, output], feed_dict={inputs: batch_x, labels: batch_y})
  if epoch % 100 == 0:
    print "Epoch: {}; loss: {}".format(epoch, current_loss)


matches = top_matches(sess.run(output, feed_dict={inputs: state.normalize()}), 5)
print matches[0], matches[1], matches[2], matches[3], matches[4]


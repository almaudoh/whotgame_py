import numpy as np
import card


# Creates a new data set from saved game data.
class GameDataSet():
    def __init__(self, filename):
        self.filename = filename
        x_data, y_data = [], []
        cnt = 0
        with open(filename, "r") as file:
            for row in file.read().split("\n"):
                x, y = normalize(row)
                x_data.append(x)
                y_data.append(y)
                cnt += 1
                if cnt % 500 == 0:
                    print('Read {0} documents from {1}'.format(cnt, filename))
            print('Read {0} documents from {1}'.format(cnt, filename))
            self.x_data = np.array(x_data)
            self.y_data = np.array(y_data)


# Normalizes the contents of the file into a flattened-out 5 x 20 feature matrix.
def normalize(row):
    grid = np.zeros([5, 20], np.float32)

    # Get the data points and convert.
    move, top, called, mode, hand = row.split(',', 4)

    # Transform the top card using a binary coded representation of the card label as the columns
    # and the card shape to specify the row e.g:
    # - Circle 12 [shape=2, label=12 (1100 in binary)] will have [0, 0, 1, 1, ...] written in the
    #   first 4 columns of the 3rd row of the grid.
    # - Cross 7 [shape=1, label=7 (0111 in binary)] will have [1, 1, 1, 0, ...] written in the
    #   first 4 columns of the 2nd row of the grid...etc.
    top = card.Card(top)
    count = 0
    if top.shape != 5:
        while count < 4:
            grid[top.shape, count] = (top.label >> count) & 1
            count += 1
    elif called in card.shapes:
            # If the card played is whot, then we just use the called card as the top card, while we
            # use the special number 15 (binary 1111) to indicate Whot was played.
            grid[card.shape_strings[called], 0:3] = 1.

    # Transform the market mode.
    pos = {'Normal': 0, 'PickTwo': 1, 'General': 2}[mode]
    grid[pos, 4] = 1.

    # Transform the cards in hand.
    num_whots = 0
    for items in cards(hand):
        if items.shape == 5:     # WHOT
            grid[num_whots, 19] = 1.
            num_whots += 1
        else:
            grid[items.shape, 4 + items.label] = 1.

    # Transform the move to a one-hot vector.
    # 0 = market, 1 - 70 = cards, 71 = WHOT.
    move, y = card.Card(move), [0.] * 72
    y[move.cardpoint] = 1.

    return [np.reshape(grid, [-1]), np.array(y)]


# Compresses a string of card specs (e.g. Circle 10,Cross 5,Whot 30) into a list of card point positions.
def cards(cardspecs):
    cards = []
    for cardspec in cardspecs.split(','):
        cards.append(card.Card(cardspec))
    return cards


# Iterator class for reading through the data
class DataIterator():
    def __init__(self, data, target, should_shuffle=True):
        self.data = data
        self.target = target
        self.size = len(self.data)
        self.epochs = 0
        self.cursor = 0
        self.should_shuffle = should_shuffle

    def shuffle(self):
        shuffle_indices = np.random.permutation(np.arange(self.size))
        self.data = self.data[shuffle_indices]
        self.target = self.target[shuffle_indices]
        self.cursor = 0

    def next_batch(self, n):
        n = min(n, self.size)  # In case the batch size is bigger than the data
        if self.cursor + n > self.size:
            self.epochs += 1
            if self.should_shuffle:
                self.shuffle()
            else:
                n = (self.size - self.cursor)

        batch_x, batch_y = (self.data[self.cursor:self.cursor + n], self.target[self.cursor:self.cursor + n])
        self.cursor += n
        return batch_x, batch_y

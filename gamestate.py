import numpy as np
import card


def cardspace(cards):
    cardspace = [0.] * 120
    numWhots = 0
    for card in cards:
        if card.shape == 5:  # WHOT is repeated, we should capture all.
            numWhots += 1
            cardspace[20 * numWhots - 1] = 1.
        elif card.cardpoint:
            cardspace[card.cardpoint] = 1.
    return cardspace


class GameState(object):

    def __init__(self, environment=None, player='agent'):
        if environment is None:
            self.top_card = None
            self.called_card = None
            self.player_count = 0
            self.hand = []
            self.current_player = None
            self.market_mode = 'Normal'
            self.general_market_player = None

        else:
            self.top_card = environment.exposed[-1]
            self.called_card = environment.called_card
            self.player_count = len(environment.opponents) + 1
            self.hand = environment.dealed[player]
            self.current_player = environment.current_player
            self.market_mode = environment.market_mode
            self.general_market_player = environment.general_market_player

    def __str__(self):
        return "Top: {}; Called: {}; Current player: {}; Mode: {}; Hand: {}"\
            .format(card.from_cardpoint(self.top_card), 'None' if self.called_card is None else card.shapes[self.called_card],
                    self.current_player, self.market_mode, [str(card.from_cardpoint(item)) for item in self.hand])

    # Normalizes the game state into a vector input
    def normalize(self):
        grid = np.zeros([5, 20], np.float32)

        # Transform the top card using a binary coded representation of the card label as the columns
        # and the card shape to specify the row e.g:
        # - Circle 12 [shape=2, label=12 (1100 in binary)] will have [0, 0, 1, 1, ...] written in the
        #   first 4 columns of the 3rd row of the grid.
        # - Cross 7 [shape=1, label=7 (0111 in binary)] will have [1, 1, 1, 0, ...] written in the
        #   first 4 columns of the 2nd row of the grid...etc.
        top = card.from_cardpoint(self.top_card)
        count = 0
        if top.shape != 5:
            while count < 4:
                grid[top.shape, count] = (top.label >> count) & 1
                count += 1
        elif self.called_card is not None:
            # If the card played is whot, then we just use the called card as the top card, while we
            # use the special number 15 (binary 1111) to indicate Whot was played.
            grid[self.called_card, 0:3] = 1.

        # Transform the market mode.
        pos = {'Normal': 0, 'PickTwo': 1, 'General': 2}[self.market_mode]
        grid[pos, 4] = 1.

        # Transform the cards in hand.
        num_whots = 0
        for item in self.hand:
            item_card = card.from_cardpoint(item)
            if item_card.shape == 5:  # WHOT
                grid[num_whots, 19] = 1.
                num_whots += 1
            else:
                grid[item_card.shape, 4 + item_card.label] = 1.

        return np.reshape(grid, [-1, 100])


class GameStates(list):
    def normalize(self):
        states = [s.normalize() for s in self]
        return np.reshape(states, [-1, 100])

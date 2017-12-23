import random
import whotgame.card as card
import numpy as np

from whotgame.gamerule import GameRule


class GameAgent(object):
    """ Game Agent """

    def __init__(self):
        self.name = None
        self.hand = []

    def get_move(self, game_state): raise NotImplementedError

    def get_name(self):
        if self.name is None:
            self.name = 'agent-' + str(random.randint(0, 300))
        return self.name

    def deal_cards(self, hand):
        self.hand = hand

    def call_card(self, game_state):
        shapes = [x for x in self.hand if x != 71]
        if len(shapes) > 0:
            index = random.randint(0, len(shapes)-1)
            return card.from_cardpoint(shapes[index]).shape
        else:
            return random.randint(0, 4)


class RandomAgent(GameAgent):

    def __init__(self):
        super(RandomAgent, self).__init__()

    def get_move(self, game_state):
        return random.randint(0, 71)


class SimpleAgent(GameAgent):

    def __init__(self):
        super(SimpleAgent, self).__init__()
        self.game_rule = GameRule()

    def get_move(self, game_state):
        valid = self.game_rule.filter_valid_moves(self.hand, game_state)
        if len(valid) > 0:
            # Pick the first valid move
            return valid.pop()

        else:
            # MARKET
            return 0

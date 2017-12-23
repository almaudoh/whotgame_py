import random
import sys
from six import StringIO
import gym
from gym import spaces
from gym.utils import seeding
from whotgame.agents.agent import GameAgent
from whotgame.gamerule import GameRule
from whotgame.gamestate import GameState
from whotgame import card


class WhotEnv(gym.Env):

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, card_stack, game_rule, opponents, illegal_move_mode):
        self.game_monitor = GameMonitor(game_rule, self)
        self.card_stack = card_stack

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        self.opponents = opponents
        self.action_space = spaces.Discrete(71)

        # Initialize the environment game state
        self.initialize_state()

    def initialize_state(self):
        self.last_action = ''
        self.rewards = {agent.get_name(): 0 for agent in self.opponents}
        self.rewards['agent'] = 0

        # Card stacks
        self.exposed = []
        self.covered = []
        # Cards held by players
        self.dealed = {}

        # Game state
        self.called_card = None
        self.current_player = None
        self.market_mode = 'Normal'   # Normal = 0, PickTwo = 1, General = 2
        self.pick_two_count = 0
        self.general_market_player = None
        self.is_suspension = False
        self.is_hold_on = False

        # Game over flag
        self.done = False

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        # Start new game.
        self.initialize_state()

        # Deal cards.
        self.game_monitor.deal_cards()

        # TODO: Opponents should play their turn first

        return GameState(self, 'agent')

    def _step(self, action):

        # If already terminal, then don't do anything
        if self.done:
            return GameState(self, 'agent'), self.rewards['agent'], True, {'environment': self}

        # If resigned, then we're done
        # if action == _resign_action(self.board_size):
        #     self.done = True
        #     return self.state.board.encode(), -1., True, {'state': self.state}

        # Play
        agent = GameAgent()
        agent.name, agent.hand = 'agent', self.dealed['agent']
        self.game_monitor.play(action, agent)
        state = GameState(self, 'agent')
        # If agent has won or hold-on return immediately
        if self.done or self.is_hold_on:
            return state, self.rewards['agent'], self.done, {'environment': self}

        # Play opponents
        self.play_opponents()
        while self.is_suspension:
            self.is_suspension = False
            self.play_opponents()

        return GameState(self, 'agent'), self.rewards['agent'], self.done, {'environment': self}
        # try:
        #     self.game_monitor.play(action, self.current_player)
        # except Exception:
        # if self.illegal_move_mode == 'raise':
        #         six.reraise(*sys.exc_info())
        #     elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                # self.done = True
                # return self.state.board.encode(), -1., True, {'state': self.state}
            # else:
            #     raise error.Error('Unsupported illegal move action: {}'.format(self.illegal_move_mode))

    def _render(self, mode='human', close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(self.last_action)
        self.last_action = ''
        return outfile

    def play_opponents(self):
        # All opponents play in turn
        for opponent in self.opponents:
            if self.is_suspension:
                # Suspension means the next opponent should be skipped
                self.is_suspension = False
                continue

            self._play_opponent_move(opponent)
            # Hold-on means the current player should continue to play
            while self.is_hold_on and not self.done:
                self.is_hold_on = False
                self._play_opponent_move(opponent)

            # Check if any of the opponents has won
            if self.done:
                return GameState(self, 'agent'), -1., True, {'environment': self}

    def _play_opponent_move(self, opponent):
        self.current_player = opponent.get_name()
        opponent_move = opponent.get_move(GameState(self, opponent.get_name()))
        self.game_monitor.play(opponent_move, opponent)
        opponent.hand = self.dealed[opponent.get_name()]


class GameMonitor(object):

    def __init__(self, game_rule, environment):
        assert isinstance(game_rule, GameRule)
        self.game_rule = game_rule
        self.environment = environment

    # share the cards to the players
    def deal_cards(self):
        environment = self.environment
        # Make a copy of the card stack supplied and shuffle it
        environment.covered = environment.card_stack[:]
        random.shuffle(environment.covered)

        # Number of cards each player is to get (random b/w 3 and 9 inclusive)
        max_cards = len(environment.covered) // ((len(environment.opponents) + 1) * 2)
        num_cards = random.randint(3, 6)
        for player in environment.opponents:
            cards = []
            # Remove random cards and give to this user.
            for i in range(num_cards):
                picked = environment.covered[i * len(environment.opponents)]
                cards.append(picked)
                environment.covered.remove(picked)

            player.deal_cards(cards)
            environment.dealed[player.get_name()] = cards

        cards = []
        # Remove random cards and give to agent.
        for i in range(num_cards):
            picked = environment.covered[i * len(environment.opponents)]
            cards.append(picked)
            environment.covered.remove(picked)

        environment.dealed['agent'] = cards

        # Place the first card that will begin the game and remove it from the reserve.
        top = environment.covered[-1]
        environment.exposed.append(top)
        environment.covered.remove(top)

        # Ensure whot 20 is not on top
        while top == 71:
            top = environment.covered[-1]
            environment.exposed.append(top)
            environment.covered.remove(top)

    def play(self, move, player):
        environment = self.environment
        environment.current_player = player.get_name()
        environment.last_action += '{} played {}; {}\n'.format(player.get_name(), card.from_cardpoint(move),
                                                               GameState(environment, 'agent'))

        if self.game_rule.is_valid_move(move, GameState(environment, player.get_name())) \
            and (move not in environment.exposed or move == 71) \
            and (move in environment.dealed[player.get_name()] or move == 0):
            # Once a move is played, then hold-on and suspension are cancelled.
            environment.is_hold_on = False
            environment.is_suspension = False

            # Update environment based on the move
            if move == 0:   # MARKET
                # Number to pick can be 1 (normal market), 2 (pick-two or general), 4 or more (extended pick-two)
                to_pick = max(environment.pick_two_count * 2, 1)

                # Reload cards if the covered stack is finishing
                if len(environment.covered) < to_pick:
                    self.reload_covered()
                    if len(environment.covered) < to_pick:
                        # Market is finished, game cannot continue, everyone loses (stalemate)
                        if environment.illegal_move_mode == 'lose':
                            environment.done = True
                            environment.rewards = {name: -1 for name in environment.rewards.keys()}
                            environment.last_action += 'Market has run out - stalemate!\n'
                            return
                        else:
                            raise Exception('Market has run out - stalemate!')

                # Ensure the right number of cards is picked - including pick-two and general market
                for i in range(to_pick):
                    market = environment.covered[-1]
                    environment.dealed[player.get_name()].append(market)
                    environment.covered.remove(market)
                environment.pick_two_count = 0

                # Once you go market, the pick-two is neutralized
                if environment.market_mode == 'PickTwo':
                    environment.market_mode = 'Normal'

                # Reset general market status if everyone has picked
                if environment.current_player == environment.general_market_player and environment.market_mode == 'General':
                    environment.general_market_player = None
                    environment.market_mode = 'Normal'

            else:   # Regular card
                # Remove the card from the player's hand and add to the exposed stack
                environment.exposed.append(move)
                environment.dealed[player.get_name()].remove(move)
                environment.called_card = None

                # Set conditions for pick-two, general market, suspension and hold-on
                state = GameState(environment, player.get_name())
                if self.game_rule.is_pick_two(move, state):
                    environment.pick_two_count += 1
                    environment.general_market_player = None
                    environment.market_mode = 'PickTwo'

                elif self.game_rule.is_general_market(move, state):
                    environment.pick_two_count = 1
                    environment.general_market_player = player.get_name()
                    environment.market_mode = 'General'

                else:
                    environment.pick_two_count = 0
                    environment.general_market_player = None
                    environment.market_mode = 'Normal'

                    if self.game_rule.is_suspension(move, state):
                        environment.is_suspension = True

                    elif self.game_rule.is_hold_on(move, state):
                        environment.is_hold_on = True

            # Check if game has been won.
            if len(environment.dealed[player.get_name()]) <= 0:
                environment.game_won = True
                environment.game_winner = player
                environment.done = True
                environment.last_action += "Game won by {}\n".format(player.get_name())
                # All other players should have negative rewards
                environment.rewards = {name: -1 for name in environment.rewards.keys()}
                environment.rewards[player.get_name()] = 1
                return

            # Called card if whot 20 is played
            if move == 71:
                called_card = player.call_card(GameState(environment, player.get_name()))
                environment.last_action += "{} called '{}'\n".format(player.get_name(), card.shapes[called_card])

                if card.is_illegal_card(called_card, 1):
                    if environment.illegal_move_mode == 'lose':
                        environment.done = True
                        environment.rewards[player.get_name()] = -1000
                        environment.last_action += "Illegal card '{}' called\n".format(card.shapes[called_card])
                    else:
                        raise Exception("Illegal card '{}' called".format(card.shapes[called_card]))

                environment.called_card = called_card

        else:  # An illegal move played
            if environment.illegal_move_mode == 'lose':
                environment.done = True
                environment.rewards[player.get_name()] = -1000
                environment.last_action += "Illegal move played: '{}'\n".format(card.from_cardpoint(move))
            else:
                raise Exception("Illegal move played: '{}'".format(card.from_cardpoint(move)))

    # Reloads the covered set by transferring everything from the exposed set except the topmost card
    def reload_covered(self):
        environment = self.environment
        top = environment.exposed[-1]
        environment.exposed.remove(top)
        random.shuffle(environment.exposed)
        environment.covered += environment.exposed
        environment.exposed = [top]

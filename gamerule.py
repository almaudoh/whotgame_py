import card


class GameRule(object):

    def __init__(self):
        # The number that is used as the pick two number
        self.pick_two_label = 7

        # The number that is used as the general market number
        self.general_market_label = 4

        # The number that is used for suspension
        self.suspension_label = 8

        # The number that is used as hold on
        self.hold_on_label = 1

        self.go_market = "MARKET"

    def filter_valid_moves(self, hand, state):
        top_card = card.from_cardpoint(state.top_card)
        # Different playing compulsions and scenarios
        if state.market_mode == 'PickTwo':
            ret = []
            for item in hand:
                if card.from_cardpoint(item).label == self.pick_two_label:
                    ret.append(item)

            return ret
        elif state.market_mode == 'General':
            # For general market, nothing can be played
            return []
        elif top_card.shape == 5:
            # If whot 20 is played, then look at the called card and return the matches
            # Whot 20 will also match
            ret = []
            for item in hand:
                item_card = card.from_cardpoint(item)
                if item_card.shape == state.called_card or item_card.shape == 5:
                    ret.append(item)

            return ret
        else:
            # Otherwise, choose all cards that match the top card either shape or label
            ret = []
            for item in hand:
                item_card = card.from_cardpoint(item)
                if item_card.shape == top_card.shape or item_card.label == top_card.label or item_card.shape == 5:
                    ret.append(item)

            return ret

    def is_valid_move(self, move, state):
        return move == 0 or self.matches_top_card(move, state) and self.is_pick_two_counter(move, state) \
            and (state.market_mode != 'General' or state.current_player == state.general_market_player)

    def is_pick_two_counter(self, move, state):
        move_card = card.from_cardpoint(move)
        return state.market_mode != 'PickTwo' or move_card.label == self.pick_two_label

    def matches_top_card(self, move, state):
        top_card = card.from_cardpoint(state.top_card)
        move_card = card.from_cardpoint(move)
        return move_card.label == top_card.label or move_card.shape == top_card.shape \
               or move_card.shape == 5 or move_card.shape == state.called_card

    def is_pick_two(self, move, state):
        return card.from_cardpoint(move).label == self.pick_two_label

    def is_general_market(self, move, state):
        return card.from_cardpoint(move).label == self.general_market_label

    def is_suspension(self, move, state):
        return card.from_cardpoint(move).label == self.suspension_label

    def is_hold_on(self, move, state):
        return card.from_cardpoint(move).label == self.hold_on_label

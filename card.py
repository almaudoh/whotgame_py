shape_strings = {
    'star': 0,
    'cross': 1,
    'circle': 2,
    'square': 3,
    'triangle': 4,
    'whot': 5,
}

shapes = ['star', 'cross', 'circle', 'square', 'triangle', 'whot']


class Card:

    # Construct a card from a string description like "Circle 3"
    def __init__(self, cardspec=None):
        if cardspec is None:
            self.shape = 0
            self.label = 0
            self.cardpoint = None
        else:
            parts = cardspec.split(" ")
            if len(parts) > 1:
                self.shape = shape_strings[parts[0].lower()]
                self.label = int(parts[1])
                # Cardpoint is the position of this card in a 72-length vector with index 0 => MARKET,
                # index 71 => WHOT, and indices 1 - 70 being real cards.
                self.cardpoint = 71 if self.shape == 5 else self.shape * 14 + self.label
            else:
                # Assume market (unknown move)
                self.shape = -100
                self.label = -100
                # Cardpoint is the position of this card in a vectorized 20 x 6 array of cards.
                self.cardpoint = 0

    def __str__(self):
        if self.shape == -100:
            return "MARKET"
        elif self.shape == 5:
            return "whot 20"
        else:
            return "{} {}".format(shapes[self.shape], self.label)

    def __eq__(self, other):
        return self.cardpoint == other.cardpoint

    def __cmp__(self, other):
        return 1 if self.cardpoint > other.cardpoint else (-1 if self.cardpoint < other.cardpoint else 0)

    def shape_string(self):
        return shapes[self.shape]


def from_cardpoint(cardpoint):
    card = Card()
    card.cardpoint = cardpoint
    if cardpoint == 71:
        # WHOT
        card.label = 20
        card.shape = 5
    elif cardpoint in list(range(1, 71)):
        # Cards
        card.label = ((cardpoint - 1) % 14) + 1
        card.shape = (cardpoint - 1) // 14
    else:
        # market
        card.label = -100
        card.shape = -100
    return card


def get_full_cardstack(use_cardpoints=False):
    cardstack = []
    for shape in range(0, 5):
        for label in range(1, 15):
            if not is_illegal_card(shape, label):
                # add a new WhotCard object to the WhotCardSet if the shape and label are legal.
                card = Card()
                card.shape = shape
                card.label = label
                card.cardpoint = 71 if card.shape == 5 else card.shape * 14 + card.label
                cardstack.append(card.cardpoint if use_cardpoints else card)
    # add the whots themselves (ie.the jokers)
    cardstack += [71 if use_cardpoints else Card("whot 20")] * 5
    return cardstack


def is_illegal_card(shape, label):
    return shape < 0 or shape > 5  or label < 1 or (shape != 5 and label > 14) \
       or label in excluded_numbers[shape] or (shape == 5 and label != 20)

excluded_numbers = [
    [6, 9, 10, 11, 12, 13, 14], # Star
    [4, 6, 8, 9, 12],           # Cross
    [6, 9],                     # Circle
    [4, 6, 8, 9, 12],           # Square
    [6, 9],                     # Triangle
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] # Whot
]
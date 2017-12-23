from gym.envs.registration import register
from whot import WhotEnv
from whotgame.agents.agent import SimpleAgent, RandomAgent
from whotgame.gamerule import GameRule
from whotgame import card

register(
    id='WhotGame-Single-v0',
    entry_point='whotgame.envs:WhotEnv',
    kwargs={
        'card_stack': card.get_full_cardstack(use_cardpoints=True),
        'game_rule': GameRule(),
        'opponents': [SimpleAgent()],
        'illegal_move_mode': 'lose',
    },
    max_episode_steps=1000,
    reward_threshold=0.78,
)

register(
    id='WhotGame-Double-v0',
    entry_point='whotgame.envs:WhotEnv',
    kwargs={
        'card_stack': card.get_full_cardstack(use_cardpoints=True),
        'game_rule': GameRule(),
        'opponents': [SimpleAgent(), SimpleAgent()],
        'illegal_move_mode': 'lose',
    },
    max_episode_steps=1000,
    reward_threshold=0.78,
)

register(
    id='WhotGame-Random-Single-v0',
    entry_point='whotgame.envs:WhotEnv',
    kwargs={
        'card_stack': card.get_full_cardstack(use_cardpoints=True),
        'game_rule': GameRule(),
        'opponents': [RandomAgent()],
        'illegal_move_mode': 'lose',
    },
    max_episode_steps=1000,
    reward_threshold=0.78,
)

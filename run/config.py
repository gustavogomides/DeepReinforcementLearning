# GAME
GAME = 'TICTAC'

# SELF PLAY
EPISODES = 3
MCTS_SIMS = 50
MEMORY_SIZE = 30000
TURNS_UNTIL_TAU0 = 10  # turn on which it starts playing deterministically
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8

# RETRAINING
BATCH_SIZE = 256
EPOCHS = 10
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)}
]

# EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3

PLOT_HISTORY_GRAPH = True

LOG_MAIN = True
LOG_MEMORY = False
LOG_TOURNEY = True
LOG_MCTS = False
LOG_MODEL = False

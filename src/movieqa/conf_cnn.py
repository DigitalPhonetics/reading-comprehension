# hierarchical CNN training config

# model config
FILTER_SIZES = [1, 3, 5]
HIDDEN_SIZE = 150

# train config
INITIAL_LEARNING_RATE = 0.001
BATCH_SIZE = 30
MAX_STEPS = 50000
DROPOUT_RATE = 0.0
LOSS_FUNC = 'entropy'
OPTIMIZER = "adam"
NUM_EPOCHS = 1

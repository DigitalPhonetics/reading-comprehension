# global initial configuration for the dataset paths for the models

# preprocessing options
P_MAX_WORD_PER_SENT_COUNT = 100
Q_MAX_WORD_PER_SENT_COUNT = 50
P_MAX_SENT_COUNT = 101

# model config
EMBEDDING_SIZE = 300
MODEL_NAME = 'model'

# path settings
DATA_PATH = "data"
RECORD_DIR = "records_new"
TRAIN_RECORD_PATH = RECORD_DIR + "/train"
TEST_RECORD_PATH = RECORD_DIR + "/test"
EMBEDDING_DIR = RECORD_DIR + '/embeddings_' + str(EMBEDDING_SIZE) + "d/"
OUTPUT_DIR = "outputs"
TRAIN_DIR = OUTPUT_DIR + '/train_' + MODEL_NAME + '/'

# vocab settings
PRETRAINED_EMBEDDINGS_PATH = '../../glove/glove.840B.' + str(EMBEDDING_SIZE) + 'd.txt'
VOCAB_SIZE = 27633

# eval config
MODE = 'val'
EVAL_FILE_VERSION = ''  # for original, unmodified val and test files
EVAL_FILE = DATA_PATH + '/data/qa.json'
MODE_MODEL_NAME_EVAL_FILE_VERSION = MODE + '_' + MODEL_NAME + EVAL_FILE_VERSION
EVAL_DIR = OUTPUT_DIR + '/' + MODE_MODEL_NAME_EVAL_FILE_VERSION
EVAL_RECORD_PATH = RECORD_DIR + '/' + MODE + EVAL_FILE_VERSION

PLOT_SAMPLES_NUM = 0

import sys
import os
import numpy as np
import argparse

sys.path.append('movieqa')
sys.path.append('glove')
sys.path.append('core')


def set_data_conf_values(model, mode, model_name, evaluation_file_version):
    data_conf = model.data_conf

    data_conf.MODEL_NAME = model_name
    data_conf.TRAIN_DIR = data_conf.OUTPUT_DIR + '/train_' + data_conf.MODEL_NAME + '/'

    data_conf.MODE = mode

    print("mode %s, data_conf.MODE %s" % (mode, data_conf.MODE))

    if evaluation_file_version:
        data_conf.EVAL_FILE_VERSION = "_" + evaluation_file_version
        data_conf.EVAL_MODIFIED_MODE = True

    if mode in ['val', 'test']:
        data_conf.MODE_MODEL_NAME_EVAL_FILE_VERSION = mode + '_' + model_name + data_conf.EVAL_FILE_VERSION
        data_conf.EVAL_DIR = data_conf.OUTPUT_DIR + '/' + data_conf.MODE_MODEL_NAME_EVAL_FILE_VERSION
        data_conf.EVAL_RECORD_PATH = data_conf.RECORD_DIR + '/' + mode + data_conf.EVAL_FILE_VERSION

        print("conf.MODE_MODEL_NAME_EVAL_FILE_VERSION", data_conf.MODE_MODEL_NAME_EVAL_FILE_VERSION)

        # use "qa.json" for validation + test, unless another eval_file_version is specified
        if mode == 'val' and data_conf.EVAL_FILE_VERSION:
            data_conf.EVAL_FILE = data_conf.DATA_PATH + '/data/qa_%s%s.json' % (mode, data_conf.EVAL_FILE_VERSION)

        if not os.path.exists(data_conf.EVAL_RECORD_PATH):
            print("Evaluating MovieQA in modified mode...")
            print("The records for the evaluation data %s are created and will be stored at %s:"
                  % (data_conf.EVAL_FILE_VERSION, data_conf.EVAL_RECORD_PATH))
            os.makedirs(data_conf.EVAL_RECORD_PATH)
            import movieqa.preprocess as pp
            new_vocab_size = pp.create_validation_dataset(data_conf.MODE)
            if new_vocab_size and new_vocab_size > 0:
                data_conf.VOCAB_SIZE = new_vocab_size

                model.load_embeddings()


def main(mode, model_type, model_name, dropout, learning_rate, loss_function, batch_size, evaluation_file_version):
    os.chdir("movieqa")

    model = None
    if model_type == "lstm":
        from movieqa import run_lstm as model
    elif model_type == "cnn":
        from movieqa import run_cnn as model
    elif model_type == "word-level-cnn":
        from movieqa import run_cnn_word_level as model

    set_data_conf_values(model, mode, model_name, evaluation_file_version)

    model_conf = model.model_conf

    if dropout:
        model_conf.DROPOUT_RATE = dropout
    if learning_rate:
        model_conf.INITIAL_LEARNING_RATE = learning_rate
    if loss_function:
        model_conf.LOSS_FUNC = loss_function
    if batch_size:
        model_conf.BATCH_SIZE = batch_size

    if mode == "train":
        model.train_model()
    else:
        model.eval_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["train", "val", "test"], help="Task to perform,"
                                                                       "val tests on validation set, test on test set")
    parser.add_argument("model_type", choices=["lstm", "cnn", "word-level-cnn"])
    parser.add_argument("model_name", help="Name of model to be saved after training or loaded for evaluation")
    parser.add_argument("-dropout", default=0.0, help="Dropout on the input embeddings")
    parser.add_argument("-learning_rate", default=0.001, help="Learning rate for Adam optimizer")
    parser.add_argument("-loss_function", choices=["_entropy_", "_hinge_"],
                        help="Type of loss function to compute error,"
                             "either cross entropy or hinge loss")
    parser.add_argument("-batch_size", default=30)
    parser.add_argument("-eval_file_version",
                        help="Model is evaluated on data/data/qa_MODE_{EVAL_FILE_VERSION}.json, empty default"
                             "evaluates on original val or test file")
    parser.add_argument("-gpuid", default='-1', help="Id of GPU to run the model on (default: run only on CPU)")

    args = parser.parse_args()
    np.set_printoptions(linewidth=100000)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    print("Run on GPU %s" % str(os.environ['CUDA_VISIBLE_DEVICES']))

    main(args.mode, args.model_type, args.model_name, args.dropout, args.learning_rate, args.loss_function,
         args.batch_size, args.eval_file_version)

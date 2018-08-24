import glob
import sys
import os
import numpy as np
import argparse

sys.path.append('movieqa')
sys.path.append('glove')
sys.path.append('core')


def main(mode, model_type, models_folder, dropout, learning_rate, loss_function, batch_size, attack_level,
         num_modified_words, percentage_attacked_samples):
    os.chdir("movieqa")

    if model_type == "lstm":
        from movieqa import run_lstm as model
    elif model_type == "cnn":
        from movieqa import run_cnn as model
    elif model_type == "word-level-cnn":
        from movieqa import run_cnn_word_level as model

    from movieqa import run_adversarial_white_box as adversary

    model_conf = model.model_conf

    if dropout:
        model_conf.DROPOUT_RATE = dropout
    if learning_rate:
        model_conf.INITIAL_LEARNING_RATE = learning_rate
    if loss_function:
        model_conf.LOSS_FUNC = loss_function
    if batch_size:
        model_conf.BATCH_SIZE = batch_size

    filenames = glob.glob(os.path.join("outputs", models_folder) + "/*")
    print("Running adversarial attack on models %s" % str(filenames))
    for model_name in filenames:
        set_data_conf_values(model, mode, model_name, attack_level)

        if mode == "adversarial-train":
            adversary.train_model(model_type, attack_level, num_modified_words, percentage_attacked_samples)
        else:
            adversary.eval_model(model_type, attack_level, num_modified_words, percentage_attacked_samples)


def set_data_conf_values(model, mode, model_name, attack_level):
    # split of mode from model name first
    model_name = os.path.basename(model_name).split("_")
    if len(model_name) > 1:
        model_name = "_".join(model_name[1:])
    else:
        model_name = model_name[0]
    print("Model name %s" % model_name)

    data_conf = model.data_conf

    data_conf.MODEL_NAME = model_name
    data_conf.TRAIN_DIR = data_conf.OUTPUT_DIR + '/train_' + data_conf.MODEL_NAME + '/'

    data_conf.MODE = mode

    print("mode %s, data_conf.MODE %s" % (mode, data_conf.MODE))

    if mode in ['val', 'test']:
        data_conf.MODE_MODEL_NAME_EVAL_FILE_VERSION = "%s%s_adversarial_%s-level_whitebox_%s" \
                                                      % (mode, data_conf.EVAL_FILE_VERSION, attack_level, model_name)
        data_conf.EVAL_DIR = data_conf.OUTPUT_DIR + '/' + data_conf.MODE_MODEL_NAME_EVAL_FILE_VERSION

        if not os.path.exists(data_conf.EVAL_DIR):
            os.makedirs(data_conf.EVAL_DIR)

        data_conf.EVAL_RECORD_PATH = data_conf.RECORD_DIR + '/' + mode + data_conf.EVAL_FILE_VERSION


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["adversarial-train", "val", "test"],
                        help="Task to perform; val tests on validation set, test on test set")
    parser.add_argument("model_type", choices=["lstm", "cnn", "word-level-cnn"])
    parser.add_argument("models_folder", help="Path to folder within movieqa/outputs directory that contains the"
                                              "training directories of all models which should be attacked"
                                              "or adversarially trained")
    parser.add_argument("attack_level", choices=["word", "sentence"])
    parser.add_argument("-num_modified_words", choices=[1, 2, 3, 4, 5, 10, 20, 40], default=1, type=int,
                        help="Number of top k attended words in the most relevant sentence that "
                             "are modified by the attack (only relevant for attack_level=word)")
    parser.add_argument("-percentage_attacked_samples", choices=range(0, 101), default=100, type=int,
                        help="Percentage of the instances in the dataset"
                             "that are attacked (0 = no attack) ->"
                             "(100 = all instances attacked)")
    parser.add_argument("-dropout", default=0.0, help="Dropout on the input embeddings")
    parser.add_argument("-learning_rate", default=0.001, help="Learning rate for Adam optimizer")
    parser.add_argument("-loss_function", choices=["_entropy_", "_hinge_"],
                        help="Type of loss function to compute error,"
                             "either cross entropy or hinge loss")
    parser.add_argument("-batch_size", default=30)

    parser.add_argument("-gpuid", default='-1', help="Id of GPU to run the model on (default: run only on CPU)")

    args = parser.parse_args()
    np.set_printoptions(linewidth=100000)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    print("Run on GPU %s" % str(os.environ['CUDA_VISIBLE_DEVICES']))

    main(args.mode, args.model_type, args.models_folder, args.dropout, args.learning_rate, args.loss_function,
         args.batch_size, args.attack_level, args.num_modified_words, args.percentage_attacked_samples)

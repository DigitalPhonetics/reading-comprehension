"""
Calls the sentence-level black-box adversarial attacks,
which are our reimplementation and extension of Jia and Liang (2017)'s AddAny attack
"""
import glob
import sys
import os
import numpy as np
import argparse

os.chdir("movieqa")

from movieqa.adversarial_addAny import create_addAny_examples as create_addany
from movieqa.adversarial_addAny import eval_addAny as evaluate_addany

sys.path.append('movieqa')
sys.path.append('glove')
sys.path.append('core')


def main(mode, model, attack_type, models_folder, examples_folder, instances_to_attack):
    filenames = glob.glob(os.path.join("outputs", models_folder) + "/*")
    print("Running adversarial attack on models %s" % str(filenames))

    total_acc = 0
    for f in filenames:
        f_examples_folder = os.path.join(f, examples_folder)

        if mode == "create_examples":
            print("create adversarial examples in %s for %s" % (examples_folder, f))
            create_addany.run_creation(model, attack_type, f, f_examples_folder, instances_to_attack)

        elif mode == "eval_examples":
            print("evaluate adversarial examples from %s for %s" % (examples_folder, f))
            acc = evaluate_addany.run_evaluation(model, attack_type, f, f_examples_folder, instances_to_attack)
            total_acc += acc

    print(total_acc / len(filenames))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["create_examples", "eval_examples"],
                        help="Task to perform,"
                             "create_example creates adversarial sentences,"
                             "eval_examples evaluates the model's accuracy")
    parser.add_argument("model_type", choices=["lstm", "cnn"], help="Word-level CNN model is currently not supported")
    parser.add_argument("attack_type", choices=["addC", "addQ", "addA", "addQA"],
                        help="Controls the word pool from which the adversarial sentence is created;"
                             "addC: common words from 'common_english.txt';"
                             "addQ: common words + question words;"
                             "addA: common words + wrong answer candidate words;"
                             "addQA: common words + question words + wrong answer candidate words")
    parser.add_argument("models_folder", help="Path to folder within movieqa/outputs directory that contains the"
                                              "training directories of all models which should be attacked")
    parser.add_argument("instances_to_attack", help="folder containing the preprocessed instances to be attacked"
                                                    " in .tfrecords and .pickle format (obtain them via preprocess.py)")
    parser.add_argument("-examples_folder", default="addAny_sentences",
                        help="Name of subfolders within each attacked model_folder where"
                             "adversarial sentences are stored")
    parser.add_argument("-gpuid", default='-1', help="Id of GPU to run the model on (default: run only on CPU)")

    args = parser.parse_args()
    np.set_printoptions(linewidth=100000)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    print("Run on GPU %s" % str(os.environ['CUDA_VISIBLE_DEVICES']))

    main(args.mode, args.model_type, args.attack_type, args.models_folder, args.examples_folder,
         args.instances_to_attack)

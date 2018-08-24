# compute accuracy of majority vote of all prediction files (probabilities_{MODEL_NANE} in ensemble directory

import glob
import numpy as np
import argparse
import os


def eval_ensemble(ensemble_directory, gold_labels_file, predictions_file, evaluation_file, ensemble_name):

    # read the gold labels
    with open(gold_labels_file) as f:
        content = f.readlines()
    gold_labels = [int(x.split()[1]) for x in content]

    # read the predictions of the individual models
    predictions = np.zeros((len(gold_labels), 5))
    model_prediction_files = glob.glob(ensemble_directory + "/probabilities_*")
    print(model_prediction_files)
    for model_prediction_file in model_prediction_files:
        print("evaluating model %s" % model_prediction_file)
        with open(model_prediction_file) as f:
            content = f.readlines()
        for i, line in enumerate(content[1:]):
            pred_ind = int(line.split("\t")[2])
            predictions[i][pred_ind] += 1

    # compute majority votes and evaluate against gold standard
    total_acc = 0.0
    with open(predictions_file, "w") as p, open(evaluation_file, "w") as e:
        for i, sample in enumerate(predictions):
            correct = int(gold_labels[i] == np.argmax(sample))
            print("label: " + str(gold_labels[i]) + " - " + str(sample) + " - " + str(
                gold_labels[i] == np.argmax(sample)))
            result = (ensemble_name + ":" + str(i) + " " + str(np.argmax(sample)))
            p.write(result + "\n")
            e.write("%s on %d: %d\n" % (ensemble_name, i, correct))
            if gold_labels[i] == np.argmax(sample):
                total_acc += 1.0

    print("total accuracy for " + ensemble_name + ": " + str(total_acc / len(gold_labels)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("ensemble_directory", help="Folder that contains all probabilites_{MODEL}.txt files"
                                                   " that should be incorporated into the ensemble")
    parser.add_argument("gold_labels_file",
                        help="Contains correct answers for all evaluation set questions line-by-line"
                             " in the format 'data_conf.TEST_SET:question_id correct_answer_id'"
                             "question_id and answer_id are zero-indexed")
    parser.add_argument("-ensemble_name", default="ensemble on validation set", help="Optional name for the ensemble,"
                                                                                     "for submitting test results"
                                                                                     "to the MovieQA sever use"
                                                                                     "'test'")

    args = parser.parse_args()

    ensemble_directory = args.ensemble_directory
    gold_labels_file = args.gold_labels_file

    # majority-vote ensemble predictions for each question
    predictions_file = os.path.join(ensemble_directory, "predictions.txt")

    # for each question 1 if ensemble predicts correct answer, else 0
    evaluation_file = os.path.join(ensemble_directory, "evaluation.txt")

    ensemble_name = args.ensemble_name

    eval_ensemble(ensemble_directory, gold_labels_file, predictions_file, evaluation_file, ensemble_name)

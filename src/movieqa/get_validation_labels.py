import sys
from modify_movieqa import read_qa_json


def get_validation_labels(qa_file):
    qas = read_qa_json(qa_file, split='val')
    labels = ["%s %d" % (qa.qid, qa.correct_index) for qa in qas]
    with open('data/data/labels_val.txt', "w") as of:
        of.write("\n".join(labels) + "\n")


if __name__ == "__main__":
    qa_json = sys.argv[1]  # data/data/qa.json
    get_validation_labels(qa_json)

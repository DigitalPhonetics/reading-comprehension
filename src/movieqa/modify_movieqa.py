import sys
import os
import inspect
import json
import collections
import csv
import numpy as np

# import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir + '/core')

import util


class Synonyms:
    def __init__(self, synonymfile):
        self.synonyms = {}

        with open(synonymfile) as f:
            reader = csv.reader(f, delimiter="\t")
            # skip header
            next(reader, None)
            for line in reader:
                # print(line)
                phrase, exclude, synonym, alternative_synonyms = line
                phrases_to_exclude = exclude.split(", ")
                exclude = [phrase_to_exclude for phrase_to_exclude in phrases_to_exclude if phrase_to_exclude.strip()]
                phrase = phrase.strip()
                self.synonyms[phrase] = (exclude, synonym, alternative_synonyms)

    def _phrase_in_text(self, phrase, text):
        phrase_space_before = ' ' + phrase
        phrase_space_after = phrase + ' '

        # match single words - they must appear as single words
        if not " " in phrase:
            words = text.split()
            if phrase in words:
                # additionally make sure that the word does not appear as part of another word in the text
                for word in words:
                    if phrase in word and word != phrase:
                        return False
                return True
        # match phrases: they must either have a space after them or before them
        elif phrase_space_after in text or phrase_space_before in text:
            return True
        return False

    def replace(self, text):
        """
        :param text: string
        :return:
        """
        was_modified = False
        # keep track of phrases that were replaced to avoid circular replacements,
        # e.g. film -> movie, movie -> film
        replaced_phrases = []
        for phrase in self.synonyms:
            if self._phrase_in_text(phrase, text) and phrase not in replaced_phrases:
                phrases_to_exclude, synonym, alternative_synonyms = self.synonyms[phrase]
                # print("phrase", phrase, phrases_to_exclude, synonym, alternative_synonyms)
                # do not treat multiple appearances of same phrase in text differently
                contains_phrase_to_exclude = False
                for phrase_to_exclude in phrases_to_exclude:
                    if phrase_to_exclude in text:
                        # print("phrase to exclude: " + phrase_to_exclude)
                        contains_phrase_to_exclude = True
                        break
                if not contains_phrase_to_exclude:
                    text = text.replace(phrase, synonym)
                    was_modified = True
                    replaced_phrases.append(phrase)

        return text, was_modified


QAInfo = collections.namedtuple('QAInfo',
                                'qid question answers imdb_key correct_index plot_alignment video_clips')


def read_qa_json(qa_file, split=None, qids=None):
    """Create a list of QaInfo for all question and answers.
    """
    with open(qa_file, 'r') as f:
        qa_json = json.load(f)

    qa_list = []
    for qa in qa_json:
        keep = True
        id = qa['qid']
        if split:
            if not id.startswith(split):
                keep = False
            if qids:
                if id not in qids:
                    keep = False
        if keep:
            qa_list.append(
                QAInfo(id, qa['question'], qa['answers'], qa['imdb_key'], qa['correct_index'], qa['plot_alignment'],
                       qa['video_clips']))

    count_answer_length(qa_list)
    return qa_list

def count_answer_length(qa_list):
    answer_lengths = []
    for qa in qa_list:
        for answer in qa.answers:
            answer_lengths.append(len(answer.split()))
    
    print("average %.2f, standard deviation %.2f " % (np.mean(answer_lengths), np.std(answer_lengths)))
    print(answer_lengths)
    exit()
    

def modify_validation_questions(qa_file, synonyms_file, outfile):
    synonyms = Synonyms(synonyms_file)

    qas = read_qa_json(qa_file, split='val')

    modified_texts_counter = 0

    print("Modified questions:")
    for k, question in enumerate(qas):
        question_text = util.normalize_text(question.question)
        modified_question, was_modified = synonyms.replace(" ".join(question_text))
        # if modified_question != question:
        if was_modified:
            modified_texts_counter += 1
            print("%s: %s --> %s" % (question.qid, " ".join(question_text), modified_question))
            qas[k] = qas[k]._replace(question=modified_question)

        # convert namedtuple to dict to dump as json
        qas[k] = qas[k]._asdict()

    print("Modified %d questions, %.2f%% of the dataset" % (
    modified_texts_counter, float(modified_texts_counter) / len(qas) * 100))

    json_list = json.dumps(qas, indent=4)
    with open(outfile, "w") as of:
        of.write(json_list)


if __name__ == '__main__':
    json_file = sys.argv[1]  # data/data/qa.json
    synonyms_file = sys.argv[2]  # '../../../data_movieqa/similar_words/valid_qa_manual_synonyms.csv'
    modified_json_file = sys.argv[3]  # 'data/data/qa_val_synonyms.json'

    modify_validation_questions(json_file, synonyms_file, modified_json_file)

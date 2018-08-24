# prepare word embeddings, records and vocabulary

import sys
import os
import numpy as np
import tensorflow as tf
import _pickle as pickle

present_path = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.join(present_path, '../'))
import core.util as util
import movieqa.data_conf as data_conf

if not os.path.exists(data_conf.DATA_PATH):
    print(
        "ERROR: could not find MovieqQA data folder."
        "Please manually download the dataset and save the files to: %s" % data_conf.DATA_PATH)
    exit(1)

sys.path.append('data')

import movieqa.data.data_loader as movie

plot_dict = {}


# creation of data for one set
def create_movieqa_data(qa_json_file, name, outfolder, embeddings, qa_ids=None):
    valid_count = 0

    movie.cfg.QA_JSON = qa_json_file
    print("Preprocessing qa file and creating records for %s" % movie.cfg.QA_JSON)

    mqa = movie.DataLoader()
    story, qa = mqa.get_story_qa_data(name, 'split_plot')
    set_path = outfolder + "/" + name + ".tfrecords"
    writer = tf.python_io.TFRecordWriter(set_path)

    # filter questions by ids
    if qa_ids:
        qa = filter_qa(qa, qa_ids)
        print("Selected %d questions based on %d provided ids" % (len(qa), len(qa_ids)))
        with open(os.path.join(outfolder, 'val.pickle'), 'wb') as handle:
            pickle.dump(qa, handle)

    for k, question in enumerate(qa):
        q = []
        ans = []
        l = np.zeros(shape=[5], dtype=float)

        ex = tf.train.SequenceExample()
        words = util.normalize_text(question.question)

        # lowercase now
        words = [word.lower() for word in words]

        movie_id = question.imdb_key
        question_size = len(words)

        if name != "test":
            l[question.correct_index] = 1.0
        if words[0] in util.question_types:
            question_type = util.question_types[words[0]]
        else:
            question_type = -1
        ex.context.feature["question_type"].int64_list.value.append(question_type)

        for i, word in enumerate(words):
            if i < data_conf.Q_MAX_WORD_PER_SENT_COUNT:
                w_vec = (util.get_word_vector(embeddings, word, data_conf.EMBEDDING_SIZE))
                if not w_vec:
                    w_vec = (util.get_word_vector(embeddings, word, data_conf.EMBEDDING_SIZE))
                q.append(w_vec)

        if not movie_id in plot_dict:
            plot = story.get(movie_id)
            p_word_ids = create_plot_record(embeddings, plot, movie_id)
            plot_dict[movie_id] = p_word_ids
        else:
            p_word_ids = plot_dict[movie_id]

        for i, answer in enumerate(question.answers):
            a = []
            words = util.normalize_text(answer)

            for j, word in enumerate(words):
                if j < data_conf.Q_MAX_WORD_PER_SENT_COUNT:
                    w_vec = (util.get_word_vector(embeddings, word, data_conf.EMBEDDING_SIZE))
                    if not w_vec:
                        w_vec = (util.get_word_vector(embeddings, word, data_conf.EMBEDDING_SIZE))
                    a.append(w_vec)
            ans.append(a)

        q_type_feature = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[question_type]))

        q_size_feature = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[question_size]))

        movie_id_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[str.encode(movie_id)]))

        label_list_feature = [
            tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
            for label in l]

        answer_list_feature = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=aw))
            for aw in ans]

        plot_list_feature = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=pl))
            for pl in p_word_ids]

        question_list_feature = [tf.train.Feature(int64_list=tf.train.Int64List(
            value=q))]

        feature_list = {
            "labels": tf.train.FeatureList(
                feature=label_list_feature),
            "answers": tf.train.FeatureList(
                feature=answer_list_feature),
            "question": tf.train.FeatureList(
                feature=question_list_feature),
            "plot": tf.train.FeatureList(
                feature=plot_list_feature),
        }

        context = tf.train.Features(feature={
            "question_type": q_type_feature,
            "question_size": q_size_feature,
            "movie_id": movie_id_feature
        })

        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        example_sequence = tf.train.SequenceExample(
            feature_lists=feature_lists, context=context)
        serialized = example_sequence.SerializeToString()
        writer.write(serialized)
        valid_count += 1

    print(name + ' set completed - files written to ' + set_path)


def filter_qa(qas, qa_ids):
    return [qa for qa in qas if qa.qid in qa_ids]


# create vocab ids for plot
def create_plot_record(embeddings, plot, movie_id):
    p = []
    sent_lens = []

    plot_size = 0
    for i, pl in enumerate(plot):
        words = util.normalize_text(plot[i])
        if (len(words) > 0) and (words[0] != ''):
            p_sent = []
            word_count = 0
            plot_size += 1

            for j, word in enumerate(words):
                if (j < data_conf.P_MAX_WORD_PER_SENT_COUNT) and (plot_size < data_conf.P_MAX_SENT_COUNT):
                    p_sent.append(util.get_word_vector(embeddings, word, data_conf.EMBEDDING_SIZE))
                    word_count += 1
            sent_lens.append(word_count)
            p.append(p_sent)
    return p


def create_data_set(dataset_file, setname, outfolder, embeddings):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    create_movieqa_data(dataset_file, setname, outfolder, embeddings)


def load_and_create_vocab():
    print('loading glove model...')
    embeddings = util.loadGloveModel(data_conf.PRETRAINED_EMBEDDINGS_PATH)
    print('loading complete - loaded model with ' + str(len(embeddings)) + ' words')
    util.init_vocab(data_conf.EMBEDDING_SIZE)
    # use previous vocab for adding new words in adversarial paraphrasing
    util.restore_vocab(data_conf.EMBEDDING_DIR)
    return embeddings


def create_complete_dataset():
    embeddings = load_and_create_vocab()
    create_data_set(movie.cfg.QA_JSON, 'train', data_conf.TRAIN_RECORD_PATH, embeddings)
    create_data_set(data_conf.EVAL_FILE, 'val', data_conf.EVAL_RECORD_PATH, embeddings)
    create_data_set(movie.cfg.QA_JSON, 'test', data_conf.TEST_RECORD_PATH, embeddings)
    print("saving embeddings")
    util.save_embeddings(data_conf.EMBEDDING_DIR, data_conf.EMBEDDING_SIZE)


def create_validation_dataset(split):
    print("Prepare embeddings for modified input ...")
    embeddings = load_and_create_vocab()

    create_movieqa_data(data_conf.EVAL_FILE, split, data_conf.EVAL_RECORD_PATH, embeddings)
    # save updated vocab file with additional new words
    new_vocab_size = util.save_embeddings(data_conf.EMBEDDING_DIR, data_conf.EMBEDDING_SIZE)
    return new_vocab_size


def read_qa_ids(filename):
    with open(filename, "r") as f:
        return [id.strip() for id in f.readlines()]


def create_200_random_validation_dataset(qa_ids_file):
    embeddings = load_and_create_vocab()

    outfolder = os.path.join(data_conf.RECORD_DIR, 'val_random_200')
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    qa_ids = read_qa_ids(qa_ids_file)

    create_movieqa_data(movie.cfg.QA_JSON, 'val', outfolder, embeddings, qa_ids)

    print("saving embeddings")
    util.save_embeddings(data_conf.EMBEDDING_DIR, data_conf.EMBEDDING_SIZE)


if __name__ == "__main__":
    qa_ids_file = sys.argv[1]  # 'data/200_random_validation_qas_white_box_attacks.txt'
    create_200_random_validation_dataset(qa_ids_file)

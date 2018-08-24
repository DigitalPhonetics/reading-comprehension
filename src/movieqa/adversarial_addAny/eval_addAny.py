import sys
import os

sys.path.append('data')
# os.chdir("..")

import movieqa.data.data_loader as movie
import core.util as util
from random import randrange
from random import shuffle
import random
import glob
import tensorflow as tf
import core.model as model
import numpy as np
import _pickle as pickle


# implementation of AddAny/AddCommon from Jia and Liang 2017

def add_plot_sentence(plot, sentence):
    new_sent = np.zeros(shape=[1, 1, plot.shape[2]], dtype=np.int64)
    for i, index in enumerate(sentence):
        new_sent[0][0][i] = index
    m_plot = np.append(plot, new_sent, 1)
    return m_plot


def run_evaluation(model_type, attack_type, model_folder, examples_folder, instances_to_attack):
    if model_type == "lstm":
        import movieqa.run_lstm as runner
    else:
        import movieqa.run_cnn as runner

    runner.data_conf.TRAIN_DIR = model_folder
    mode = attack_type
    sent_out_dir = examples_folder

    load = False
    check_sents = []
    check_found = []
    check_num = 0
    corr_probs = []
    if not tf.gfile.Exists(sent_out_dir):
        tf.gfile.MakeDirs(sent_out_dir)
    else:
        checkpoints = glob.glob(sent_out_dir + "/[!accuracies]*")
        checkpoints = sorted(checkpoints, reverse=True)
        print(checkpoints)
        latest = checkpoints[0]
        splitted = latest.split(".txt")[0]
        check_num = int(splitted[len(splitted) - 1]) + 1

        check = open(latest, encoding="utf8")
        for line in check:
            parts = line.replace('\n', '').split("\t")
            check_words = parts[0].split(" ")
            print(check_words)
            check_sents.append(check_words)
            last_prob = float(parts[1])
            found = parts[2]
            if found == 'True':
                b_found = True
            else:
                b_found = False
            corr_probs.append(last_prob)
            check_found.append(b_found)

        load = True

    emb_dir = runner.data_conf.EMBEDDING_DIR

    vectors, vocab = util.load_embeddings(emb_dir)
    rev_vocab = dict(zip(vocab.values(), vocab.keys()))
    # print(rev_vocab)
    filename = "adversarial_addAny/common_english.txt"
    # length of the distractor sentence
    d = 10
    # pool size of common words to sample from for each word in the distractor sentence
    poolsize = 10
    common_words = {}
    fin = open(filename, encoding="utf8")
    for line in fin:
        word = line.replace('\n', '')
        # print(word)
        if word in rev_vocab:
            common_words[word] = rev_vocab[word]
        else:
            print('ERROR: word "%s" not in vocab. Run add_common_words_to_vocab.py first.' % word)
            exit(1)

    with open(instances_to_attack + '/val.pickle', 'rb') as handle:
        qa = pickle.load(handle)

    with open("question_references.txt", "w") as file:
        for q in qa:
            file.write(str(q.qid) + "\t" + q.question + "\n")

    w_s = []
    w_choices = []
    w_found = []

    q_inds = []
    pools = []

    for k, question in enumerate(qa):
        # load question indices
        q_words = util.normalize_text(question.question)
        q_ind = []
        for word in q_words:
            q_ind.append(rev_vocab[word])

        a_words = []
        for i, answer in enumerate(question.answers):
            if not i == int(question.correct_index):
                words = util.normalize_text(answer)
                a_words.extend(words)
        w = []
        w_choice = []
        rand_sent = ""

        for i in range(0, d):
            c_word = check_sents[k][i]
            w_index = rev_vocab[c_word]
            rand_sent += (c_word + " ")

            w.append(w_index)
            w_choice.append(i)

        found = check_found[k]
        w_found.append(found)

        w_s.append(w)

        print(w_s)

    filepath = instances_to_attack + '/*.tfrecords'
    filenames = glob.glob(filepath)

    global_step = tf.contrib.framework.get_or_create_global_step()
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(runner.get_single_sample)
    dataset = dataset.repeat(poolsize * d)
    batch_size = 1

    dataset = dataset.padded_batch(batch_size, padded_shapes=(
        [None], [5, None], [None], (), [None, None], ()))

    iterator = dataset.make_one_shot_iterator()

    next_q, next_a, next_l, next_plot_ids, next_plots, next_q_types = iterator.get_next()
    add_sent = tf.placeholder(tf.int64, shape=[None])
    # sent_exp = tf.expand_dims(add_sent,0)
    m_p = tf.py_func(add_plot_sentence, [next_plots, add_sent], [tf.int64])[0]
    # m_p = next_plots
    # m_p = tf.concat([next_plots,sent_exp],axis=0)

    logits, atts, sent_atts, _ = runner.predict_batch([next_q, next_a, m_p], training=False)
    probabs = model.compute_probabilities(logits=logits)
    accuracy_example = tf.reduce_mean(model.compute_accuracies(logits=logits, labels=next_l, dim=1))

    to_restore = tf.contrib.slim.get_variables_to_restore(exclude=["embeddings"])
    saver = tf.train.Saver(to_restore)

    p_counts = 0
    last_p = ''
    p_id = 0
    f_counter = 0
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(runner.data_conf.TRAIN_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
        _ = sess.run(runner.set_embeddings_op, feed_dict={runner.place: vectors})
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        accs = corr_probs

        total_acc = 0.0

        for k, question in enumerate(qa):
            w_copy = [x for x in w_s[k]]
            print("==============")

            acc_val, probs_val, gs_val, q_type_val, q_val, atts_val, sent_atts_val, labels_val, p_val, a_val, p_id_val = sess.run(
                [accuracy_example, probabs, global_step, next_q_types, next_q, atts,
                 sent_atts, next_l, m_p,
                 next_a, next_plot_ids], feed_dict={add_sent: w_copy})
            sent = ""
            for word in w_copy:
                sent += (" " + vocab[word])
            print(sent + " - acc: " + str(acc_val))
            corr = np.argmax(labels_val[0])
            pred_val = probs_val[0][corr]

            filename = ''
            q_s = ''
            for index in q_val[0]:
                word = (vocab[index])
                q_s += (word + ' ')
                filename += (word + '_')
            predicted_probabilities = probs_val[0]
            labels = labels_val[0]

            p_id = 'test'
            path = sent_out_dir + "/plots/" + p_id + "/" + filename
            if (p_counts < 20):
                for i, a_att in enumerate(atts_val[0]):
                    # a_att = np.max(a_att, 1)
                    qa_s = q_s + "? (acc: " + str(acc_val) + ")\n "
                    for index in a_val[0][i]:
                        qa_s += (vocab[index] + ' ')
                    lv = " (label: " + str(int(labels[i])) + " - prediction: " + (
                        str("%.2f" % (predicted_probabilities[i] * 100))) + "%)"
                    qa_s += lv

                    a_sents = []
                    y_labels = []

                    for j, att in enumerate(a_att):
                        a_s = []
                        y_labels.append(str("%.2f" % (sent_atts_val[0][i][j] * 100)) + "%")
                        for index in p_val[0][j]:
                            a_s.append(vocab[index])
                        a_sents.append(a_s)
                    # util.plot_attention(np.array(a_att), np.array(a_sents), qa_s, y_labels, path,filename)
                last_p = p_id
                p_counts += 1
            total_acc += acc_val
            print(total_acc / (k + 1))
        with open(sent_out_dir + "/accuracies_eval.txt", "a") as file:
            file.write(str(total_acc / (len(qa))) + "\n")
        return total_acc / (len(qa))

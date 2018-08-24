# train and evaluation functions for hierarchical CNN model

import glob
import os

import tensorflow as tf
import time
import core.model as model
import core.util as util
import movieqa.data_conf as data_conf
import movieqa.conf_cnn as model_conf
from random import randrange
from random import shuffle

import numpy as np

if not os.path.exists(data_conf.RECORD_DIR):
    os.makedirs(data_conf.RECORD_DIR)
    import movieqa.preprocess as pp

    pp.create_complete_dataset()


def load_embeddings():
    global vectors, vocab, embeddings, place, set_embeddings_op
    vectors, vocab = util.load_embeddings(data_conf.EMBEDDING_DIR)
    data_conf.VOCAB_SIZE = len(vectors)

    # init word embeddings
    embeddings = tf.Variable(
        tf.random_uniform([data_conf.VOCAB_SIZE, data_conf.EMBEDDING_SIZE], -1.3, 1.3), name="embeddings",
        trainable=False)

    place = tf.placeholder(tf.float32, shape=embeddings.shape)
    set_embeddings_op = tf.assign(embeddings, place, validate_shape=True)


load_embeddings()


ANSWER_COUNT = 5

dropout_op = tf.make_template(name_='dropout', func_=model.dropout)
prepare_op = tf.make_template(name_='prepare_embedding_a', func_=model.prep_embedding)
attention_op = tf.make_template(name_='prepare_attention', func_=model.prep_attention)
compare_op = tf.make_template(name_='compare', func_=model.compare_submult)
compare_op_2 = tf.make_template(name_='compare_2', func_=model.compare_submult)
convolution_op = tf.make_template(name_='convolution', func_=model.cnn)
convolution_op_2 = tf.make_template(name_='convolution1', func_=model.cnn)
soft_prep_op = tf.make_template(name_='softmax', func_=model.softmax_prep)
update_op = tf.make_template(name_='update', func_=model.update_params)


# load embeddings representation for vocab indices
def get_emb(indices):
    zero = tf.cast(0, dtype=tf.int64)
    zeros = tf.zeros(shape=(tf.shape(indices)[0], data_conf.EMBEDDING_SIZE))
    condition = tf.greater(indices, zero)
    res = tf.where(condition, tf.nn.embedding_lookup(embeddings, indices), zeros)
    return res


# main batch prediction op
def predict_batch(data, training):
    def predict_step(data):
        sample = data
        q = sample[0]
        q = get_emb(q)
        answers = sample[1]
        # [num_answers x num_words] -> [num_answers x num_words x emb_size]
        answers = tf.map_fn(get_emb, answers, dtype=tf.float32)
        # [num_sentences x num_words] -> [num_sentences x num_words x emb_size]
        p = sample[2]
        # Keep the word ids for the plot for printing the original words
        # (needed in run_adversarial because plot words are changed in the graph)
        p_word_indices = p
        p = tf.map_fn(get_emb, p, dtype=tf.float32)

        p_drop = dropout_op(p, training, model_conf.DROPOUT_RATE)

        # [num_sentences x num_words x emb_size] -> [num_sentences x num_words x hidden_size]
        p_prep = prepare_op(p_drop, model_conf.HIDDEN_SIZE)

        q_drop = dropout_op(q, training, model_conf.DROPOUT_RATE)
        # [num_words x hidden_size]
        q_prep = prepare_op(q_drop, model_conf.HIDDEN_SIZE)

        answers_drop = dropout_op(answers, training, model_conf.DROPOUT_RATE)
        answers_drop = tf.reshape(answers_drop, shape=(ANSWER_COUNT, -1, data_conf.EMBEDDING_SIZE))
        answers_prep = prepare_op(answers_drop, model_conf.HIDDEN_SIZE)

        # stage one: compare each plot sentence to the question and each answer
        def p_sent_step(p_sent):
            # compare a plot sentence to the question
            hq = attention_op(q_prep, p_sent, model_conf.HIDDEN_SIZE)
            tq = compare_op(p_sent, hq, model_conf.HIDDEN_SIZE)

            # compare a plot sentence to an answer
            def a_step(a):
                ha = attention_op(a, p_sent, model_conf.HIDDEN_SIZE)
                ta = compare_op(p_sent, ha, model_conf.HIDDEN_SIZE)

                return ta

            # compare plot sentence to each answer
            tanswers = tf.map_fn(a_step, elems=answers_prep)
            return tq, tanswers

        # tqs: [num_sentences x num_words x hidden_size]
        # tas: [num_sentences x num_answers x num_words_in_sentence x hidden_size]
        tqs, tas = tf.map_fn(p_sent_step, elems=p_prep, dtype=(tf.float32, tf.float32))
        # tas: [num_answers x num_sentences x num_words_in_sentence x hidden_size]
        tas = tf.einsum('ijkl->jikl', tas)

        q_prep = tf.expand_dims(q_prep, 0)
        # [1 x num_words x 2* hidden_size]
        q_con = tf.concat([q_prep, q_prep], axis=2)
        # [num_answers x num_words x 2* hidden_size]
        a_con = tf.concat([answers_prep, answers_prep], axis=2)
        # [1 x hidden_size * num_filter_sizes]
        rq_sent_feats, _ = convolution_op(q_con, model_conf.FILTER_SIZES, model_conf.HIDDEN_SIZE)
        # [num_answers x hidden_size * num_filter_sizes]
        ra_sent_feats, _ = convolution_op(a_con, model_conf.FILTER_SIZES, model_conf.HIDDEN_SIZE)

        # convolution over weighted plot representation [tq|ta] for one answer
        # num_sentences is first dimension of elems
        def a_conv(elems):
            # ta: [num_sentences x num_words x hidden_size]
            # a_sent: [hidden_size * num_filter_sizes]
            ta, a_sent = elems
            # [num_sentences x num_words x 2* hidden_size]
            tqa_con = tf.concat([tqs, ta], axis=2)
            # rpqa_sent_feats: [num_sentences x hidden_size * num_filter_sizes]
            # att_vis: []?
            # TODO two options for word attention visualization:
            # 1) after the CNN layer: coloring of words with their context within the sentence
            # -> use att_vis
            # 2) before the CNN layer (only word attention + compare, analogous to how sentence attention is extracted):
            #  each word has always the same color within a plot
            # -> use word_atts below
            rpqa_sent_feats, att_vis = convolution_op(tqa_con, model_conf.FILTER_SIZES, model_conf.HIDDEN_SIZE, 3,
                                                      'SAME')

            # [num_sentences x num_words]
            word_atts = tf.reduce_mean(tqa_con, axis=2)
            # we dont need softmax here as we only need values represented as strong and weak coloring of the words
            # and no real distribution among all words in the sentence
            # looks strange with softmax
            # word_atts = tf.nn.softmax(word_atts)

            # stage two from here, all sentence features are computed
            # [num_sentences x hidden_size * num_filter_sizes]
            hq_sent = attention_op(rq_sent_feats, rpqa_sent_feats, model_conf.HIDDEN_SIZE)
            # [1 x hidden_size * num_filter_sizes]
            a_sent = tf.expand_dims(a_sent, 0)
            # [num_sentences x hidden_size * num_filter_sizes]
            ha_sent = attention_op(a_sent, rpqa_sent_feats, model_conf.HIDDEN_SIZE)
            # compare is element-wise, so dimension of output does not change
            tq_sent = compare_op_2(rpqa_sent_feats, hq_sent, model_conf.HIDDEN_SIZE)
            ta_sent = compare_op_2(rpqa_sent_feats, ha_sent, model_conf.HIDDEN_SIZE)
            # [num_sentences x 2 * hidden_size * num_filter_sizes]
            tqa_sent = tf.concat([tq_sent, ta_sent], 1)
            return tqa_sent, word_atts

        # first dimension of tas and ra_sent_feats is the number of answers
        # t_sent: [num_answers x num_sentences x 2 * hidden_size * num_filter_sizes]
        # word_atts: [num_answers x num_sentences x num_words]
        t_sent, word_atts = tf.map_fn(a_conv, elems=[tas, ra_sent_feats], dtype=(tf.float32, tf.float32))

        # [num_answers x hidden_size * num_filter_sizes]
        r_final_feats, _ = convolution_op_2(t_sent, model_conf.FILTER_SIZES, model_conf.HIDDEN_SIZE, 1, 'SAME')
        # [num_answers]
        result = soft_prep_op(r_final_feats, model_conf.HIDDEN_SIZE)
        result = tf.reshape(result, shape=[-1])

        # [num_answers x num_sentences]
        sent_atts = tf.reduce_mean(t_sent, axis=2)
        sent_soft = tf.nn.softmax(sent_atts)
        return result, word_atts, sent_soft, p_word_indices

    predict_step_op = tf.make_template(name_='predict_step', func_=predict_step)

    # first dimension of data is batch size
    batch_predictions = tf.map_fn(fn=predict_step_op, parallel_iterations=1,
                                  elems=data, infer_shape=False,
                                  dtype=(tf.float32, tf.float32, tf.float32, tf.int64))
    return batch_predictions


# get single record sample for set
def get_single_sample(sample):
    context_features = {
        "question_size": tf.FixedLenFeature([], dtype=tf.int64),
        "question_type": tf.FixedLenFeature([], dtype=tf.int64),
        "movie_id": tf.FixedLenFeature([], dtype=tf.string),
    }
    sequence_features = {
        "question": tf.VarLenFeature(dtype=tf.int64),
        "labels": tf.VarLenFeature(dtype=tf.float32),
        "answers": tf.VarLenFeature(dtype=tf.int64),
        "plot": tf.VarLenFeature(dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=sample,
        context_features=context_features,
        sequence_features=sequence_features
    )

    label = sequence_parsed['labels']
    answers = sequence_parsed['answers']
    plot = sequence_parsed['plot']

    question = sequence_parsed['question']

    question_type = context_parsed['question_type']
    movie_id = context_parsed['movie_id']

    plot = tf.sparse_tensor_to_dense(plot)

    answers = tf.sparse_tensor_to_dense(answers)
    question = tf.sparse_tensor_to_dense(question)
    label = tf.sparse_tensor_to_dense(label)

    answers = tf.reshape(answers, shape=[ANSWER_COUNT, -1])
    label = tf.reshape(label, shape=[ANSWER_COUNT])
    question = tf.reshape(question, shape=[-1])

    return question, answers, label, movie_id, plot, question_type


# main eval function for one epoch
def eval_model():
    if not tf.gfile.Exists(data_conf.EVAL_DIR):
        tf.gfile.MakeDirs(data_conf.EVAL_DIR)

    util.save_config_values(data_conf, data_conf.EVAL_DIR + "/data")
    util.save_config_values(model_conf, data_conf.EVAL_DIR + "/model")

    filepath = data_conf.EVAL_RECORD_PATH + '/*'
    filenames = glob.glob(filepath)

    print("Evaluate model on records stored in %s" % str(filenames))

    global_step = tf.contrib.framework.get_or_create_global_step()
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(get_single_sample)
    batch_size = 1

    dataset = dataset.padded_batch(batch_size, padded_shapes=(
        [None], [ANSWER_COUNT, None], [None], (), [None, None], ()))

    iterator = dataset.make_one_shot_iterator()

    next_q, next_a, next_l, next_plot_ids, next_plots, next_q_types = iterator.get_next()

    logits, atts, sent_atts, pl_d = predict_batch([next_q, next_a, next_plots], training=False)

    next_q_types = tf.reshape(next_q_types, ())

    probabs = model.compute_probabilities(logits=logits)
    loss_example = model.compute_batch_mean_loss(logits, next_l, model_conf.LOSS_FUNC)
    accuracy_example = tf.reduce_mean(model.compute_accuracies(logits=logits, labels=next_l, dim=1))

    # do not take saved embeddings from model graph for case the vocab size has changed
    to_restore = tf.contrib.slim.get_variables_to_restore(exclude=["embeddings"])
    saver = tf.train.Saver(to_restore)
    summary_writer = tf.summary.FileWriter(data_conf.TRAIN_DIR)

    step = 0
    total_acc = 0.0
    total_loss = 0.0
    type_counts = np.zeros(6, dtype=np.int32)
    type_accs = np.zeros(6)
    p_counts = 0
    last_p = ''
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(data_conf.TRAIN_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
        # print("Feeding embeddings %s " % str(vectors))
        _ = sess.run(set_embeddings_op, feed_dict={place: vectors})
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                loss_val, acc_val, probs_val, gs_val, q_type_val, q_val, atts_val, sent_atts_val, labels_val, p_val, a_val, p_id_val = sess.run(
                    [loss_example, accuracy_example, probabs, global_step, next_q_types, next_q, atts, sent_atts,
                     next_l,
                     pl_d, next_a, next_plot_ids])
                type_accs[q_type_val + 1] += acc_val
                type_counts[q_type_val + 1] += 1

                total_loss += loss_val
                total_acc += acc_val

                predicted_probabilities = probs_val[0]
                sentence_attentions = sent_atts_val[0]
                pred_index = np.argmax(predicted_probabilities)
                labels = labels_val[0]
                gold = np.argmax(labels)

                filename = ''
                q_s = ''
                for index in q_val[0]:
                    word = (vocab[index])
                    q_s += (word + ' ')
                    filename += (word + '_')

                filename += "?"

                p_id = str(p_id_val[0].decode("utf-8"))
                path = data_conf.EVAL_DIR + "/plots/" + p_id + "_" + str(step) + "/"  # + filename

                # write attention heat-map
                if (p_id != last_p and p_counts < data_conf.PLOT_SAMPLES_NUM):
                    # if True:
                    for i, a_att in enumerate(atts_val[0]):
                        qa_s = q_s + "? (acc: " + str(acc_val) + ")\n "
                        for index in a_val[0][i]:
                            word = vocab[index]
                            qa_s += (word + ' ')
                            filename += word + "_"
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
                        util.plot_attention(np.array(a_att), np.array(a_sents), qa_s, y_labels, path, filename)
                    last_p = p_id
                    p_counts += 1

                print("Sample loss: " + str(loss_val))
                print("Sample labels: " + str(labels))
                print("Sample probabilities: " + str(predicted_probabilities))
                print("Sample acc: " + str(acc_val))

                util.print_predictions(data_conf.EVAL_DIR, step, gold, predicted_probabilities, data_conf.MODE)
                util.print_sentence_attentions(data_conf.EVAL_DIR, step, sentence_attentions)

                step += 1

                print("Total acc: " + str(total_acc / step))
                print("Local_step: " + str(step * batch_size))
                print("Global_step: " + str(gs_val))
                print("===========================================")

        except tf.errors.OutOfRangeError:

            summary = tf.Summary()
            summary.value.add(tag='validation_loss', simple_value=total_loss / step)
            summary.value.add(tag='validation_accuracy', simple_value=(total_acc / step))
            summary_writer.add_summary(summary, gs_val)
            keys = util.get_question_keys()
            if data_conf.MODE == "val":
                with open(data_conf.EVAL_DIR + "/val_accuracy.txt", "a") as file:
                    file.write("global step: " + str(gs_val) + " - total accuracy: " + str(
                        total_acc / step) + "- total loss: " + str(total_loss / step) + "\n")
                    file.write("===================================================================" + "\n")
                    util.save_eval_score(
                        "global step: " + str(gs_val) + " - acc : " + str(
                            total_acc / step) + " - total loss: " + str(
                            total_loss / step) + " - " + data_conf.TRAIN_DIR + "_" + str(gs_val))
        finally:
            coord.request_stop()
        coord.join(threads)


# main training function for one epoch
def train_model():
    print("train")
    global_step = tf.contrib.framework.get_or_create_global_step()

    init = False
    if not tf.gfile.Exists(data_conf.TRAIN_DIR):
        init = True
        print("RESTORING WEIGHTS")
        tf.gfile.MakeDirs(data_conf.TRAIN_DIR)
    util.save_config_values(data_conf, data_conf.TRAIN_DIR + "/data")
    util.save_config_values(model_conf, data_conf.TRAIN_DIR + "/model")

    filenames = glob.glob(data_conf.TRAIN_RECORD_PATH + '/*')
    print("Reading training dataset from %s" % filenames)
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(get_single_sample)
    dataset = dataset.shuffle(buffer_size=9000)
    dataset = dataset.repeat(model_conf.NUM_EPOCHS)
    batch_size = model_conf.BATCH_SIZE

    dataset = dataset.padded_batch(model_conf.BATCH_SIZE, padded_shapes=(
        [None], [ANSWER_COUNT, None], [None], (), [None, None], ()))

    iterator = dataset.make_one_shot_iterator()

    next_q, next_a, next_l, next_plot_ids, next_plots, next_q_types = iterator.get_next()

    logits, _, _, _ = predict_batch([next_q, next_a, next_plots], training=True)

    probabs = model.compute_probabilities(logits=logits)
    loss_batch = model.compute_batch_mean_loss(logits, next_l, model_conf.LOSS_FUNC)
    accuracy = model.compute_accuracies(logits=logits, labels=next_l, dim=1)
    accuracy_batch = tf.reduce_mean(accuracy)
    tf.summary.scalar("train_accuracy", accuracy_batch)
    tf.summary.scalar("train_loss", loss_batch)

    training_op = update_op(loss_batch, global_step, model_conf.OPTIMIZER, model_conf.INITIAL_LEARNING_RATE)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=data_conf.TRAIN_DIR,
            save_checkpoint_secs=60,
            save_summaries_steps=5,
            hooks=[tf.train.StopAtStepHook(last_step=model_conf.MAX_STEPS),
                   ], config=config) as sess:
        step = 0
        total_acc = 0.0
        if init:
            _ = sess.run(set_embeddings_op, feed_dict={place: vectors})
        while not sess.should_stop():
            _, loss_val, acc_val, probs_val, lab_val, gs_val = sess.run(
                [training_op, loss_batch, accuracy_batch, probabs, next_l, global_step])
            print(probs_val)
            print(lab_val)
            print("Batch loss: " + str(loss_val))
            print("Batch acc: " + str(acc_val))
            step += 1
            total_acc += acc_val

            print("Total acc: " + str(total_acc / step))
            print("Local_step: " + str(step * batch_size))
            print("Global_step: " + str(gs_val))
            print("===========================================")
    util.copy_model(data_conf.TRAIN_DIR, gs_val)


if __name__ == '__main__':
    eval_model()

# word-level and sentence-level white-box adversarial attack for hierarchical CNN and RNN-LSTM model

import glob
import os
import sys

import tensorflow as tf
import core.model as model
import core.util as util
import movieqa.data_conf as data_conf
from random import randrange
from random import shuffle

import numpy as np

model_conf = None

if not os.path.exists(data_conf.RECORD_DIR):
    os.makedirs(data_conf.RECORD_DIR)
    import movieqa.preprocess as pp

    pp.create_complete_dataset()

vectors, vocab = util.load_embeddings(data_conf.EMBEDDING_DIR)
data_conf.VOCAB_SIZE = len(vectors)

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
lstm_op = tf.make_template(name_='lstm1', func_=model.lstm)
lstm_op_2 = tf.make_template(name_='lstm2', func_=model.lstm)

embeddings = tf.Variable(
    tf.random_uniform([data_conf.VOCAB_SIZE, data_conf.EMBEDDING_SIZE], -1.3, 1.3), name="embeddings", trainable=False)
place = tf.placeholder(tf.float32, shape=embeddings.shape)
set_embeddings_op = tf.assign(embeddings, place, validate_shape=True)


# word-level attack: modify words in the most attended sentence
def modify_plot_sentence(p, w_att, s_att, labels, num_modified_words, percentage_attacked_samples):
    prob = randrange(0, 100)
    if prob <= percentage_attacked_samples:
        for i, p_samp in enumerate(p):
            w_samp = w_att[i]
            s_samp = s_att[i]
            l_samp = labels[i]
            corr_ind = np.argmax(l_samp)
            w_corr = w_samp[corr_ind]

            w_corr = np.mean(w_corr, 2)
            s_corr = s_samp[corr_ind]
            s_max_ind = np.argmax(s_corr)
            p_sent = p_samp[s_max_ind]
            w_red = w_corr[s_max_ind]

            w_order = np.argsort(w_red).tolist()

            valid_count = 0
            for id in p_sent:
                if id == 0:
                    break
                else:
                    valid_count += 1
            s_len = valid_count
            if valid_count > 0:
                rand_inds = [z for z in range(valid_count)]
                shuffle(rand_inds)
            else:
                rand_inds = []

            for k in range(0, num_modified_words):
                replace_most_attended_words = True
                if replace_most_attended_words:
                    if k >= s_len:
                        break
                    m_ind = int(w_order.pop())
                    r_ind = m_ind
                # replace random words, not used in the experiments reported in the paper
                else:
                    if valid_count > 0 and len(rand_inds) > 0:
                        r_ind = rand_inds.pop()
                    else:
                        r_ind = 0

                r_word = randrange(0, data_conf.VOCAB_SIZE)
                p[i][s_max_ind][r_ind] = r_word

    return p


# sentence-level attack: remove the most attended sentence
def remove_plot_sentence(p, s_att, labels):
    m_p = np.zeros(shape=(1, len(p[0]) - 1, len(p[0][0])), dtype=np.int64)
    for i, p_samp in enumerate(p):
        s_samp = s_att[i]
        l_samp = labels[i]
        corr_ind = np.argmax(l_samp)
        s_corr = s_samp[corr_ind]
        s_max_ind = np.argmax(s_corr)
        sl1 = p[i][:s_max_ind]
        if s_max_ind < (len(p[i]) - 1):
            sl2 = p[i][s_max_ind + 1:]
            conc = np.concatenate([sl1, sl2])
            m_p[i] = conc
        else:
            m_p[i] = p[i][:len(p[i]) - 1]
    return m_p


# load embeddings representation for vocab indices
def get_emb(indices):
    zero = tf.cast(0, dtype=tf.int64)
    zeros = tf.zeros(shape=(tf.shape(indices)[0], data_conf.EMBEDDING_SIZE))
    condition = tf.greater(indices, zero)
    res = tf.where(condition, tf.nn.embedding_lookup(embeddings, indices), zeros)
    return res


# main batch prediction op
def predict_batch(model_type, data, training):
    def predict_step(data):
        sample = data
        q = sample[0]
        q = get_emb(q)
        aws = sample[1]
        aws = tf.map_fn(get_emb, aws, dtype=tf.float32)
        p = sample[2]
        with tf.device('/cpu:0'):
            p_d = p
        p = tf.map_fn(get_emb, p_d, dtype=tf.float32)

        p_drop = dropout_op(p, training, model_conf.DROPOUT_RATE)
        p_drop = tf.reshape(p_drop, shape=(tf.shape(p)[0], -1, data_conf.EMBEDDING_SIZE))
        p_prep = prepare_op(p_drop, model_conf.HIDDEN_SIZE)

        q_drop = dropout_op(q, training, model_conf.DROPOUT_RATE)
        q_prep = prepare_op(q_drop, model_conf.HIDDEN_SIZE)

        aws_drop = dropout_op(aws, training, model_conf.DROPOUT_RATE)
        aws_drop = tf.reshape(aws_drop, shape=(ANSWER_COUNT, -1, data_conf.EMBEDDING_SIZE))
        aws_prep = prepare_op(aws_drop, model_conf.HIDDEN_SIZE)

        # stage one
        def p_sent_step(p_sent):
            h = attention_op(q_prep, p_sent, model_conf.HIDDEN_SIZE)
            t = compare_op(p_sent, h, model_conf.HIDDEN_SIZE)

            def a_step(a):
                ha = attention_op(a, p_sent, model_conf.HIDDEN_SIZE)
                ta = compare_op(p_sent, ha, model_conf.HIDDEN_SIZE)

                return ta

            a_feats = tf.map_fn(a_step, elems=aws_prep)
            return t, a_feats

        tqs, tas = tf.map_fn(p_sent_step, elems=p_prep, dtype=(tf.float32, tf.float32))
        tas = tf.einsum('ijkl->jikl', tas)

        q_prep = tf.expand_dims(q_prep, 0)
        q_con = tf.concat([q_prep, q_prep], axis=2)
        a_con = tf.concat([aws_prep, aws_prep], axis=2)
        if model_type == "lstm":
            q_sent_feats = lstm_op(q_con, model_conf.HIDDEN_SIZE)
            a_sent_feats = lstm_op(a_con, model_conf.HIDDEN_SIZE)
        else:
            q_sent_feats, _ = convolution_op(q_con, model_conf.FILTER_SIZES, model_conf.HIDDEN_SIZE)
            a_sent_feats, _ = convolution_op(a_con, model_conf.FILTER_SIZES, model_conf.HIDDEN_SIZE)

        def a_conv(elems):
            ta, a_sent = elems
            con = tf.concat([tqs, ta], axis=2)
            # att_vis = tf.reduce_mean(con, axis=2)
            if model_type == "lstm":
                pqa_sent_feats = lstm_op(con, model_conf.HIDDEN_SIZE)
            else:
                pqa_sent_feats, _ = convolution_op(con, model_conf.FILTER_SIZES, model_conf.HIDDEN_SIZE, 3, 'SAME')

            # stage two from here, all sentence features are computed
            hq_sent = attention_op(q_sent_feats, pqa_sent_feats, model_conf.HIDDEN_SIZE)
            a_sent = tf.expand_dims(a_sent, 0)
            ha_sent = attention_op(a_sent, pqa_sent_feats, model_conf.HIDDEN_SIZE)
            tq_sent = compare_op_2(pqa_sent_feats, hq_sent, model_conf.HIDDEN_SIZE)
            ta_sent = compare_op_2(pqa_sent_feats, ha_sent, model_conf.HIDDEN_SIZE)
            sent_feats = tf.concat([tq_sent, ta_sent], 1)
            return sent_feats, con

        t_sent, pqa_atts = tf.map_fn(a_conv, elems=[tas, a_sent_feats], dtype=(tf.float32, tf.float32))

        if model_type == "lstm":
            r_final_feats = lstm_op_2(t_sent, model_conf.HIDDEN_SIZE)
        else:
            r_final_feats, _ = convolution_op_2(t_sent, model_conf.FILTER_SIZES, model_conf.HIDDEN_SIZE, 1)

        sent_atts = tf.reduce_mean(t_sent, axis=2)
        sent_soft = tf.nn.softmax(sent_atts)
        result = soft_prep_op(r_final_feats, model_conf.HIDDEN_SIZE)
        result = tf.reshape(result, shape=[-1])
        return result, pqa_atts, sent_soft, p_d

    predict_step_op = tf.make_template(name_='predict_step', func_=predict_step)

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


# main train function for one epoch
def train_model(model_type, attack_level, num_modified_words, percentage_attacked_samples):
    print("train")
    print("%s white-box adversarial attack modifies %d words of %d%% of the instances: " % (
        attack_level, num_modified_words, percentage_attacked_samples))

    global model_conf
    if model_type == "lstm":
        import movieqa.conf_lstm as model_conf
    else:
        import movieqa.conf_cnn as model_conf

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
    dataset = dataset.repeat(data_conf.NUM_EPOCHS)
    batch_size = data_conf.BATCH_SIZE

    dataset = dataset.padded_batch(data_conf.BATCH_SIZE, padded_shapes=(
        [None], [ANSWER_COUNT, None], [None], (), [None, None], ()))

    iterator = dataset.make_one_shot_iterator()

    next_q, next_a, next_l, next_plot_ids, next_plots, next_q_types = iterator.get_next()

    _, w_atts, s_atts, _ = predict_batch(model_type, [next_q, next_a, next_plots], training=True)

    if attack_level == "sentence":
        m_p = tf.py_func(remove_plot_sentence, [next_plots, s_atts, next_l], [tf.int64])[0]
    elif attack_level == "word":
        m_p = tf.py_func(modify_plot_sentence, [next_plots, w_atts, s_atts, next_l], [tf.int64])[0]

    logits, _, _, _ = predict_batch(model_type, [next_q, next_a, m_p], training=True)

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


# main eval function for one epoch
def eval_model(model_type, attack_level, num_modified_words, percentage_attacked_samples):
    print("evaluate")
    print("%s white-box adversarial attack modifies %d words of %d%% of the instances: " % (
        attack_level, num_modified_words, percentage_attacked_samples))

    global model_conf
    if model_type == "lstm":
        import movieqa.conf_lstm as model_conf
    else:
        import movieqa.conf_cnn as model_conf

    if not tf.gfile.Exists(data_conf.EVAL_DIR):
        tf.gfile.MakeDirs(data_conf.EVAL_DIR)

    util.save_config_values(data_conf, data_conf.TRAIN_DIR + "/data")
    util.save_config_values(model_conf, data_conf.TRAIN_DIR + "/model")

    filepath = data_conf.EVAL_RECORD_PATH + '/*'
    filenames = glob.glob(filepath)
    print("Evaluating adversarial attack on %s" % filenames)

    global_step = tf.contrib.framework.get_or_create_global_step()
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(get_single_sample)
    batch_size = 1

    dataset = dataset.padded_batch(batch_size, padded_shapes=(
        [None], [ANSWER_COUNT, None], [None], (), [None, None], ()))

    iterator = dataset.make_one_shot_iterator()

    next_q, next_a, next_l, next_plot_ids, next_plots, next_q_types = iterator.get_next()

    _, w_atts, s_atts, _ = predict_batch(model_type, [next_q, next_a, next_plots], training=False)

    if attack_level == "sentence":
        m_p = tf.py_func(remove_plot_sentence, [next_plots, s_atts, next_l], [tf.int64])[0]
    elif attack_level == "word":
        m_p = tf.py_func(modify_plot_sentence,
                         [next_plots, w_atts, s_atts, next_l, num_modified_words, percentage_attacked_samples],
                         [tf.int64])[0]

    logits, atts, sent_atts, pl_d = predict_batch(model_type, [next_q, next_a, m_p], training=False)

    next_q_types = tf.reshape(next_q_types, ())

    probabs = model.compute_probabilities(logits=logits)
    loss_example = model.compute_batch_mean_loss(logits, next_l, model_conf.LOSS_FUNC)
    accuracy_example = tf.reduce_mean(model.compute_accuracies(logits=logits, labels=next_l, dim=1))

    to_restore = tf.contrib.slim.get_variables_to_restore(exclude=["embeddings"])
    saver = tf.train.Saver(to_restore)
    summary_writer = tf.summary.FileWriter(data_conf.TRAIN_DIR)

    step = 0
    total_acc = 0.0
    total_prec = 0.0
    total_rank = 0.0
    total_loss = 0.0
    type_counts = np.zeros(6, dtype=np.int32)
    type_accs = np.zeros(6)
    max_sent_atts = {}
    max_atts = {}
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

                predicted_probabilities = probs_val[0]
                sentence_attentions = sent_atts_val[0]

                total_loss += loss_val
                total_acc += acc_val

                pred_index = np.argmax(probs_val[0])
                labels = labels_val[0]
                gold = np.argmax(labels)

                filename = ''
                q_s = ''
                for index in q_val[0]:
                    word = (vocab[index])
                    q_s += (word + ' ')
                    filename += (word + '_')

                p_id = str(p_id_val[0].decode("utf-8"))
                path = data_conf.EVAL_DIR + "/plots/" + p_id + "/" + filename

                corr_ans = np.argmax(labels_val[0])

                max_att_val = np.argmax(sent_atts_val[0][corr_ans])

                att_row = np.max(atts_val[0][corr_ans][max_att_val], 1)

                red = np.max(atts_val[0][corr_ans][max_att_val], 1)

                att_inds = np.argsort(red)[::-1]

                if (p_id != last_p and p_counts < 20):
                    for i, a_att in enumerate(atts_val[0]):
                        a_att = np.mean(a_att, 2)
                        qa_s = q_s + "? (acc: " + str(acc_val) + ")\n "
                        for index in a_val[0][i]:
                            qa_s += (vocab[index] + ' ')
                        lv = " (label: " + str(int(labels_val[0][i])) + " - prediction: " + (
                            str("%.2f" % (probs_val[0][i] * 100))) + "%)"
                        qa_s += lv

                        a_sents = []
                        y_labels = []

                        for j, att in enumerate(a_att):
                            a_s = []
                            y_labels.append(str("%.2f" % (sent_atts_val[0][i][j] * 100)) + "%")
                            for index in p_val[0][j]:
                                a_s.append(vocab[index])
                            a_sents.append(a_s)
                    # util.plot_attention(np.array(a_att), np.array(a_sents),qa_s,y_labels,path,filename)
                    last_p = p_id
                    p_counts += 1

                m_ap = util.example_precision(probs_val[0], labels_val[0], 5)
                rank = util.example_rank(probs_val[0], labels_val[0], 5)
                total_prec += m_ap
                total_rank += rank

                print("Sample loss: " + str(loss_val))
                print("Sample acc: " + str(acc_val))
                print("Sample prec: " + str(m_ap))
                print("Sample rank: " + str(rank))

                util.print_predictions(data_conf.EVAL_DIR, step, gold, predicted_probabilities, data_conf.MODE)
                util.print_sentence_attentions(data_conf.EVAL_DIR, step, sentence_attentions)

                step += 1

                print("Total acc: " + str(total_acc / step))
                print("Total prec: " + str(total_prec / step))
                print("Total rank: " + str(total_rank / step))
                print("Local_step: " + str(step * batch_size))
                print("Global_step: " + str(gs_val))
                if attack_level == "word":
                    print("%d modified word(s)" % num_modified_words)
                print("===========================================")
        except tf.errors.OutOfRangeError:

            summary = tf.Summary()
            summary.value.add(tag='validation_loss', simple_value=total_loss / step)
            summary.value.add(tag='validation_accuracy', simple_value=(total_acc / step))
            summary_writer.add_summary(summary, gs_val)
            keys = util.get_question_keys()
            with open(data_conf.EVAL_DIR + "/accuracy.txt", "a") as file:
                file.write("global step: " + str(gs_val) + " - total accuracy: " + str(
                    total_acc / step) + "- total loss: " + str(total_loss / step) + str(num_modified_words) + "" "\n")
                file.write("Types (name / count / correct / accuracy):\n")
                for entry in zip(keys, type_counts, type_accs, (type_accs / type_counts)):
                    file.write(str(entry) + "\n")
                file.write("===================================================================" + "\n")
                util.save_eval_score(
                    "global step: " + str(gs_val) + " - acc : " + str(
                        total_acc / step) + " - total loss: " + str(
                        total_loss / step) + " - " + data_conf.TRAIN_DIR + "_" + str(gs_val))
        finally:
            coord.request_stop()
        coord.join(threads)

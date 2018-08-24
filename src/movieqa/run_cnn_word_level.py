# original baseline for movieqa without sentence attention, may be not up to date

import glob
import os

import tensorflow as tf
import core.model as model
import core.util as util
import movieqa.data_conf as data_conf
import movieqa.conf_cnn_word_level as model_conf

import numpy as np

print("loading records from %s, loading embeddings from %s" % (data_conf.RECORD_DIR, data_conf.EMBEDDING_DIR))

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
convolution_op_2 = tf.make_template(name_='convolution', func_=model.cnn)
soft_prep_op = tf.make_template(name_='softmax', func_=model.softmax_prep)
update_op = tf.make_template(name_='update', func_=model.update_params)


def get_emb(indices):
    zero = tf.cast(0, dtype=tf.int64)
    zeros = tf.zeros(shape=(tf.shape(indices)[0], data_conf.EMBEDDING_SIZE))
    condition = tf.greater(indices, zero)
    res = tf.where(condition, tf.nn.embedding_lookup(embeddings, indices), zeros)
    return res


# main prediction op for batch
def predict_batch(data, training):
    def predict_step(data):
        sample = data
        q = sample[0]
        q = get_emb(q)
        q_drop = dropout_op(q, training, model_conf.DROPOUT_RATE)
        aws = sample[1]
        p = sample[2]
        p = tf.reshape(p, shape=[-1])
        p = get_emb(p)
        p_drop = dropout_op(p, training, model_conf.DROPOUT_RATE)
        q_prep = prepare_op(q_drop, model_conf.HIDDEN_SIZE)
        p_prep = prepare_op(p_drop, model_conf.HIDDEN_SIZE)
        h = attention_op(q_prep, p_prep, model_conf.HIDDEN_SIZE)
        t = compare_op(p_prep, h, model_conf.HIDDEN_SIZE)

        def answer_step(a):
            a = get_emb(a)
            a_drop = dropout_op(a, training, model_conf.DROPOUT_RATE)
            a_prep = prepare_op(a_drop, model_conf.HIDDEN_SIZE)
            h2 = attention_op(a_prep, p_prep, model_conf.HIDDEN_SIZE)
            t2 = compare_op(p_prep, h2, model_conf.HIDDEN_SIZE)
            t_con = tf.concat([t, t2], axis=1)
            return t_con

        output = tf.map_fn(answer_step, elems=aws, dtype=tf.float32)
        output, _ = convolution_op(output, model_conf.FILTER_SIZES, model_conf.HIDDEN_SIZE)
        result = soft_prep_op(output, model_conf.HIDDEN_SIZE)
        result = tf.reshape(result, shape=[-1])
        return result

    predict_step_op = tf.make_template(name_='predict_step', func_=predict_step)

    batch_predictions = tf.map_fn(fn=predict_step_op, parallel_iterations=10,
                                  elems=data, infer_shape=False,
                                  dtype=tf.float32)
    return batch_predictions


# load next sample from set's record files
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
    plot = tf.reshape(plot, shape=[-1])

    return question, answers, label, movie_id, plot, question_type


# main training function for one epoch
def train_model():
    print("train")
    global_step = tf.contrib.framework.get_or_create_global_step()

    init = False
    if not tf.gfile.Exists(data_conf.TRAIN_DIR):
        init = True
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
        [None], [ANSWER_COUNT, None], [None], (), [None], ()))

    iterator = dataset.make_one_shot_iterator()

    next_q, next_a, next_l, next_plot_ids, next_plots, next_q_types = iterator.get_next()

    logits = predict_batch([next_q, next_a, next_plots], training=True)

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


# main eval / testing function
def eval_model():
    if not tf.gfile.Exists(data_conf.EVAL_DIR):
        tf.gfile.MakeDirs(data_conf.EVAL_DIR)

    util.save_config_values(data_conf, data_conf.EVAL_DIR + "/data")
    util.save_config_values(model_conf, data_conf.EVAL_DIR + "/model")

    filepath = data_conf.EVAL_RECORD_PATH + '/*'
    filenames = glob.glob(filepath)

    print("Evaluate model on %s" % str(filenames))

    global_step = tf.contrib.framework.get_or_create_global_step()
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(get_single_sample)
    batch_size = 1

    dataset = dataset.padded_batch(batch_size, padded_shapes=(
        [None], [ANSWER_COUNT, None], [None], (), [None], ()))

    iterator = dataset.make_one_shot_iterator()

    next_q, next_a, next_l, next_plot_ids, next_plots, next_q_types = iterator.get_next()

    logits = predict_batch([next_q, next_a, next_plots], training=False)

    next_q_types = tf.reshape(next_q_types, ())

    probabs = model.compute_probabilities(logits=logits)
    loss_example = model.compute_batch_mean_loss(logits, next_l, model_conf.LOSS_FUNC)
    accuracy_example = tf.reduce_mean(model.compute_accuracies(logits=logits, labels=next_l, dim=1))

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(data_conf.TRAIN_DIR)

    step = 0
    total_acc = 0.0
    total_loss = 0.0
    type_counts = np.zeros(6, dtype=np.int32)
    type_accs = np.zeros(6)
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(data_conf.TRAIN_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                loss_val, acc_val, probs_val, gs_val, q_type_val, labels_val = sess.run(
                    [loss_example, accuracy_example, probabs, global_step, next_q_types, next_l])

                predicted_probabilities = probs_val[0]
                pred_index = np.argmax(probs_val[0])
                labels = labels_val[0]
                gold = np.argmax(labels)

                type_accs[q_type_val + 1] += acc_val
                type_counts[q_type_val + 1] += 1

                total_loss += loss_val
                total_acc += acc_val

                print("Sample loss: " + str(loss_val))
                print("Sample acc: " + str(acc_val))

                util.print_predictions(data_conf.EVAL_DIR, step, gold, predicted_probabilities, data_conf.MODE)

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


if __name__ == '__main__':
    train_model()
    eval_model()

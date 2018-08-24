# Machine Question Answering with Attention-based Convolutional Neural Networks

This code accompanies the paper [Comparing Attention-based Convolutional and Recurrent Neural Networks:
Success and Limitations in Machine Reading Comprehension](arxiv link) published at CoNLL 2018.

If you use or reimplement any of this source code, please cite the following paper:

```bibtex
@InProceedings{QASuccessAndLimitationsBlohm18,
  title =     {Comparing Attention-based Convolutional and Recurrent Neural Networks:
               Success and Limitations in Machine Reading Comprehension},
  author =    {Blohm, Matthias and Jagfeld, Glorianna and Sood, Ekta and Yu, Xiang and Vu, Thang},
  booktitle = {Proceedings of the 22nd Conference on Computational Natural Language Learning (CoNLL 2018)},
  publisher = {Association for Computational Linguistics},
  location =  {Brussels, Belgium},
  year =      {2018}
}
```

## Prerequisites

0. All paths in these instructions are provided relative to the repository's source folder 'story_understanding'.
   The code was only tested under Linux and will for sure not run under Windows without adapations due to the file path
    formattings.
1. Create (virtual) environment with Python 3.6
    - `python3 -m venv --system-site-packages virtualenv-dir`
    - `source virtualenv-dir/bin/activate`
2. [Install TensorFlow](https://www.tensorflow.org/install/) version 1.5.
    - `pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.5.1-cp36-cp36m-linux_x86_64.whl`
    - Maybe you need to install the additional dependencies matplotlib, pysrt via pip
    
3. To obtain the MovieQA data, [register for an account](http://movieqa.cs.toronto.edu/register/).
   Download the MovieQA dataset, unzip it and put the contents into the folder "src/movieqa/data".
   You need the folders 'data' and 'story' and the scripts config.py, data_loader.py, story_loader.py, __init__.py.
    Since the python scripts of the MovieQA dataset preprocessing code are in python2, but our code is written in python3
    you have to convert the scripts data_loader.py and story_loader.py to python3 by calling the following script from
    within the folder 'src/movieqa' 
    
    ```python   
    python convert_movieqa_to_python3.py
    ```

4. Download [pretrained GloVe model](http://nlp.stanford.edu/data/glove.840B.300d.zip) and extract them into a folder called "glove".
   If the embeddings are stored elsewhere, the PRETRAINED_EMBEDDINGS_PATH variable in the config file needs to be changed.
   
5. The sentence-level black-box adversarial attack requires nltk and the Brown corpus resource.

## Reproducing the main results (model training and evaluation): Hierarchical Attention-based Compare-Aggregate Model & Compare-Aggregate Model

To train models and evaluate them on the validation or test set, the script src/main.py is used,
which has to be called within the src directory.
 
```python
python main.py MODE MODEL_TYPE MODEL_NAME [opts]

MODE: train, val, or test
MODEL_TYPE: word-level-cnn, cnn, lstm
MODEL_NAME: Name of the trained model to save or load
```

MODEL_TYPES:
Our hierarchical attention-based compare-aggregate models have MODEL_TYPE cnn (CNN aggregation function) and
lstm (RNN-LSTM aggregation function).
The word-level only CNN, corresponding to our own slightly modified reimplementation of the
 [Compare-Aggregate model of Wang & Jiang (ICLR 2017)](https://arxiv.org/abs/1611.01747),
 has MODEL_TYPE word_level_cnn.


The outputs are stored in a folder src/movieqa/outputs/MODE_{MODEL_NAME}.

Example call to train a hierarchical model with lstm aggregation function called 'A' from within 'src' folder:

```python
python main.py train cnn A
```

mode == train produces the following outputs in src/movieqa/outputs/train_{MODEL_NAME}:
- checkpoint
- config.txt
- events.out.tfevents....
- graph.pbtxt
- model.ckpt-NO.data...
- model.ckpt-NO.index
- model.ckpt-NO.meta


IMPORTANT: When training for the first time, the dataset records and embeddings have to be created.
For this to be triggered, the folder specified in data_conf.py/RECORD_DIR must not be present/created yet.

The following subfolders will be created under src/movieqa/RECORD_DIR:

- Representation of the dataset splits and plots as tf.records in the folders train, val, test.
- Word-embeddings (GloVe, vectors for words not contained therein are initialized randomly): embeddings_{EMB_DIM}d that contains 
vocab.pickle, vectors.pickle

Example call to evaluate a hierarchical model with CNN aggregation called 'A' on the validation set:

```python
python main.py val cnn A
```

mode == val produces the following outputs in src/movieqa/outputs/val_{MODEL_NAME}.


- val_accuracy.txt: average accuracy and loss on the validation set
- data_config.txt: config values used in this call (from file + arguments)
- model_config: config values used in this call (from file + arguments)
- probabilities.txt: predicted probability distributions over the answer candidates for each question
- attentions.txt: attention distribution over each sentence in the plot for each question (only for hierarchical models)

### Creating and Evaluating Ensembles

Majority-vote ensembles can be evaluated by the script src/eval_ensemble.py.
You can find a usage example in src/run.sh.
Before running ensemble evaluation on the validation set for the first time, you have to create the gold labels file 'src/movieqa/data/data/labels_val.txt'
by running from 'src/movieqa'

```python
python get_validation_labels.py data/data/qa.json
```

## Adversarial Experiments
Note that all adversarial experiments were only implemented for the hierarchical models and are likely
not to work with the word-level CNN.

### Black-box attacks
#### Word-level black-box adversarial attack
Create a modified version of the validation set by calling the following script from the folder 'src/movieqa'

```python
python modify_movieqa.py data/data/qa.json data/validation_synonyms_word_level_black_box_attack.csv data/data/qa_val_synonyms.json
```

Evaluate trained models on the modified validation set as follows (call from 'src' folder):
```python
python main.py val cnn A -eval_file_version synonyms
```

#### Sentence-level black box adversarial attack
Get list of 1000 common English words from Brown corpus by running script src/movieqa/adversarial_addAny/english_words.py
from within the 'adversCreation' folder.

Add the 1000 common English words to the vocabulary by running from 'src/movieqa'
```python
python adversarial_addAny/add_common_words_to_vocab.py
```

Since these attacks are computationally very expensive, we only ran them on a random subset of 200 validation set questions.
To obtain this subset in 'src/movieqa/' run
```python
python preprocess.py data/200_random_validation_qas_white_box_attacks.txt
```

This will extract the 200 random validation instances we used to val.pickle (texts) and val.tfrecords in 'src/movieqa/records/val_random_200'.

Create adversarial sentences with the addCommon attack for all CNN models in 'movieqa/outputs/'cnn_adversarial_eval_models';
run from 'src' folder':
```python
python adversarial_sentence_level_black_box.py create_examples cnn addC cnn_adversarial_eval_models $PROJECT/story-understanding/src/movieqa/records/val_random_200/ -examples_folder addC_adversarial_examples
```

Evaluate the models on the created adversarial sentences:
```python
python adversarial_sentence_level_black_box.py eval_examples cnn addC cnn_adversarial_eval_models $PROJECT/story-understanding/src/movieqa/records/val_random_200/ -examples_folder addC_adversarial_examples
```
### White-box attacks

The white-box attacks are started via 'adversarial_white_box.py' from the 'src' folder.
The average accuracy for the evaluated dataset is written to 'src/movieqa/outputs/{EVAL_SET}_adversarial_{ATTACK_LEVEL}-level_whitebox_{MODEL_NAME}/accuracy.txt'.
See the script for further options.


#### Word-level white-box adversarial attack
Remove 5 most attended to plot words from the most attended sentence of all CNN models in 'movieqa/outputs/'cnn_adversarial_eval_models'
and evaluate on the validation set.
```python
python adversarial_white_box.py val cnn cnn_adversarial_eval_models word -num_modified_words 5
```

#### Sentence-level white-box adversarial attack
Remove most attended sentence of all CNN models in 'movieqa/outputs/'cnn_adversarial_eval_models' and evaluate on the validation set.
```python
python adversarial_white_box.py val cnn cnn_adversarial_eval_models sentence
```
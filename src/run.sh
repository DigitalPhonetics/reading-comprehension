# usage examples for the methods necessary to reproduce the published results

models=(A B C)
model_type=lstm #lstm, cnn, word-level cnn

## train 11 models for 5 epochs ##
epochs=5
for model in "${models[@]}"
do
    for e in $(seq 1 $epochs)
    do
        nohup python -u main.py train $model_type "$model"_"$model_type" -gpuid 0\
         &> train_"$model"_"$model_type"_epoch_"$e".log &
    done
done

## evaluate the models on the validation set ##
for model in "${models[@]}"
do
    nohup python -u main.py val $model_type "$model"_"$model_type" -gpuid -1\
     &> eval_"$model"_"$model_type".log &
done

## majority vote ensemble ##
# manually select the 9 best models to be included in the ensemble
ensemble_models=(2 3) #TODO change here to selected models' names

ensemble_directory=movieqa/outputs/ensemble_"$model_type"/ #path for my local expts movieqa/outputs/ensemble/val_lstm/
mkdir -P $ensemble_directory
for model in "${ensemble_models[@]}"
do
    ln -s $ensemble_directory/probabilities_"$model".txt movieqa/outputs/val_"$model"_"$model_type"/probabilities.txt
done

# evaluate ensemble on validation set
python eval_ensemble.py $ensemble_directory movieqa/data/data/labels_val.txt -ensemble_name "$model_type""on validation set"

## evaluate the models on the test set ##
for model in "${models[@]}"
do
    nohup python -u main.py test $model_type "$model"_"$model_type" -gpuid -1\
     &> test_"$model"_"$model_type".log &
done

### adversarial attacks
## sentence-level black box: apply addAny (e.g. AddCommon) attack to models (e.g. CNN)
# create examples
python adversarial_sentence_level_black_box.py create_examples cnn addC cnn_adversarial_eval_models -examples_folder addC_adversarial_examples


# Evaluate the models on the created adversarial sentences:
python adversarial_sentence_level_black_box.py eval_examples cnn addC cnn_adversarial_eval_models -examples_folder addC_adversarial_examples
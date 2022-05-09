export CUDA_VISIBLE_DEVICES=2

DATA=TASK2-enzh
SUFFIX=roberta
LABEL_SUFFIX=hter
# accelerate launch --fp16 run_quality_estimation.py \
#  --do_train \
#  --do_partial_prediction \
#  --suffix_a mt \
#  --data_dir $DATA/original-data \
#  --model_type bert \
#  --model_path ./chinese-roberta-wwm-ext \
#  --output_dir ./QE_outputs/$DATA-sent-$SUFFIX \
#  --batch_size 8 \
#  --learning_rate 1e-5 \
#  --max_epoch 20 \
#  --valid_steps 500 \
#  --train_type sent \
#  --valid_type sent \
#  --sentlab_suffix $LABEL_SUFFIX \
#  --stop_criterion 10 \
#  --best_metric pearson \
#  --overwrite_output_dir \
#  --overwrite_cache

INFER_PREFIX=test20
python run_quality_estimation.py \
 --do_infer \
 --do_partial_prediction \
 --suffix_a mt \
 --data_dir $DATA/original-data \
 --infer_prefix $INFER_PREFIX \
 --model_type bert \
 --model_path ./QE_outputs/$DATA-sent-$SUFFIX/best_pearson \
 --batch_size 16 \
 --infer_type sent \

python eval_sentence_level.py $DATA/original-data/$INFER_PREFIX.sent $DATA/original-data/$INFER_PREFIX.$LABEL_SUFFIX -v
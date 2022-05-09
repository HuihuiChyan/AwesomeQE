export CUDA_VISIBLE_DEVICES=0

DATA=TASK2-enzh
SUFFIX=roberta
# accelerate launch --fp16 run_quality_estimation.py \
#  --do_train \
#  --do_partial_prediction \
#  --suffix_a mt \
#  --data_dir $DATA/original-data \
#  --model_type bert \
#  --model_path ./chinese-roberta-wwm-ext \
#  --output_dir ./QE_outputs/$DATA-word-$SUFFIX \
#  --batch_size 8 \
#  --learning_rate 1e-5 \
#  --max_epoch 20 \
#  --valid_steps 500 \
#  --train_type word \
#  --valid_type word \
#  --add_gap_to_target_text \
#  --stop_criterion 10 \
#  --best_metric mcc \
#  --overwrite_output_dir \
#  --overwrite_cache

INFER_PREFIX=test21
python run_quality_estimation.py \
 --do_infer \
 --data_dir $DATA/original-data \
 --infer_prefix $INFER_PREFIX \
 --model_type bert \
 --model_path ./QE_outputs/$DATA-word-$SUFFIX/best_mcc \
 --batch_size 16 \
 --infer_type word \

python eval_word_level.py $DATA/original-data/$INFER_PREFIX.word $DATA/original-data/$INFER_PREFIX.tags -v
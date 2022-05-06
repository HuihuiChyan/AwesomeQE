export CUDA_VISIBLE_DEVICES=3

DATA=TASK2-enzh
SUFFIX=01
echo "Model saved to ./QE_outputs/$DATA-word-$SUFFIX"

accelerate launch --fp16 run_quality_estimation.py \
 --do_train \
 --data_dir $DATA/original-data \
 --model_type xlm_r \
 --model_path ./xlm-roberta-base \
 --output_dir ./QE_outputs/$DATA-word-$SUFFIX \
 --batch_size 8 \
 --learning_rate 1e-5 \
 --max_epoch 20 \
 --valid_steps 20 \
 --train_type word \
 --valid_type word \
 --add_gap_to_target_text \
 --best_metric mcc \
 --overwrite_output_dir \
 --overwrite_cache
echo "Model saved to ./QE_outputs/$DATA-word-$SUFFIX"

# python run_quality_estimation.py \
#  --do_infer \
#  --add_gap_to_text \
#  --data_dir ./WMT20_task2/$DATA_DIR \
#  --model_type xlm_r \
#  --model_path ./QE_outputs/WMT20_task2-enzh23095/best_mcc \
#  --output_dir ./WMT20_task2 \
#  --batch_size 16 \
#  --infer_type word \
#  --language_a en \
#  --language_b zh \

# python eval_word_level.py ./WMT20_task2/test.xlmr-base.word ./WMT20_task2/test.tags -v
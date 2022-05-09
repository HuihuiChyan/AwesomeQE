# AwesomeQE: 基于预训练模型的QE系统


QE（Quality Estimation，翻译质量评估）旨在无参考译文的前提下对机器翻译的结果进行自动化评估。QE可以用于在笔译场景下对翻译结果进行初筛，控制后编辑成本，也可以用于对翻译输出进行风险预警和质量控制，在机器翻译的应用场景中具有很重要的价值。

近年来，基于预训练模型的QE成为了主流方法，经过长时间大规模预训练得到的模型，对于数据稀缺的QE任务有很大提升作用。但是，现有的QE框架（比如[OpenKiwi](https://github.com/Unbabel/OpenKiwi) 和 [Transquest][https://github.com/TharinduDR/TransQuest] ）等，仅仅集成了mBERT, XLM-R等少数预训练模型，没有充分挖掘预训练模型的潜力。

本仓库致力于集成尽可能多的预训练模型，包括encoder模型（BERT，XLM等）和encoder-decoder（BART，MarianMT）等。在离线场景下，使用本系统所提供的多个预训练模型的集成系统，不需要额外的数据增强或者架构工程，就可以达到顶尖的QE精度。

有任何问题欢迎随时提出issue，我会第一时间反馈修正（您也可以添加我的微信huanghui2020708）。

## 特征
- 词汇级别、句子级别或者二者联合的翻译质量评估（词汇级别支持GAP预测）；
- 支持几乎所有huggingface模型库中的多语言预训练模型，包括encoder架构和encoder-decoder架构；
  - mBERT ([microsoft/Multilingual-MiniLM-L12-H384 · Hugging Face](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384))
  - XLM-R ([xlm-roberta-base · Hugging Face](https://huggingface.co/xlm-roberta-base))
  - DistilBERT ([distilbert-base-multilingual-cased · Hugging Face](https://huggingface.co/distilbert-base-multilingual-cased))
  - InfoXLM ([microsoft/infoxlm-base · Hugging Face](https://huggingface.co/microsoft/infoxlm-base))
  - mBART50 ([facebook/mbart-large-50-many-to-many-mmt · Hugging Face](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt))
  - MarianMT ([transformers/marian.mdx at main · huggingface/transformers (github.com)](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/marian.mdx)
  - XLM-MLM ([xlm-mlm-100-1280 · Hugging Face](https://huggingface.co/xlm-mlm-100-1280))
  - XLM-TLM ([xlm-mlm-tlm-xnli15-1024 · Hugging Face](https://huggingface.co/xlm-mlm-tlm-xnli15-1024))
  - mMiniLM (https://huggingface.co/nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large)
- 支持仅使用单端（源端或者目标端）进行翻译质量评估，评估源句子的复杂度或者目标句子的流畅度；
- 基于 [accelerate](https://github.com/huggingface/accelerate) 实现了混合精度和分布式训练；

## 结果

- mlqe-pe TASK1, DA预测

|            | ende test20 | ende test21 | enzh test20 | enzh test21 | average |
| ---------- | ----------- | ----------- | ----------- | ----------- | ------- |
| xlmr-base  | 0.4058      | 0.4691      | 0.4134      | 0.4843      | 0.4432  |
| mbert      | 0.3847      | 0.4060      | 0.4397      | 0.4923      | 0.4307  |
| distilbert | 0.3626      | 0.4427      | 0.3400      | 0.4074      | 0.3882  |
| mMiniLMv2  | 0.3417      | 0.3410      | 0.3423      | 0.4055      | 0.3576  |
| opus-mt    | 0.3454      | 0.3837      | 0.4640      | 0.4315      | 0.4062  |
| bert(mono) | 0.4518      | 0.4063      | 0.4528      | 0.4978      | 0.4522  |

- mlqe-pe TASK2, hter预测

|            | ende test20 | ende test21 | enzh test20 | enzh test21 | average |
| ---------- | ----------- | ----------- | ----------- | ----------- | ------- |
| xlmr-base  | 0.4691      | 0.5374      | 0.3228      | 0.2654      | 0.3987  |
| mbert      | 0.4853      | 0.5422      | 0.3366      | 0.2683      | 0.4081  |
| distilbert | 0.4318      | 0.4666      | 0.3395      | 0.2574      | 0.3738  |
| mMiniLMv2  | 0.4109      | 0.4229      | 0.3083      | 0.2782      | 0.3551  |
| opus-mt    | 0.5180      | 0.6134      | 0.3409      | 0.2826      | 0.4387  |
| bert(mono) | 0.4770      | 0.5438      | 0.3688      | 0.2830      | 0.4182  |

- mlqe-pe TASK2, target tag预测

|            | ende test20 | ende test21 | enzh test20 | enzh test21 | average |
| ---------- | ----------- | ----------- | ----------- | ----------- | ------- |
| xlmr-base  | 0.3627      | 0.3286      | 0.3925      | 0.2042      | 0.3220  |
| mbert      | 0.3371      | 0.3079      | 0.4118      | 0.2846      | 0.3354  |
| bert(mono) | 0.3473      | 0.3430      | 0.4077      | 0.2568      | 0.3387  |

## 示例

- 句子级别训练

```bash
DATA=TASK2-enzh
accelerate launch --fp16 run_quality_estimation.py \
 --do_train \
 --data_dir $DATA/original-data \
 --model_type xlmr \
 --model_path ./xlm-roberta-base \
 --output_dir ./QE_outputs/$DATA-sent \
 --batch_size 8 \
 --learning_rate 1e-5 \
 --max_epoch 20 \
 --valid_steps 500 \
 --train_type sent \
 --valid_type sent \
 --sentlab_suffix hter \
 --best_metric pearson \
 --overwrite_output_dir \
 --overwrite_cache
```

- 句子级别预测

```bash
python run_quality_estimation.py \
 --do_infer \
 --data_dir $DATA/original-data \
 --model_type xlmr \
 --model_path ./QE_outputs/$DATA-sent/best_pearson \
 --output_dir $DATA/original-data \
 --batch_size 16 \
 --infer_type sent \
```

- 词汇级别训练

```bash
DATA=TASK2-enzh
accelerate launch --fp16 run_quality_estimation.py \
 --do_train \
 --data_dir $DATA/original-data \
 --model_type xlmr \
 --model_path ./xlm-roberta-base \
 --output_dir ./QE_outputs/$DATA-word \
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
```

- 词汇级别预测

```bash
python run_quality_estimation.py \
 --do_infer \
 --add_gap_to_text \
 --data_dir $DATA/original-data \
 --model_type xlmr \
 --model_path ./QE_outputs/$DATA-word/best_mcc \
 --output_dir $DATA/original-data \
 --batch_size 16 \
 --infer_type word \
```
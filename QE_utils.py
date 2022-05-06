import os
import torch
import random
import logging
import datasets
from typing import Optional
from functools import partial
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

import pdb

logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForQE(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    padding: Optional[bool] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side

        if padding_side == "right":
            if "wordlab_lines" in features[0].keys():
                batch["wordlab_lines"] = [
                    list(line["wordlab_lines"]) + [-100] * (sequence_length - len(line["wordlab_lines"])) for line in features
                ]
            if "subword_mask" in features[0].keys():
                batch["subword_mask"] = [
                    list(line["subword_mask"]) + [0] * (sequence_length - len(line["subword_mask"])) for line in features
                ]
        else:
            if "wordlab_lines" in features[0].keys():
                batch["wordlab_lines"] = [
                    [-100] * (sequence_length - len(line["wordlab_lines"])) + list(line) for line in features
                ]
            if "subword_mask" in features[0].keys():
                batch["subword_mask"] = [
                    [0] * (sequence_length - len(line["subword_mask"])) + list(line) for line in features
                ]

        for k, v in batch.items():
            if k == 'sentlab_lines':
                batch[k] = torch.tensor(v, dtype=torch.float)
            else:
                batch[k] = torch.tensor(v, dtype=torch.long)

        return batch

def get_subword_mask(word_ids, wordlab_line, token_type_ids, attention_mask, predict_side="second"):
    previous_word_idx = None
    subword_mask = []
    subword_labels = []
    if predict_side == "first":
        segment = 0
    else:
        segment = 1
    for word_idx, token_type, attn in zip(word_ids, token_type_ids, attention_mask):
        if token_type == segment and attn == 1:
            if word_idx is None:
                subword_mask.append(0)
                if wordlab_line is not None:
                    subword_labels.append(-100)
            elif word_idx != previous_word_idx:
                subword_mask.append(1)
                if wordlab_line is not None:
                    subword_labels.append(wordlab_line.pop(0))
            else:
                subword_mask.append(0)
                if wordlab_line is not None:
                    subword_labels.append(-100)
        else:
            subword_mask.append(0)
            if wordlab_line is not None:
                subword_labels.append(-100)
        previous_word_idx = word_idx
    if wordlab_line is None:
        subword_labels = None
    return subword_mask, subword_labels

def get_token_type_ids_for_sentpair_in_roberta(special_tokens_mask):
    msk_sum = 0
    token_type_ids = []
    for msk in special_tokens_mask:
        msk_sum += msk
        if msk_sum <= 2:
            token_type_ids.append(0)
        elif 2 < msk_sum <= 4:
            token_type_ids.append(1)
        else:
            token_type_ids.append(0)

    return token_type_ids

def read_examples(args, data_prefix, data_type=None):

    logger.info("LOOKING AT Huihui's QE %s data in data_dir" % (data_prefix))

    lines_a = [line.strip() for line in open(os.path.join(args.data_dir, data_prefix+'.'+args.suffix_a), encoding="utf-8").readlines()]
    examples = {"lines_a": lines_a}
    if not args.do_partial_prediction:
        examples["lines_b"] = [line.strip() for line in open(os.path.join(args.data_dir, data_prefix+'.'+args.suffix_b), encoding="utf-8").readlines()]
    if data_type in ('sent', 'joint'):
        examples["sentlab_lines"] = [float(line.strip()) for line in open(os.path.join(args.data_dir, data_prefix+'.'+args.sentlab_suffix), encoding="utf-8").readlines()]
    if data_type in ('word', 'joint'):
        word2label = {'OK':1, 'BAD':0}
        examples["wordlab_lines"] = [[word2label[tag] for tag in line.strip().split()] for line in open(os.path.join(args.data_dir, data_prefix+'.'+args.wordlab_suffix), encoding="utf-8").readlines()]
    if args.has_additional_feature:
        examples["additional_features"] = [[float(l) for l in line.strip().split()] for line in open(os.path.join(args.data_dir, data_prefix+'.'+args.feature_suffix), encoding="utf-8").readlines()]
         
    return examples

def preprocess_function_for_single_sequence(
    examples,
    tokenizer,
    max_length=None,
    pad_to_max_length=False,
    has_language_embedding=False,
    has_segment_embedding=False,
    lang2id=None, 
    language_a=None,
    language_b=None,
    is_split_into_words=False,
    add_gap_to_target_text=True,
):

    assert "lines_b" not in examples.data.keys()

    if "wordlab_lines" in examples.data.keys():
        assert is_split_into_words == True
        truncation = False
    else:
        truncation = True

    if is_split_into_words:
        examples["lines_a"] = [line.split() for line in examples["lines_a"]]
        if add_gap_to_text: # make sure word labels already include GAP
            examples["lines_a"] = [(' <gap> ' + ' <gap> '.join(line) + ' <gap> ').split() for line in examples["lines_a"]]

    padding = "max_length" if pad_to_max_length else False

    encodings = tokenizer(examples["lines_a"],
                          padding=padding,
                          truncation=truncation, 
                          max_length=max_length,
                          return_special_tokens_mask=True,
                          is_split_into_words=is_split_into_words)
  
    if "wordlab_lines" in examples.data.keys():
        batched_wordlab_lines = []
        batched_subword_masks = []
        for idx, wordlab_line in enumerate(examples["wordlab_lines"]):
            token_type_ids = [0 for i in encodings.input_ids[idx]]
            attention_mask = encodings.attention_mask[idx]
            subword_mask, wordlab_line = get_subword_mask(encodings[idx].word_ids, wordlab_line, token_type_ids, attention_mask, predict_side="first")
            batched_subword_masks.append(subword_mask)
            batched_wordlab_lines.append(wordlab_line)
        encodings["subword_mask"] = batched_subword_masks
        encodings["wordlab_lines"] = batched_wordlab_lines

    if has_language_embedding:
        batched_token_type_ids = []
        for i, wordlab_line in enumerate(examples["token_type_ids"]):
            token_type_ids = [lang2id[language_a] for t in token_type_ids]
            batched_token_type_ids.append(token_type_ids)
        encodings["token_type_ids"] = batched_token_type_ids

    return encodings

def preprocess_function_for_paired_sequence(
    examples,
    tokenizer,
    max_length=None,
    pad_to_max_length=False,
    has_language_embedding=False,
    has_segment_embedding=False,
    lang2id=None, 
    language_a=None,
    language_b=None,
    is_split_into_words=False,
    add_gap_to_target_text=True,
):

    if "wordlab_lines" in examples.data.keys():
        assert is_split_into_words == True
        truncation = "only_first"
    else:
        truncation = True

    if is_split_into_words:
        examples["lines_a"] = [line.split() for line in examples["lines_a"]]
        examples["lines_b"] = [line.split() for line in examples["lines_b"]]
        if add_gap_to_target_text: # make sure word labels already include GAP
            examples["lines_b"] = [(' <gap> ' + ' <gap> '.join(line) + ' <gap> ').split() for line in examples["lines_b"]]

    # for i in range(len(examples["lines_a"])):
    #     assert len(examples["lines_b"][i]) == len(examples["wordlab_lines"][i])

    padding = "max_length" if pad_to_max_length else False

    encodings = tokenizer(examples["lines_a"],
                          text_pair=examples["lines_b"],
                          padding=padding,
                          truncation=truncation, 
                          max_length=max_length,
                          return_special_tokens_mask=True,
                          is_split_into_words=is_split_into_words)
  
    if "wordlab_lines" in examples.data.keys():
        batched_wordlab_lines = []
        batched_subword_masks = []
        for idx, wordlab_line in enumerate(examples["wordlab_lines"]):
            if not (has_segment_embedding or has_language_embedding):
                token_type_ids = get_token_type_ids_for_sentpair_in_roberta(encodings['special_tokens_mask'][idx])
            else:
                token_type_ids = encodings['token_type_ids'][idx]
            attention_mask = encodings.attention_mask[idx]
            subword_mask, wordlab_line = get_subword_mask(encodings[idx].word_ids, wordlab_line, token_type_ids, attention_mask)
            batched_subword_masks.append(subword_mask)
            batched_wordlab_lines.append(wordlab_line)
        encodings["subword_mask"] = batched_subword_masks
        encodings["wordlab_lines"] = batched_wordlab_lines

    if has_language_embedding:
        batched_token_type_ids = []
        for i, wordlab_line in enumerate(examples["token_type_ids"]):
            token_type_ids = [lang2id[language_a] if t == 0 else lang2id[language_b] for t in token_type_ids]
            batched_token_type_ids.append(token_type_ids)
        encodings["token_type_ids"] = batched_token_type_ids

    return encodings

def read_and_process_examples(args, tokenizer, evaluate=False):

    if args.do_train:
        data_prefix = args.valid_prefix if evaluate else args.train_prefix
        data_type = args.valid_type if evaluate else args.train_type
    else:
        data_prefix = args.infer_prefix
        data_type = None

    dataset = read_examples(args, data_prefix=data_prefix, data_type=data_type)
    dataset = datasets.Dataset.from_dict(dataset)

    is_split_into_words = (data_type in ['word', 'joint'])
    has_language_embedding = (args.model_type in ['xlm-tlm', 'xlm-mlm'])
    has_segment_embedding = (args.model_type == 'bert')

    if args.model_type == "xlm-mlm":
        args.langtok_a = args.langtok_b = 'en'

    if args.do_partial_prediction:
        preprocess_function = preprocess_function_for_single_sequence
        remove_columns = ["lines_a"]
    else:
        preprocess_function = preprocess_function_for_paired_sequence
        remove_columns = ["lines_a", "lines_b"]
    
    partial_preprocess_function = partial(preprocess_function,
                                          tokenizer=tokenizer,
                                          max_length=args.max_length,
                                          pad_to_max_length=args.pad_to_max_length,
                                          has_language_embedding=has_language_embedding,
                                          has_segment_embedding=has_segment_embedding,
                                          lang2id=args.lang2id, 
                                          language_a=args.langtok_a,
                                          language_b=args.langtok_b,
                                          is_split_into_words=is_split_into_words,
                                          add_gap_to_target_text=args.add_gap_to_target_text)

    dataset = dataset.map(partial_preprocess_function, batched=True, remove_columns=remove_columns)
                  
    # Log a few random samples from the training set:
    for index in random.sample(range(len(dataset)), 100):
        logger.info(f"Sample {index} of the {data_prefix} set: {dataset[index]}.")

    return dataset
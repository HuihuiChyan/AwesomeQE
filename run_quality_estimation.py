import os
import torch
import random
import logging
import argparse
import datasets
import numpy as np
import transformers
from torch import nn
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
	AutoConfig,
	AutoTokenizer,
	default_data_collator,
	get_linear_schedule_with_warmup,
)
from modeling_QE import (
	BertPreTrainedModelForQE,
	XLMPreTrainedModelForQE,
	XLMRobertaPreTrainedModelForQE,
	DistilBertPreTrainedModelForQE,
	MBartPreTrainedModelForQE,
	MarianPreTrainedModelForQE,
)
from QE_utils import read_and_process_examples, DataCollatorForQE

import pdb

logger = logging.getLogger(__name__)

best_result = {'best_step': 0, 'best_metric': {}}

def pearson_and_spearman(preds, labels):
	pearson_corr = pearsonr(preds, labels)[0]
	spearman_corr = spearmanr(preds, labels)[0]
	return {
		"pearson": pearson_corr,
		"spearmanr": spearman_corr,
		"corr": (pearson_corr + spearman_corr) / 2,
	}

def mcc_and_multi_f1(preds, labels):
	mcc = matthews_corrcoef(labels, preds)
	preds = (np.arange(2)==np.array(preds)[:,None]).astype(np.integer)
	labels = (np.arange(2)==np.array(labels)[:,None]).astype(np.integer)
	f1_bad, f1_good = f1_score(preds, labels, average=None, pos_label=None)

	return {
		"mcc": mcc,
		"f1_good": f1_good,
		"f1_bad": f1_bad,
		"f1_multi": f1_good * f1_bad,
	}

def evaluate(args, model, tokenizer, valid_dataloader, accelerator, global_step, stop_time):

	best_output_dir = os.path.join(args.output_dir, 'best_'+args.best_metric)

	# Eval!
	logger.info("***** Running evaluation *****")
	logger.info("  Num examples = %d", len(valid_dataloader) * args.batch_size * accelerator.num_processes)
	logger.info("  Batch size = %d", args.batch_size * accelerator.num_processes)

	model.eval()

	all_sent_outputs = []
	all_sent_labels = []
	all_word_outputs = []
	all_word_labels = []
	for batch in valid_dataloader:
		
		with torch.no_grad():
			sent_outputs, word_outputs = model(**batch)

		if args.valid_type in ['sent', 'joint']:
			all_sent_outputs.extend(accelerator.gather(sent_outputs).detach().cpu().tolist())
			all_sent_labels.extend(accelerator.gather(batch["sentlab_lines"]).detach().cpu().tolist())

		if args.valid_type in ['word', 'joint']:
			if not args.pad_to_max_length:
				word_outputs = accelerator.pad_across_processes(word_outputs, dim=-1, pad_index=-100)
				word_labels = accelerator.pad_across_processes(batch["wordlab_lines"], dim=-1, pad_index=-100)
				subword_mask = accelerator.pad_across_processes(batch["subword_mask"], dim=-1, pad_index=-100)
			
			word_outputs = accelerator.gather(word_outputs)
			word_labels = accelerator.gather(word_labels)
			subword_mask = accelerator.gather(subword_mask)

			active_word_positions = (subword_mask == 1).view(-1)
			word_labels = word_labels.view(-1)[active_word_positions]
			word_outputs = torch.argmax(word_outputs.view(-1, 2)[active_word_positions], axis=-1)

			all_word_outputs.extend(word_outputs.detach().cpu().tolist())
			all_word_labels.extend(word_labels.detach().cpu().tolist())

	result = {}
	if args.valid_type in ['joint', 'sent']:
		sent_result = pearson_and_spearman(all_sent_outputs, all_sent_labels)
		result['pearson'] = sent_result['pearson']
		result['spearmanr'] = sent_result['spearmanr']

	if args.valid_type in ['joint', 'word']:
		word_result = mcc_and_multi_f1(all_word_outputs, all_word_labels)
		result['mcc'] = word_result['mcc']
		result['f1_good'] = word_result['f1_good']
		result['f1_bad'] = word_result['f1_bad']
		result['f1_multi'] = word_result['f1_multi']
		
	if best_result['best_metric'] == {} or\
		result[args.best_metric] > best_result['best_metric'][args.best_metric]:

		best_result['best_step'] = global_step
		best_result['best_metric'] = result
		
		accelerator.wait_for_everyone()
		unwrapped_model = accelerator.unwrap_model(model)
		unwrapped_model.save_pretrained(best_output_dir, save_function=accelerator.save)
		if accelerator.is_main_process:
			tokenizer.save_pretrained(best_output_dir)

		logger.info("Saving best model checkpoint to %s", best_output_dir)
		stop_time = 0
	else:
		stop_time += 1

	logger.info("***** Eval result at step {} *****".format(global_step))
	for key in sorted(result.keys()):
		logger.info("  %s = %s", key, str(result[key]))

	logger.info("***** Best eval result at step {} *****".format(best_result['best_step']))
	for key in sorted(best_result['best_metric'].keys()):		
		logger.info("  %s = %s", key, str(best_result['best_metric'][key]))

	return stop_time

def train(args, model, tokenizer, train_dataloader, valid_dataloader, accelerator):

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{	"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
			"weight_decay": 0.0
		},
	]

	optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

	# Use the device given by the `accelerator` object.
	device = accelerator.device
	model.to(device)

	model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
		model, optimizer, train_dataloader, valid_dataloader
	)
	# Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
	# shorter in multiprocess)

	if args.max_steps is None:
		args.max_steps = args.max_epoch * len(train_dataloader)
	else:
		args.max_epoch = args.max_steps // len(train_dataloader) + 1

	if args.valid_steps is None:
		args.valid_steps = len(train_dataloader)

	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
	)

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Max train steps = %d", args.max_steps)
	logger.info("  Total train batch size (w. distributed) = %s", args.batch_size * accelerator.num_processes)

	progress_bar = tqdm(range(args.max_steps), desc="Training", disable=not accelerator.is_local_main_process)

	global_step = 0
	stop_time = 0
	sent_loss = 0
	word_loss = 0

	if args.train_type in ['sent', 'joint']:
		sent_loss_fct = nn.MSELoss()

	if args.train_type in ['word', 'joint']:
		# Notice: we assign a weight of 5 for BAD words
		word_loss_fct = nn.CrossEntropyLoss(torch.Tensor([5, 1]).cuda())

	for epoch in range(args.max_epoch):
		for batch in train_dataloader:
			model.train()

			sent_outputs, word_outputs = model(**batch)
			if args.train_type in ['sent', 'joint']:
				sent_loss = sent_loss_fct(sent_outputs, batch["sentlab_lines"])
			if args.train_type in ['word', 'joint']:
				word_loss = word_loss_fct(word_outputs.view(-1, 2), batch["wordlab_lines"].view(-1))

			loss = sent_loss * 5 + word_loss

			accelerator.backward(loss)	
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()
			progress_bar.update(1)
			global_step += 1

			if global_step % args.valid_steps == 0:

				stop_time = evaluate(args=args,
									 model=model,
									 tokenizer=tokenizer,		 
									 valid_dataloader=valid_dataloader,
									 accelerator=accelerator,
									 global_step=global_step,
									 stop_time=stop_time)

			if args.save_steps is not None and global_step % args.save_steps == 0:
				# Save model checkpoint
				accelerator.wait_for_everyone()
				unwrapped_model = accelerator.unwrap_model(model)
				steps_output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
				unwrapped_model.save_pretrained(steps_output_dir, save_function=accelerator.save)
				if accelerator.is_main_process:
					tokenizer.save_pretrained(steps_output_dir)

			if global_step > args.max_steps:
				train_dataloader.close()
				valid_dataloader.close()
				break

			if args.stop_criterion is not None and stop_time >= args.stop_criterion:
				train_dataloader.close()
				valid_dataloader.close()
				break

def infer(args, model, tokenizer, infer_dataloader):

	if args.output_dir is None:
		args.output_dir = args.data_dir
	
	infer_dataloader = torch.utils.data.DataLoader(infer_dataset, shuffle=False, batch_size=args.batch_size)

	model.cuda() # 推断阶段我就懒得写分布式了，请使用单卡推断吧

	# Test!
	logger.info("***** Running infer *****")
	logger.info("  Num examples = %d", len(infer_dataloader) * args.batch_size)
	logger.info("  Batch size = %d", args.batch_size)

	all_sent_outputs = []
	all_word_outputs = []
	model.eval()
	for batch in infer_dataloader:

		with torch.no_grad():
			if has_additional_feature:
				sent_outputs, word_outputs = model(**batch)
			else:
				sent_outputs, word_outputs = model(**batch)

			word_preds = None
			sent_preds = None
			if args.infer_type in ['sent', 'joint']:
				all_sent_outputs.append(sent_outputs.detach().cpu().numpy().tolist())
			if args.infer_type in ['word', 'joint']:
				for i in range(len(batch[0])):
					active_word_positions = batch['subword_mask'][i] == 1
					active_word_outputs = word_outputs[i][active_word_positions]
					all_word_outputs.append(torch.argmax(active_word_outputs, axis=-1).detach().cpu().tolist())

	if args.infer_type in ['sent', 'joint']:
		with open(os.path.join(args.output_dir, args.infer_prefix+'.sent'), "w") as fout:
			for line in all_sent_outputs:
				fout.write(str(line)+'\n')
		logger.info("Sentence level predictions written to %s", os.path.join(args.output_dir, args.infer_prefix+'.sent'))

	if args.infer_type in ['word', 'joint']:
		with open(os.path.join(args.output_dir, args.infer_prefix+'.word'), "w") as fout:
			for line in all_word_outputs:
				line = ['OK' if tag == 1 else 'BAD' for tag in line]
				fout.write(' '.join(line)+'\n')
		logger.info("Word level predictions written to %s", os.path.join(args.output_dir, args.infer_prefix+'.word'))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--model_type", default='xlm_r', type=str, choices=('bert', 'xlm-mlm', 'xlm-tlm', 'xlm_r', 'distilbert'),
		help="Pretrained model you want to use. Choose from bert, xlm-mlm, xlm-tlm and xlm_r.")
	parser.add_argument("--model_path", default='xlm-roberta-base', type=str, help="Path to your downloaded pretrained model.")
	parser.add_argument("--data_dir", type=str, required=True, 
		help="The directory where the train, dev and infer data are stored.")
	parser.add_argument("--output_dir", type=str, required=True,
		help="The output directory where the model checkpoints will be written.")
	parser.add_argument("--max_length", default=None, type=int,
		help="The maximum total input sequence length after tokenization. Longer will be truncated.")
	parser.add_argument("--pad_to_max_length", action="store_true", help="Whether to pad to max_length when preprocessing.")

	parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
	parser.add_argument("--do_infer", action="store_true", help="Whether to run inference on the test set.")

	parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training and inference.")
	parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")

	parser.add_argument("--max_epoch", default=20, type=int, help="Total number of training epochs to perform.")
	parser.add_argument("--max_steps", default=None, type=int, 
		help="Total number of training steps to perform. Will be replaced by max_epoch if not specified.")
	parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
	parser.add_argument("--valid_steps", default=None, type=int, help="Valid and log every X updates steps.")
	parser.add_argument("--save_steps", default=None, type=int, help="Save checkpoint every X updates steps.")
	parser.add_argument("--stop_criterion", default=None, type=int, help="How many evaluation to stop if no imporvement.")

	parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
	parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")

	parser.add_argument("--seed", type=int, default=None, help="random seed for initialization")

	parser.add_argument("--train_prefix", type=str, default="train", help="The train prefix.")
	parser.add_argument("--valid_prefix", type=str, default="dev", help="The development prefix.")
	parser.add_argument("--infer_prefix", type=str, default="test", help="The inference prefix.")
	parser.add_argument("--suffix_a", type=str, default="src", help="The suffix for your first segment.")
	parser.add_argument("--suffix_b", type=str, default="mt", help="The suffix for your second segment.")
	parser.add_argument("--sentlab_suffix", type=str, default="hter", help="The suffix for sentence-level labels.")
	parser.add_argument("--wordlab_suffix", type=str, default="tags", help="The suffix for word-level labels.")

	parser.add_argument("--langtok_a", type=str, default=None, help="langtok for first segment in XLM.")
	parser.add_argument("--langtok_b", type=str, default=None, help="langtok for second segment in XLM.")
	parser.add_argument("--do_partial_prediction", action="store_true", 
		help="Whether to only use one side to do prediction. You need to store everything in segment_a if you do so.")
	parser.add_argument("--add_gap_to_target_text", action='store_true', 
		help="Whether to add gap to target sequence for gap tag prediction.")

	parser.add_argument("--train_type", type=str, default='sent', choices=('sent', 'word', 'joint'),
		help='Training type. Choose from sent, word and joint.')
	parser.add_argument("--valid_type", type=str, default='sent', choices=('sent', 'word', 'joint'),
		help='Evaluation type. Choose from sent, word and joint.')
	parser.add_argument("--infer_type", type=str, default='sent', choices=('sent', 'word', 'joint'),
		help='Inference type. Choose from sent, word and joint.')
	parser.add_argument("--best_metric", type=str, default=None, choices=('pearson', 'mcc', 'f1_multi'),
		help='Best metric to save your model.')

	parser.add_argument("--has_additional_feature", action="store_true")

	args = parser.parse_args()

	# Initialize the accelerator. We will let the accelerator handle device placement for us.
	kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
	accelerator = Accelerator(kwargs_handlers=kwargs_handlers)

	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)
	logger.info(accelerator.state)
	logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

	if args.seed is not None:
		transformers.set_seed(args.seed)

	args.model_type = args.model_type.lower()

	ModelDict = {'bert': BertPreTrainedModelForQE, 
				 'xlm-mlm': XLMPreTrainedModelForQE,
				 'xlm-tlm': XLMPreTrainedModelForQE,
				 'xlm_r': XLMRobertaPreTrainedModelForQE,
				 'distilbert': DistilBertPreTrainedModelForQE,
				 'mbart': MBartPreTrainedModelForQE,
				 'opus-mt': MarianPreTrainedModelForQE}

	PreTrainedModelForQE = ModelDict[args.model_type]

	if args.do_train:

		if (
			os.path.exists(args.output_dir)
			and os.listdir(args.output_dir)
			and not args.overwrite_output_dir
		):
			raise ValueError(
				"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
					args.output_dir
				)
			)

		if accelerator.is_main_process:
			if not os.path.exists(args.output_dir):
				os.makedirs(args.output_dir)
		accelerator.wait_for_everyone()

		if args.best_metric is None:
			if args.valid_type in ['sent', 'joint']:
				args.best_metric = 'pearson'
			else:
				args.best_metric = 'f1_multi'
		
		config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)

		if args.model_type in ["xlm-mlm", "xlm-tlm"]:
			args.lang2id = config.lang2id
			assert args.langtok_a in args.lang2id.keys()
			assert args.langtok_b in args.lang2id.keys()
		else:
			args.lang2id = None

		tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
		model = PreTrainedModelForQE.from_pretrained(
			args.model_path,
			config=config,
			args=args,
			local_files_only=True,
		)

		logger.info("Training/evaluation parameters %s", args)

		if args.add_gap_to_target_text:
			tokenizer.add_tokens('<gap>', special_tokens=True)
			model.resize_token_embeddings(len(tokenizer))

		train_dataset = read_and_process_examples(args, tokenizer, evaluate=False)
		valid_dataset = read_and_process_examples(args, tokenizer, evaluate=True)

		if args.pad_to_max_length and args.max_length is not None:
			# If padding was already done ot max length, we use the default data collator that will just convert everything
			# to tensors.
			data_collator = default_data_collator
		else:
			# Otherwise, `DataCollatorForQE` will apply dynamic padding for us (by padding to the maximum length of
			# the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
			# of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
			data_collator = DataCollatorForQE(tokenizer, pad_to_multiple_of=8 if accelerator.use_fp16 else None)

		train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
		valid_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, batch_size=args.batch_size)

		train(args, model, tokenizer, train_dataloader, valid_dataloader, accelerator)

	if args.do_infer:

		logger.info("Infering the following checkpoint: %s", args.model_path)

		config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)

		if args.model_type in ["xlm-mlm", "xlm-tlm"]:
			args.lang2id = config.lang2id
			assert args.langtok_a in args.lang2id.keys()
			assert args.langtok_b in args.lang2id.keys()

		tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
		model = PreTrainedModelForQE.from_pretrained(
			args.model_path,
			config=config,
			args=args,
			local_files_only=True,
		)
		infer_dataset = load_infer_examples(args, tokenizer)

		if args.pad_to_max_length:
			# If padding was already done ot max length, we use the default data collator that will just convert everything
			# to tensors.
			data_collator = default_data_collator
		else:
			# Otherwise, `DataCollatorForQE` will apply dynamic padding for us (by padding to the maximum length of
			# the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
			# of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
			data_collator = DataCollatorForQE(tokenizer, pad_to_multiple_of=8 if accelerator.use_fp16 else None)

		infer_dataloader = DataLoader(infer_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)

		if args.output_dir is None:
			args.output_dir = args.data_dir

		infer(args, model, tokenizer, infer_dataset)
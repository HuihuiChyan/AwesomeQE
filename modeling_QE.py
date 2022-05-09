import torch
from torch import nn
from transformers import (
	BertModel,
	XLMModel,
	XLMRobertaModel,
	DistilBertModel,
	MBartModel,
	MarianModel,
	BertPreTrainedModel,
	RobertaPreTrainedModel,
	DistilBertPreTrainedModel,
	MBartPreTrainedModel,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.models.marian.modeling_marian import MarianPreTrainedModel
import pdb

class QEBaseClass(object):
	# 所有的子类均应该继承这个类，而且是通过多继承的方式，并且将这个类放置在第一个继承的位置，从而获得forward方法

	def forward(
		self,
		input_ids,	
		attention_mask=None,
		token_type_ids=None,
		decoder_input_ids=None,
		decoder_attention_mask=None,
		additional_feature=None,
		**kwargs,
	):
		if self.model_type in ['xlm-tlm', 'xlm-mlm']:
			outputs = self.pretrained_model(
				input_ids,
				langs=token_type_ids,
				attention_mask=attention_mask,
			)
			sequence_outputs = self.dropout(outputs[0])
			pooled_outputs = sequence_outputs[:, 0, :]
		elif self.model_type == 'bert':
			outputs = self.bert(
				input_ids,
				token_type_ids=token_type_ids,
				attention_mask=attention_mask,
			)
			sequence_outputs = self.dropout(outputs[0])
			pooled_outputs = sequence_outputs[:, 0, :]
		elif self.model_type == 'xlmr':
			outputs = self.roberta(
				input_ids,
				attention_mask=attention_mask,
			)
			sequence_outputs = self.dropout(outputs[0])
			pooled_outputs = sequence_outputs[:, 0, :]
		elif self.model_type == 'distilbert':
			outputs = self.distilbert(
				input_ids,
				attention_mask=attention_mask,
			)
			sequence_outputs = self.dropout(outputs[0])
			pooled_outputs = sequence_outputs[:, 0, :]
		elif self.model_type in ['mbart', 'opus-mt']:

			eos_mask = decoder_input_ids.eq(self.config.eos_token_id)
			eos_indices = eos_mask.nonzero(as_tuple=True)[1]

			decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)

			outputs = self.model(
				input_ids,
				attention_mask=attention_mask,
				decoder_input_ids=decoder_input_ids,
				decoder_attention_mask=decoder_attention_mask,
			)
			sequence_outputs = self.dropout(outputs[0])

			pooled_outputs = sequence_outputs[eos_mask, :].view(sequence_outputs.size(0), -1, sequence_outputs.size(-1))[
				:, -1, :
			]

		else:
			raise Exception('Please check your model_type!')

		if additional_feature is not None:
			pooled_outputs = torch.cat([pooled_outputs, additional_feature], dim=-1)

		sent_outputs = self.sent_classifier(pooled_outputs).contiguous().view(-1)
		word_outputs = self.word_classifier(sequence_outputs)

		return sent_outputs, word_outputs

class BertPreTrainedModelForQE(QEBaseClass, BertPreTrainedModel):
	def __init__(self, config, args):
		super().__init__(config)
		# config.num_labels = 1 # 在config里默认是2，但是我们这里需要设置成1
		self.model_type = args.model_type
		self.bert = BertModel(config=config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

		if args.has_additional_feature:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size + args.additional_feature_size, 1)
		else:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size, 1)

		self.init_weights()

class XLMRobertaPreTrainedModelForQE(QEBaseClass, RobertaPreTrainedModel):
	def __init__(self, config, args):
		super().__init__(config)
		# config.num_labels = 1 # 在config里默认是2，但是我们这里需要设置成1
		self.model_type = args.model_type
		self.roberta = XLMRobertaModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

		if args.has_additional_feature:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size + args.additional_feature_size, 1)
		else:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size, 1)

		self.init_weights()

class XLMPreTrainedModelForQE(QEBaseClass, BertPreTrainedModel):
	def __init__(self, config, args):
		super().__init__(config)
		config.num_labels = 1 # 在config里默认是2，但是我们这里需要设置成1
		self.pretrained_model= XLMModel(config)
		self.dropout = nn.Dropout(config.dropout) #参考huggingface的BertForSequenceClassification
		self.use_bigru = args.use_bigru
		self.use_sigmoid = args.use_sigmoid
		self.bad_weight = args.bad_weight
		self.model_type = args.model_type
		self.bigru_dropout = self.dropout

		if args.has_additional_feature:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size + args.additional_feature_size, 1)
		else:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size, 1)

		self.init_weights()

class DistilBertPreTrainedModelForQE(QEBaseClass, DistilBertPreTrainedModel):
	def __init__(self, config, args):
		super().__init__(config)
		# config.num_labels = 1 # 在config里默认是2，但是我们这里需要设置成1
		self.model_type = args.model_type
		self.distilbert = DistilBertModel(config)
		self.dropout = nn.Dropout(config.dropout)

		if args.has_additional_feature:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size + args.additional_feature_size, 1)
		else:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size, 1)

		self.init_weights()

class MBartPreTrainedModelForQE(QEBaseClass, MBartPreTrainedModel):
	def __init__(self, config, args):
		super().__init__(config)
		# config.num_labels = 1 # 在config里默认是2，但是我们这里需要设置成1
		self.model_type = args.model_type
		self.model = MBartModel(config=config)
		self.dropout = nn.Dropout(config.dropout)

		if args.has_additional_feature:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size + args.additional_feature_size, 1)
		else:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size, 1)

		self.init_weights()

class MarianPreTrainedModelForQE(QEBaseClass, MarianPreTrainedModel):
	def __init__(self, config, args):
		super().__init__(config)
		# config.num_labels = 1 # 在config里默认是2，但是我们这里需要设置成1
		self.model_type = args.model_type
		self.model = MarianModel(config=config)
		self.dropout = nn.Dropout(config.dropout)

		if args.has_additional_feature:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size + args.additional_feature_size, 1)
		else:
			self.word_classifier = nn.Linear(config.hidden_size, 2)
			self.sent_classifier = nn.Linear(config.hidden_size, 1)

		self.init_weights()
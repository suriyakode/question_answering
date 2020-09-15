from transformers import AdamW, DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import *
import numpy
import json
import time
import os
import sys
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
import gzip
import multiprocessing
import pickle

from torch.nn import CrossEntropyLoss, Linear, Dropout


# max sequence length of input to model
SEQUENCE_LENGTH = 512

'''_____________________________________________________________________________________________

Class qa :

Modified version of source code for DistilBertForQuestionAnswering, which in turn is
built from DistilBertPreTrainedModel. DistilBertForQuestionAnswering added a linear layer,
qt_outputs to represent the 2 span predictions. qa also includes a qa_outputs linear layer,
whose 4 outputs represents the 4 question types. This layer is conncted to the penultimate
hidden layer, to not influence the qa_outputs, which are performed on the last hidden layer
outputs. Fianlly, the qa_outputs values are all added to the overall loss calculation.
 _____________________________________________________________________________________________
'''

class qa(DistilBertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.distilbert = DistilBertModel(config)
		self.qa_outputs = Linear(config.dim, config.num_labels)
		assert config.num_labels == 2

		self.dropout = Dropout(config.qa_dropout)

		# add linear layer mapping to 4 values
		self.qt_outputs = Linear(config.dim * SEQUENCE_LENGTH, 4)

		self.init_weights()

	def forward(self, input_ids=None, attention_mask=None,
					head_mask=None, inputs_embeds=None, start_positions=None,
					end_positions=None, question_types=None, output_attentions=None, 
					output_hidden_states=None, return_dict=None) :
		
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		distilbert_output = self.distilbert(
			input_ids=input_ids,
			attention_mask=attention_mask,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=True,
			return_dict=return_dict,
		)
		hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)


		# get the second to last hidden  layer
		second_hidden = distilbert_output[1][-2]  # (bs, max_query_len, dim)
		batch_size = second_hidden.shape[0]
		qt_layer  = second_hidden.shape[1] * second_hidden.shape[2]

		hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
		second_hidden = self.dropout(second_hidden)

		logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
		end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)

		# add processing for question type from penultimate hidden later
		q_type = self.qt_outputs(second_hidden.view(batch_size, qt_layer))


		total_loss = None
		if start_positions is not None and end_positions is not None and question_types is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			if len(question_types.size()) > 1:
				question_types = question_types.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			ignored_index = start_logits.size(1)
			start_positions.clamp_(0, ignored_index)
			end_positions.clamp_(0, ignored_index)
			question_types.clamp_(0, ignored_index)


			loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
			start_loss = loss_fct(start_logits, start_positions)
			end_loss = loss_fct(end_logits, end_positions)

			# factor in question type loss
			q_type_loss = loss_fct(q_type, question_types)
			total_loss = (start_loss + end_loss + q_type_loss) / 3

		if not return_dict:
			output = (start_logits, end_logits, q_type) + distilbert_output[2:]
			return ((total_loss,) + output) if total_loss is not None else output

		return QuestionAnsweringModelOutput(
			loss=total_loss,
			start_logits=start_logits,
			end_logits=end_logits,
			q_type=q_type,
			hidden_states=distilbert_output.hidden_states,
			attentions=distilbert_output.attentions,
		)

'''_____________________________________________________________________________________________

Gets rid of "<>" tokens to clean the example. Returns the first cleaned fragment of the context.
Args:
	context_token - List of words in the entire context
Returns:
	token_positions - Maps where the tokens in a clean context originated from in unprocessed context
	clean_context  - String of the cleaned context fragment 
 _____________________________________________________________________________________________
'''

def clean_example(context_tokens):

	token_positions = []
	tokens = []

	for i in range(len(context_tokens)):
		t = context_tokens[i]

		# only keep tokens that don't correspond to <> tags
		if not t.startswith("<"):
			token_positions.append(i)
			tokens.append(t)
		if i >= SEQUENCE_LENGTH:
			break

	return (token_positions, " ".join(tokens))

'''_____________________________________________________________________________________________

Returns the span indices into the clean context represented by token_positions, from the start and end 
indices listed for the original unprocessed context. 
Args:
	token_positions - Maps where the tokens in a clean context originated from in unprocessed context
	start - Start index in original unprocessed context
	end - End index in original unprocessed context
Returns:
	clean_start - Start index in input clean context
	clean_end - End index in input clean context
 _____________________________________________________________________________________________
'''
def get_clean_span(token_positions, start, end):

	seen_tokens = 0
	clean_start = -1
	clean_end = -1
	sflag = True

	# loop through indices to find first match
	for i in range(len(token_positions)):
		if token_positions[i] > start and sflag:
			clean_start = i  - 1
			sflag  = False
		if token_positions[i] > end:
			clean_end = i - 1
			break
		if i > SEQUENCE_LENGTH:
			return -1, -1

	return clean_start, clean_end


'''_____________________________________________________________________________________________

Tokenizes the input question and context. Returns tokenized values as well as offsets into
inputted context
Args:
	tokenizer - DistilForQuestionAnsweringTokenizer
	question - Question text
	clean_context - Cleaned context text
Returns:
	tokenized_indices - Maps to index where the tokens originiated from in the original clean context
	inputs - Tokenized values of question and context to eventually be fed to model
 _____________________________________________________________________________________________
'''

def get_tokenized_text(tokenizer, question, clean_context):

	inputs = {"input_ids": [], "attention_mask": []}
	tokenized_indices = []

	# add [CLS] token
	t = tokenizer("[CLS]", add_special_tokens=False)

	inputs['input_ids'].extend(t['input_ids'])
	inputs['attention_mask'].extend(t['attention_mask'])
	
	# tokenize the question
	for i, word in enumerate(question.split(" ")):
		t = tokenizer(word, add_special_tokens=False)

		tokenized_indices.append(len(inputs['input_ids']))

		inputs['input_ids'].extend(t['input_ids'])
		inputs['attention_mask'].extend(t['attention_mask'])

	# add [SEP] token
	t = tokenizer("[SEP]", add_special_tokens=False)
	inputs['input_ids'].extend(t['input_ids'])
	inputs['attention_mask'].extend(t['attention_mask'])

	# tokenize the context
	for i, word in enumerate(clean_context.split(" ")):
		t = tokenizer(word, add_special_tokens=False)

		tokenized_indices.append(len(inputs['input_ids']))

		inputs['input_ids'].extend(t['input_ids'])
		inputs['attention_mask'].extend(t['attention_mask'])

		# stop if we exceed the max SEQUENCE_LENGTH
		if len(inputs['input_ids']) >= SEQUENCE_LENGTH - 1:
			inputs['input_ids'] = inputs['input_ids'][:SEQUENCE_LENGTH - 1]
			inputs['attention_mask'] = inputs['attention_mask'][:SEQUENCE_LENGTH - 1]
			break

	# add [SEP] token
	t = tokenizer("[SEP]", add_special_tokens=False)
	inputs['input_ids'].extend(t['input_ids'])
	inputs['attention_mask'].extend(t['attention_mask'])

	# pad to size of sequence lenfth
	inputs['input_ids'].extend([0] * (SEQUENCE_LENGTH - len(inputs['input_ids'])))
	inputs['attention_mask'].extend([0] * (SEQUENCE_LENGTH - len(inputs['attention_mask'])))

	return tokenized_indices, inputs

'''_____________________________________________________________________________________________

Thread based function to load and tokenize the input training data.
Args:
	i - Thread index
Returns:
	all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions
	These are the corresponding lists of training example data parsed by the thread.
 _____________________________________________________________________________________________
'''
def load_train(i):

	all_input_ids = []
	all_input_mask = []
	all_start_positions = []
	all_end_positions = []
	all_question_types = []

	tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

	index = 0

	# find simplified input training file
	for filename in os.listdir("./v1.0/simple_train/"):

			with gzip.open("./v1.0/simple_train/" + filename, "rb") as train_file:	
				for line in train_file:
					# thread parallelism indexing
					index += 1
					if (index % 10 != i):
						continue

					# load and tokenize the input data
					example = json.loads(line)
					question = example['question_text']
					context = example['document_text']
					qlength = len(question.split(" "))

					context_tokens = context.split(" ")
					token_positions, clean_context = clean_example(context_tokens)
					tokenized_indices, inputs = get_tokenized_text(tokenizer, question,  clean_context)

					# loop through annotationns
					for gold in example['annotations']:

						bad_example = False

						if gold['yes_no_answer'] == "YES":
							clean_start, clean_end = get_clean_span(token_positions, 
															gold['long_answer']['start_token'], gold['long_answer']['end_token'])
							
							if clean_start < 0 or clean_start >= len(tokenized_indices) - qlength - 1 or clean_end < 0  or clean_end >= len(tokenized_indices) - qlength - 1:
								bad_example = True
								continue
							
							start_index = tokenized_indices[clean_start + qlength + 1]
							end_index = tokenized_indices[clean_end + qlength + 1]
							all_input_ids.append(inputs['input_ids'])
							all_input_mask.append(inputs['attention_mask'])
							
							# represents YES
							all_question_types.append([0])
							all_start_positions.append([clean_start])
							all_end_positions.append([clean_end])


						elif gold['yes_no_answer'] == "NO":
							clean_start, clean_end = get_clean_span(token_positions, 
															gold['long_answer']['start_token'], gold['long_answer']['end_token'])
							if clean_start < 0 or clean_start >= len(tokenized_indices) - qlength - 1 or clean_end < 0  or clean_end >= len(tokenized_indices) - qlength - 1:
								bad_example = True
								continue
							
							start_index = tokenized_indices[clean_start + qlength + 1]
							end_index = tokenized_indices[clean_end + qlength + 1]
							all_input_ids.append(inputs['input_ids'])
							all_input_mask.append(inputs['attention_mask'])
							
							# represents NO
							all_question_types.append([1])
							all_start_positions.append([clean_start])
							all_end_positions.append([clean_end])


						# load short answers, if availale
						elif len(gold['short_answers']) > 0:
							# start of first short answer
							clean_start, _ = get_clean_span(token_positions, 
															gold['short_answers'][0]['start_token'], gold['short_answers'][0]['end_token'])
							
							# end of last short answer
							_, clean_end = get_clean_span(token_positions, 
															gold['short_answers'][-1]['start_token'], gold['short_answers'][-1]['end_token'])
							
							if clean_start < 0 or clean_start >= len(tokenized_indices) - qlength or clean_end < 0  or clean_end >= len(tokenized_indices) - qlength:
								bad_example = True
								continue

							start_index = tokenized_indices[clean_start + qlength] 
							end_index = tokenized_indices[clean_end + qlength] 

							all_input_ids.append(inputs['input_ids'])
							all_input_mask.append(inputs['attention_mask'])
							
							# represents SHORT ANSWER
							all_question_types.append([2])
							all_start_positions.append([start_index])
							all_end_positions.append([end_index])

						# no long answer
						elif gold['long_answer']['start_token'] == -1:
							bad_example = True

						# load long answer
						else:
							clean_start, clean_end = get_clean_span(token_positions, 
															gold['long_answer']['start_token'], gold['long_answer']['end_token'])
							if clean_start < 0 or clean_start >= len(tokenized_indices) - qlength - 1 or clean_end < 0  or clean_end >= len(tokenized_indices) - qlength - 1:
								bad_example = True
								continue
							
							start_index = tokenized_indices[clean_start + qlength + 1]
							end_index = tokenized_indices[clean_end + qlength + 1]
							
							all_input_ids.append(inputs['input_ids'])
							all_input_mask.append(inputs['attention_mask'])
							
							# represents LONG ANSWER
							all_question_types.append([3])
							all_start_positions.append([start_index])
							all_end_positions.append([end_index])

						# could do training on "NONE" type or otherwise invalid answers here
						if bad_example:
							continue

	return all_input_ids, all_input_mask, all_question_types, all_start_positions, all_end_positions


'''_____________________________________________________________________________________________

	Load and train examples from the Natural Question dataset on a pre-trained distilbert model for 
	question answering.  
	Saves the resulting fine tuned model to the output directory given through CLI args. 
 _____________________________________________________________________________________________
'''
def train():

	os.makedirs(sys.argv[2])

	# load pre-trained tokenizer and  model
	tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
	model = qa.from_pretrained('distilbert-base-uncased')
	
	# set weight decay for optimizer
	no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
		#{'params': [p for n, p in answer_type_fn.named_parameters()], 'weight_decay': 0.01}
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)


	all_input_ids = []
	all_input_mask = []
	all_question_types = []
	all_start_positions = []
	all_end_positions = []


	print ("BEGIN LOADING\n")
	start_time = time.time()

	# if the dump file exists, load all the processed training data from there
	if os.path.exists(sys.argv[3]):
		load_cache = pickle.load(open(sys.argv[3], "rb"))
		all_input_ids =  load_cache["ii"]
		all_input_mask  =  load_cache["im"]
		all_question_types = load_cache["qt"]
		all_start_positions  =  load_cache["sp"]
		all_end_positions  =  load_cache["ep"]


	# load the training data from the simplified examples gzip in parallel
	else:
		# pool of 10 threads to run load_train() on chunks of input file
		pool = multiprocessing.Pool(10)
		try:
			out = pool.map(load_train, range(10))

		finally:
			pool.close()
			pool.join()

		# combine outputs from all 10 threads
		for ii, im, qt, sp, ep in out:
			all_input_ids.extend(ii)
			all_input_mask.extend(im)
			all_question_types.extend(qt)
			all_start_positions.extend(sp)
			all_end_positions.extend(ep)

		# cache for next time :)
		dump_cache = {
			"ii" : all_input_ids,
			"im" : all_input_mask,
			"qt" : all_question_types,
			"sp" : all_start_positions, 
			"ep" : all_end_positions
		}
		pickle.dump(dump_cache, open(sys.argv[3], "wb"))


	# convert data from List() to Tensor() and prepare the dataloader for sampling
	all_input_ids = torch.LongTensor(all_input_ids)
	all_input_mask = torch.LongTensor(all_input_mask)
	all_question_types = torch.LongTensor(all_question_types)
	all_start_positions = torch.LongTensor(all_start_positions)
	all_end_positions = torch.LongTensor(all_end_positions)

	train_batch_size = 32	
	train_data = TensorDataset(all_input_ids, all_input_mask, all_question_types,
									   all_start_positions, all_end_positions)

	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

	print (len(all_input_ids))

	print ("LOAD TIME: ")
	print (time.time() - start_time)
	print ("\n")
	print ("BEGIN TRAINING\n")
	start_time = time.time()

	# map model to GPU and prepare for training
	model.train()
	model.to("cuda")

	print (torch.cuda.memory_summary())

	num_train_epochs = 2
	for epoch in range(num_train_epochs):
		for step, batch in enumerate(train_dataloader):
			#  map tensors in batch to GPU
			batch = tuple(t.to("cuda") for t in batch) 
			input_ids, input_mask, question_type, start_positions, end_positions = batch	

			# training and model update
			outputs = model(input_ids=input_ids, attention_mask=input_mask, question_types=question_type, start_positions=start_positions, end_positions=end_positions, output_hidden_states=True)
			loss = outputs[0]
			
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()


			if (step % 50 == 0):
				print ("\nSTEP: ")
				print (step)
				print ("\nLOSS: ")
				print (loss)

	print ("TRAIN TIME: ")
	print (time.time() - start_time)
	print ("\n")

	# save fine tuned model and tokenizer for later use with eval()
	model.save_pretrained(sys.argv[2])
	tokenizer.save_pretrained(sys.argv[2])


'''_____________________________________________________________________________________________

Gets rid of "<>" tokens to clean the example. Returns all cleaned fragments of the context
Args:
	context_token - List of words in the entire context
Returns:
	ret_token_positions, ret_contexts, offsets - Parallel lists. Values represent the fragmented context
	and where the original token positions were in the uncleaned context 
 _____________________________________________________________________________________________
'''

def clean_whole_example(context_tokens):

	token_positions = []
	tokens = []
	ret_token_positions = []
	ret_contexts = []
	offsets = [0]

	for i in range(len(context_tokens)):
		t = context_tokens[i]

		# only keep tokens that don't correspond to <> tags
		if not t.startswith("<"):
			token_positions.append(i)
			tokens.append(t)

		# start appending to a new clean_context if we've reached the max input length
		if len(token_positions) == SEQUENCE_LENGTH:
			ret_token_positions.append(token_positions)
			ret_contexts.append(" ".join(tokens))
			tokens = []
			token_positions = []
			offsets.append(i)

	ret_token_positions.append(token_positions)
	ret_contexts.append(" ".join(tokens))

	return (ret_token_positions, ret_contexts, offsets)

'''_____________________________________________________________________________________________

Gets the n best argmax indices of a logits list
Args:
	logits - Input to be argmaxed
	n_best_size - Number of best indices to return
Returns:
	List of best indices
 _____________________________________________________________________________________________
'''

def get_best_indexes(logits, n_best_size):

	# rank logits
	index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
	best_indexes = []

	for i in range(len(index_and_score)):
		if i >= n_best_size:
			break
		best_indexes.append(index_and_score[i][0])

	return best_indexes

'''_____________________________________________________________________________________________

Returns the span prediction from a set of start and end logits, as well as the question type
Args:
	start_scores - Start of span logits
	end_scores -  End of span logits
	question_types - Question type logits
Returns:
	Best span prediction + question type for the given input set of logits
 _____________________________________________________________________________________________
'''

def get_prediction_span(start_scores, end_scores, question_types):

	# get best start and end predictions
	# could just use argmax, as we've decided on just 1 index
	start_indexes = get_best_indexes(start_scores, 1)
	end_indexes = get_best_indexes(end_scores, 1)

	# null prediction
	largest_score = -100000
	answer_start =  len(start_scores) - 2
	answer_end = len(end_scores) - 1

	for si in start_indexes:
		for ei in end_indexes:
			# invalid span
			if ei <= si:
				continue
			# update if better span is found
			if start_scores[si] + end_scores[ei] > largest_score:
				largest_score = start_scores[si] + end_scores[ei]
				answer_start = si
				answer_end = ei

	return answer_start, answer_end, torch.argmax(question_types, dim=0)

'''_____________________________________________________________________________________________

Called immeadiately after receiving the output from the model to convert the logits 
into a scored span prediction in the original, unprocessed context.

Args:
	tokenizer - DistilForQuestionAnsweringTokenizer
	start_scores - Start score logits outputted by model
	end_scores -  End score logits outputted by model
	question_types - Question type logits outputted by model
	uid - Example's unique id 
	tokenized_samples - Data store to represent the fragmented context in the original text
	offset_index - fragment index for the example context

Returns:
	predicted_label - Holds the scores and span predictions for the given example 
	
 _____________________________________________________________________________________________
'''

def record_prediction(tokenizer, start_scores, end_scores, question_types, uid, tokenized_samples, offset_index):

	# get the best start index, end indec and question type from the score arrays
	start, end, question_type = get_prediction_span(start_scores, end_scores, question_types)

	# unpack arguments
	qtype = question_type.item()
	uid = uid.item()
	offset_index = offset_index.item()

	question = tokenized_samples[uid][offset_index]['question']
	clean_context = tokenized_samples[uid][offset_index]['clean_context']
	context = tokenized_samples[uid][offset_index]['context']
	tokenized_indices = tokenized_samples[uid][offset_index]['tokenized_indices']
	token_positions = tokenized_samples[uid][offset_index]['token_positions']
	candidates = tokenized_samples[uid][offset_index]['candidates']
	offset = tokenized_samples[uid][offset_index]['context_offset']
	
	q_length = len(question.split(" "))


	# Default empty prediction.
	score = torch.Tensor([-10000]) 
	short_span = (-1, -1)
	long_span = (-1, -1)
	y_n = "NONE"
	long_answer = {
		"start_token": -1,
		"end_token": -1,
		"start_byte": -1,
		"end_byte": -1
	}
	short_answer = []

	# find where span is in the clean context
	sflag = True
	clean_start = -1
	clean_end = -1
	for i in range(len(tokenized_indices)):
		if tokenized_indices[i] >= start and sflag:
			clean_start = i
			sflag = False
		if tokenized_indices[i] >= end:
			clean_end = i
			break

	# if the answer does not contain the question
	if clean_start >=  q_length and clean_end >= q_length:
		# find where the span is in the original, unfragmented context
		predict_start = token_positions[clean_start - q_length]
		predict_end = token_positions[clean_end - q_length]
		if predict_start == predict_end:
			predict_end += 1

		# score the prediction span
		short_span_score = start_scores[start] + end_scores[end]
		cls_token_score = start_scores[0] + end_scores[0]

		# span logits minus the cls logits seems to be close to the best.
		score = short_span_score - cls_token_score
		short_span = (predict_start, predict_end)

		# find enclosing long candidate for the predicted answer
		for c in candidates:
			if c["start_token"] <= predict_start + 1 and c["end_token"] >= predict_end - 1:
				long_span = (c["start_token"], c["end_token"])
				if c["top_level"]:
					break
		

		long_answer = {
			"start_token": long_span[0],
			"end_token": long_span[1],
			"start_byte": -1,
			"end_byte": -1
		}

		# only add a short answer if long answer is valid
		if predict_start < predict_end and long_span[0] > 0 and long_span[1] > 0:
			short_answer = [{
				"start_token": short_span[0],
				"end_token": short_span[1],
				"start_byte": -1,
				"end_byte": -1
			}]


		# this answer isn't particularly short - leave it from the long answer
		if short_span[0] < short_span[1] - 12:
			short_answer = []

		# score and answer correction for Y_N answers
		if qtype == 0:
			y_n = "YES"
			short_answer = []
			score += 2 * cls_token_score

		elif qtype == 1:
			y_n = "NO"
			short_answer = []
			score += 2 * cls_token_score


	# return the predicted label
	predicted_label = {
		"example_id": uid,
		"long_answer": long_answer,
		"long_answer_score": score.item(),
		"short_answers": short_answer,
		"short_answers_score": score.item(),
		"yes_no_answer": y_n
	}


	'''
	if y_n == "NONE":
		text = " ".join(context.split(" ")[long_span[0]:long_span[1]])
	if qtype < 2:
		text = y_n
	print (question + " : ")
	print (text)
	print (" ".join(context.split(" ")[short_span[0]:short_span[1]]))
	print (qtype)
	print (" ".join(clean_context.split(" ")[clean_start - q_length : clean_end - q_length]))
	print ("\n")
	print (predicted_label)
	print ("\n")
	'''
	
	return predicted_label


'''_____________________________________________________________________________________________

Thread based function to load and tokenize the input eval data.
Fragments contexts of size > SEQUENCE_LENGTH so that the question will be posed to each context.
Only loads the first 5 fragments of the context. If the answer is after this, it will not be found.

Args:
	i - Thread index
Returns:
	all_input_ids, all_input_mask - tokenized texts to be fed to model
	all_sample_ids - UIDs of the samples
	tokenized_samples - Data store to represent the fragmented context in the original text
	offset_index - context fragment index
	
 _____________________________________________________________________________________________
'''
def load_eval(i):
	all_input_ids = []
	all_input_mask = []
	all_sample_ids = []
	offset_index = []
	tokenized_samples = {}

	tokenizer = DistilBertTokenizer.from_pretrained(sys.argv[2])

	index = 0

	# load the simplified eval data
	for filename in os.listdir("./v1.0/dev/"):
		if "simplified" in filename:
			with gzip.open("./v1.0/dev/" + filename, "rb") as train_file:	
				for line in train_file:
					# thread parallelism indexing
					index += 1
					if (index % 10 != i):
						continue

					# load the eval data
					example = json.loads(line)
					question = example['question_text']
					context = example['document_text']
					uid = example['example_id']
					tokenized_samples[uid] = []

					# slices input context into multiple chunks of size SEQUENCE LENGTH
					context_tokens = context.split(" ")
					all_token_positions, all_clean_contexts, offsets = clean_whole_example(context_tokens)

					# each chunk is fed to the model
					for fragment in range(len(all_clean_contexts)):
						token_positions = all_token_positions[fragment]
						clean_context = all_clean_contexts[fragment]

						# tokenize text to be input to model
						tokenized_indices, inputs = get_tokenized_text(tokenizer, question, clean_context)

						all_input_ids.append(inputs['input_ids'])
						all_input_mask.append(inputs['attention_mask'])
						all_sample_ids.append([uid])
						offset_index.append([fragment])

						tokenized_samples[uid].append({
							'question' : question,
							'context' : context,
							'clean_context' : clean_context,
							'tokenized_indices': tokenized_indices,
							'token_positions' : token_positions,
							'candidates' : example['long_answer_candidates'],
							'context_offset' : offsets[fragment]
						})
						# only get first 5 fragments in context
						if fragment >= 0:
							break

	return all_input_ids, all_input_mask, all_sample_ids, tokenized_samples, offset_index



'''_____________________________________________________________________________________________

Load and evaluate examples on the fine tuned queestion answering model. 
Writes predictions to output file given in CLI args. 
 _____________________________________________________________________________________________
'''
def evaluate():

	# load finetuned model
	tokenizer = DistilBertTokenizer.from_pretrained(sys.argv[2])
	model = qa.from_pretrained(sys.argv[2])

	all_input_ids = []
	all_input_mask = []
	all_sample_ids = []
	tokenized_samples = {}
	offset_index = []

	print ("BEGIN LOADING\n")
	start_time = time.time()

	# if the dump file exists, load all the processed training data from there
	if os.path.exists(sys.argv[3]):
		load_cache = pickle.load(open(sys.argv[3], "rb"))
		all_input_ids =  load_cache["ii"]
		all_input_mask  =  load_cache["im"]
		all_sample_ids = load_cache["si"]
		tokenized_samples = load_cache["ts"]
		offset_index = load_cache["oi"]
	
	# load the eval data from the example gzip files in parallel
	else:
		# pool of 5 threads to run load_eval() on different	
		pool = multiprocessing.Pool(10)

		try:
			out = pool.map(load_eval, range(10))
		finally:
			pool.close()
			pool.join()

		# combine outputs of each thread
		for ii, im, si, ts, oi in out:
			all_input_ids.extend(ii)
			all_input_mask.extend(im)
			all_sample_ids.extend(si)
			tokenized_samples.update(ts)
			offset_index.extend(oi)

		# cache for next time :)
		dump_cache = {
			"ii" : all_input_ids,
			"im" : all_input_mask,
			"si" : all_sample_ids,
			"ts" : tokenized_samples,
			"oi" : offset_index
		}
		pickle.dump(dump_cache, open(sys.argv[3], "wb"))


	# prepare tensor data
	all_input_ids = torch.LongTensor(all_input_ids)
	all_input_mask = torch.LongTensor(all_input_mask)
	all_sample_ids = torch.LongTensor(all_sample_ids)
	offset_index = torch.LongTensor(offset_index)
	print (len(all_sample_ids))

	eval_data = TensorDataset(all_input_ids, all_input_mask, all_sample_ids, offset_index)

	predict_batch_size = 32
	eval_sampler = SequentialSampler(eval_data)
	eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=predict_batch_size)

	model.to("cuda")
	model.eval()
	print ("LOAD TIME: ") 
	print (time.time() - start_time)
	print ("\n")

	predictions = {'predictions' : []}
	results = {}
	for step, batch in enumerate(eval_dataloader):
		batch = tuple(t.to("cuda") for t in batch) 
		input_ids, input_mask, sample_id, offset_index = batch	

		with torch.no_grad():
			# model returns the predicted span and the question type
			start_scores, end_scores, question_types = model(input_ids=input_ids, attention_mask=input_mask)
			for i in range(len(start_scores)):
				# record prediction scores and context token spans
				prediction = record_prediction(tokenizer, start_scores[i], end_scores[i], question_types[i], sample_id[i], tokenized_samples, offset_index[i])
				if prediction:
					if sample_id[i].item() in results:
						results[sample_id[i].item()].append(prediction)
					else:
						results[sample_id[i].item()] = [prediction]
				else:
					print ("NULL")


	# select the best scoring context fragment for each question
	for i in results:
		r = results[i]
		max_score = -100000
		final_prediction = r[0]
		for p in r:
			if p['short_answers_score'] > max_score and p['long_answer']['start_token'] >= 0:
				max_score = p['short_answers_score']
				final_prediction = p

		if final_prediction:
			if final_prediction['yes_no_answer'] != "NONE":
				# artifically boost the score for Y_N answers to be considered for F1
				final_prediction['short_answers_score'] += 25
				final_prediction['long_answer_score'] +=  25
			predictions['predictions'].append(final_prediction)

	with open("predictions.json",  "w") as out:
		json.dump(predictions, out)

	print (len(results))
	print (len(predictions['predictions']))

def main():
	if sys.argv[1] == "train":
		train()
	elif sys.argv[1] == "eval":
		evaluate()


if __name__ == "__main__":
	main()

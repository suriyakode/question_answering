# Model for question answering based on Google's natural questions dataset.

Usage: I ran this on a fresh Ubuntu16.04 instance with a P5000 GPU through paperspace.

1) clone the repo and make working_dir
2) install dependencies (off the top of my head)
	requires python3.7 and pip3.7: https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/
	cuda v10.2 : https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal
	gsutil : https://cloud.google.com/storage/docs/gsutil_install
	pip3.7 : torch v1.6.0, transformers, absl-py



3) download dataset and store samples in ./v1.0

	```
	gsutil -m cp -R gs://natural_questions/v1.0 <path to your data directory>
	```
4) simplify training and eval data
	
	```
	sudo python3.7 simplify_nq_data.py --data_dir=v1.0/train
	
	# simplifed train data should be in ./v1.0/simple_train 
	mkdir v1.0/simple_train
	cp v1.0-simplified_simplified-nq-train.jsonl.gz v1.0/simple_train

	sudo python3.7 simplify_nq_data.py --data_dir=v1.0/dev/
	# make sure simplified-nq-.jsonl.gz exists in directory v1.0/dev/ 
	```

5) perfrom training
	
	```
	 # if train_cache exists it will load from there, else it will save there
	 # will create <SAVE_MODEL_DIR> 
	 
	sudo python3.7 QA.py train <SAVE_MODEL_DIR> <train_cache_file>
	```
6) calculate predictions
	
	```
	# if eval_cache exists it will load from there, else it will save there
	# expects <LOAD_MODEL_DIR> to exist 
	
	sudo python3.7 QA.py eval <LOAD_MODEL_DIR> <eval_cache_file>
	```
7) score
	
	```
	sudo python3.7 nq_eval.py --gold_path=v1.0/dev/simplified-nq-.jsonl.gz --predictions_path=predictions.json
	```

Writeup :

Hello Cortx team! I wanted to start off by thanking you for this exciting project. It proved to be an incredile learning experience, and on top of that, a ton of fun! As a newcomer to ML libraries, I found myself engrossed in the code and spent a significant number of hours just poking around to try and obtain more satisfying results. Overall, I was very content spending time on this model and would be enthused working at Cortx on similar projects.

 I think the hardest part of the project for me was the data processing, as it took a lot of tests to properly work with the input contexts and also convert the model outputs back to the correct form. The processing took a lot of time for me too, so I even added features like thread parallelism and caching for loading data.

Design : 
I originally found the BertForQuestionAnswering model and decided to work with that. Unfortunately, I could never load a Bert model onto my GPU for training as it would immeadiately complain about memory usage. I found this odd, as the model shouldn't take up the 16GB of GPU space. Luckily I found the DistilBertForQuestionAnswering model, which boasted similar results to Bert while being light-weight. 

For my final model, I chose to modify the source code of DBFQA. I noticed that DBFQA was essentially a wrapper for DB with an extra linear layer from the last hidden outputs to identify the span start and end. I copied the source code and added another linear layer to identify the question type. I chose to get the input for the question type layer from the SECOND to last hidden layer, as I read this would interfere less with the span prediction. I also inluded the question type prediction in loss calculations. I chose only 4 possible question types: "YES, "NO", "SHORT ANSWER" and "LONG ANSWER".

One con of using this model, is that there is a maximum sequence length of 512. Thus, if the answer is more than 512 words into the context, we could not find it unless we were to fragment the entire context and ask questions on each fragment. 

For training I excluded any examples where the answer would not be found in the first 512 words on the text. I chose NOT to perform fragmentation of contexts during training time, giving me ~ 80000 valid training samples.

For evaluation, I originally tried evaluating on the first five fragments for every example (time constraint). If the answer was outside of that fragment range, it would not be found. For each fragment of a context, I posed the same question and discovered the answer and score within that fragment. Then, I obtained the overall outputs for each context by taking their best scoring fragment.

With all of this in place, I noticed that my model was getting very good at returning the correct answer, if one exists within the fragment. I performed a quick test myself to discover that if the answer is within the context, it is correctly identified 82% of the time. 

Unfortunately, my model does not have a good way of noting if a good answer was NOT found. For instance, particularly in cases where "who" or "when" questions are asked on a fragment NOT containing the answer, it may score higher than the correct answer fragment. As a result, when combining fragment outputs, even if one fragment was correct we may predict the wrong answer. As a result, my score decreased when evaluating over multiple fragments of a context instead of just the first.

To try and remedy this, I attempted to add a new 5th question type "NONE", to represent that a good answer is not present on the fragment text. In the end, I wasn't sure how to train this. It seemed counterintuitive to add wrong answers during training. However, without doing so, my evaluations never predicted that quetion type.

As a result, I ended up only using the first fragments for my final evaluation. As discussed before, if the answer is outside of the fragment range, it cannot be found, which automatically invalidates a lot of the examples. 

I believe that if I could do a little bit more work on the scoring, especially in the case of incorrect answers, I should be able to combine fragment outputs properly and increase my F1 score significantly. I am quite confident that the overall logic of the model is sound, as it consistently finds the correct answer if one is available. 
In addition, with better equipment and more time, I would like to experiment with more powerful models such as Bert or RoBerta. I would also be able to train/eval on all fragments of every context, which currently takes too much time. 

# Score
{"long-best-threshold-f1": 0.41912246234446626, "long-best-threshold-precision": 0.42160737812911725, "long-best-threshold-recall": 0.4166666666666667, "long-best-threshold": 17.67715072631836, "long-recall-at-precision>=0.5": 0.3077256944444444, "long-precision-at-precision>=0.5": 0.5001763668430335, "long-recall-at-precision>=0.75": 0, "long-precision-at-precision>=0.75": 0, "long-recall-at-precision>=0.9": 0, "long-precision-at-precision>=0.9": 0, "short-best-threshold-f1": 0.29562982005141386, "short-best-threshold-precision": 0.33236994219653176, "short-best-threshold-recall": 0.2662037037037037, "short-best-threshold": 20.64846420288086, "short-recall-at-precision>=0.5": 0.109375, "short-precision-at-precision>=0.5": 0.5006622516556292, "short-recall-at-precision>=0.75": 0, "short-precision-at-precision>=0.75": 0, "short-recall-at-precision>=0.9": 0, "short-precision-at-precision>=0.9": 0}
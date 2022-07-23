# This script is based on
# https://keras.io/examples/nlp/text_extraction_with_bert/

import argparse
import deepspeed
import os
import utility.data_processing as dpp
import utility.testing as testing
import torch
from datasets import load_dataset, load_metric
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from torch.nn import functional as F
from datetime import datetime
from datasets.utils import disable_progress_bar
from datasets import disable_caching


disable_progress_bar()
disable_caching()

# Benchmark settings
parser = argparse.ArgumentParser(description='BERT finetuning on SQuAD')
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--download-only', action='store_true',
                    help='Download model, tokenizer, etc and exit')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

hf_model = 'bert-base-uncased'
bert_cache = os.path.join(os.getcwd(), 'cache')

slow_tokenizer = BertTokenizer.from_pretrained(
    hf_model,
    cache_dir=os.path.join(bert_cache, f'_{hf_model}-tokenizer')
)
save_path = os.path.join(bert_cache, f'{hf_model}-tokenizer')
if not os.path.exists(save_path):
    os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer(os.path.join(save_path, 'vocab.txt'),
                                   lowercase=True)

model = BertForQuestionAnswering.from_pretrained(
    hf_model,
    cache_dir=os.path.join(bert_cache, f'{hf_model}_qa')
)

if args.download_only:
    exit()

model.train()

hf_dataset = load_dataset('squad')

max_len = 384

hf_dataset.flatten()
processed_dataset = hf_dataset.flatten().map(
    lambda example: dpp.process_squad_item_batched(example, max_len,
                                                   tokenizer),
    remove_columns=hf_dataset.flatten()['train'].column_names,
    batched=True,
    num_proc=12
)

train_set = processed_dataset["train"]
train_set.set_format(type='torch')

parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=parameters,
    training_data=train_set
)

rank = torch.distributed.get_rank()

# training
num_epochs = 1
for epoch in range(num_epochs):  # loop over the dataset multiple times
    for i, batch in enumerate(trainloader, 0):
        outputs = model(input_ids=batch['input_ids'].to(model_engine.device),
                        token_type_ids=batch['token_type_ids'].to(model_engine.device),
                        attention_mask=batch['attention_mask'].to(model_engine.device),
                        start_positions=batch['start_token_idx'].to(model_engine.device),
                        end_positions=batch['end_token_idx'].to(model_engine.device))
        # forward + backward + optimize
        loss = outputs[0]
        model_engine.backward(loss)
        model_engine.step()

if rank == 0:
    print('Finished Training')
    if os.environ['SLURM_NODEID'] == '0':
        model_hash = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        model_path_name = f'./cache/model_trained_deepspeed_{model_hash}'

        # save model's state_dict
        torch.save(model.state_dict(), model_path_name)

        # create the model again since the previous one is on the gpu
        model_cpu = BertForQuestionAnswering.from_pretrained(
            "bert-base-uncased",
            cache_dir=os.path.join(bert_cache, 'bert-base-uncased_qa')
        )

        # load the model on cpu
        model_cpu.load_state_dict(
            torch.load(model_path_name,
                       map_location=torch.device('cpu'))
        )

        # load the model on gpu
        # model.load_state_dict(torch.load(model_path_name))
        # model.eval()

        eval_set = processed_dataset["validation"]
        eval_set.set_format(type='torch')
        batch_size = 1

        eval_dataloader = DataLoader(
            eval_set,
            shuffle=False,
            batch_size=batch_size
        )

        squad_example_objects = []
        for item in hf_dataset['validation'].flatten():
            squad_examples = dpp.squad_examples_from_dataset(item, max_len,
                                                             tokenizer)
            try:
                squad_example_objects.extend(squad_examples)
            except TypeError:
                squad_example_objects.append(squad_examples)

        assert len(eval_set) == len(squad_example_objects)

        start_sample = 0
        num_test_samples = 10
        for i, eval_batch in enumerate(eval_dataloader):
            if i > start_sample:
                testing.EvalUtility(eval_batch, [squad_example_objects[i]],
                                    model_cpu).results()

            if i > start_sample + num_test_samples:
                break

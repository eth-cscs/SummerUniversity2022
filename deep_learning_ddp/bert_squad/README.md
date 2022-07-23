# Fine-tuning a BERT model for text extraction with the SQuAD dataset

We are going to fine-tune [BERT implemented by HuggingFace](https://huggingface.co/bert-base-uncased) for the text-extraction task with a dataset of questions and answers with the [SQuAD (The Stanford Question Answering Dataset)](https://rajpurkar.github.io/SQuAD-explorer/) dataset.
The data is composed by a set of questions and the corresponding paragraphs that contain the answers.
The model will be trained to locate the answer in the context by giving the positions where the answer starts and ends.

The content here is based on [BERT (from HuggingFace Transformers) for Text Extraction](https://keras.io/examples/nlp/text_extraction_with_bert/).

More info:
- [Glossary - HuggingFace docs](https://huggingface.co/transformers/glossary.html#model-inputs)
- [BERT NLP — How To Build a Question Answering Bot](https://towardsdatascience.com/bert-nlp-how-to-build-a-question-answering-bot-98b1d1594d7b)

## Deepspeed version
```bash
srun --pty --wait 60 python 3_squad_bert_deepspeed.py --deepspeed_config ds_config.json
```

The evaluations are done on a single node within the same script.
Since the rest of the ranks of the job has already finished when the evaluation starts, it might be necessary to add the option `--wait 60` to `srun` to prevent Slurm from canceling the job before the evaluation finishes.

# TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection

We put together a script, data, and trained models used in our [paper](https://arxiv.org/abs/1911.04118). In a nutshell, TANDA is a technique for fine-tuning pre-trained Transformer models sequentially in two steps:
* first, transfer a pre-trained model to a model for a general task by fine-tuning it on a large and high-quality dataset;
* then, perform a second fine-tuning step to adapt the transferred model to the target domain.

## Script

We base our implementation on the [transformers](https://github.com/huggingface/transformers) package. We use the following script to enable `sequential fine-tuning` option for the package.

```
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout f3386 -b tanda-sequential-finetuning
git apply tanda-sequential-finetuning-with-asnq.diff
```

* `f3386` is the latest commit as of `Sun Nov 17 18:08:51 2019 +0900`, and `tanda-sequential-finetuning-with-asnq.diff` is the diff to enable the option.

For example, to transfer with ASNQ and adapt with a target dataset:
* download [the ASNQ dataset](#answer-sentence-natural-questions-asnq) and the target dataset (e.g. Wiki-QA, formatted similar as ASNQ), and
* run the following script
 

```
python run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name ASNQ \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir [PATH-TO-ASNQ] \
    --per_gpu_train_batch_size 150 \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --output_dir [PATH-TO-TRANSFER-FOLDER]

python run_glue.py \
    --model_type bert \
    --model_name_or_path [PATH-TO-TRANSFER-FOLDER] \
    --task_name ASNQ \
    --do_train \
    --do_eval \
    --sequential \
    --do_lower_case \
    --data_dir [PATH-TO-WIKI-QA] \
    --per_gpu_train_batch_size 150 \
    --learning_rate 1e-6 \
    --num_train_epochs 2.0 \
    --output_dir [PATH-TO-OUTPUT-FOLDER]
```

## Data

We use the following datasets in the paper:

### Answer-Sentence Natural Questions (ASNQ)
* ASNQ is a dataset for answer sentence selection derived from Google Natural Questions (NQ) dataset (Kwiatkowski et al. 2019). The dataset details can be found in our paper.
* ASNQ is used to transfer the pre-trained models in the paper, and can be downloaded [here](https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/data/asnq.tar). 

### Domain Datasets
* **Wiki-QA**: we used the Wiki-QA dataset from [here](http://aka.ms/WikiQA) and removed all the questions that have no correct answers.
* **TREC-QA**: we used the `*-filtered.jsonl` version of this dataset from [here](https://github.com/mcrisc/lexdecomp/tree/master/trec-qa).


## Models

### Models Transferred on ASNQ

 - [BERT-Base ASNQ](https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_bert_base_asnq.tar)
 - [BERT-Large ASNQ](https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_bert_large_asnq.tar)
 - [RoBERTa-Base ASNQ](https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_roberta_base_asnq.tar)
 - [RoBERTa-Large ASNQ](https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_roberta_large_asnq.tar)

### TANDA: Models Transferred on ASNQ, then Fine-Tuned with Wiki-QA

 - [TANDA: BERT-Base ASNQ &rarr; Wiki-QA](https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_bert_base_asnq_wikiqa.tar)
 - [TANDA: BERT-Large ASNQ &rarr; Wiki-QA](https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_bert_large_asnq_wikiqa.tar)
 - [TANDA: RoBERTa-Large ASNQ &rarr; Wiki-QA](https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_roberta_large_asnq_wikiqa.tar)

### TANDA: Models Transferred on ASNQ, then Fine-Tuned with TREC-QA

 - [TANDA: BERT-Base ASNQ &rarr; TREC-QA](https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_bert_base_asnq_trec.tar)
 - [TANDA: BERT-Large ASNQ &rarr; TREC-QA](https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_bert_large_asnq_trec.tar)
 - [TANDA: RoBERTa-Large ASNQ &rarr; TREC-QA](https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_roberta_large_asnq_trec.tar)

## How To Cite TANDA
The paper is to appear in the AAAI 2020 proceedings. For now, please cite [the Arxiv version](https://arxiv.org/abs/1911.04118)

```
@article{garg2019tanda,
  title={TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection},
  author={Siddhant Garg and Thuy Vu and Alessandro Moschitti},
  year={2019},
  eprint={1911.04118},
}
```

## License Summary

The documentation, including the shared [data](#data) and [models](#models), is made available under the Creative Commons Attribution-ShareAlike 3.0 Unported License. See the LICENSE file.

The sample [script](#script) within this documentation is made available under the MIT-0 license. See the LICENSE-SAMPLECODE file.


## Contact
For help or issues, please submit a GitHub issue.

For direct communication, please contact Siddhant Garg (sgarg33 is at wisc dot edu, https://github.com/sid7954), Thuy Vu (thuyvu is at amazon dot com), or Alessandro Moschitti (amosch is at amazon dot com).

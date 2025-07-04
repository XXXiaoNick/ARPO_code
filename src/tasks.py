from collections import OrderedDict
import collections
import abc
import functools
from typing import Callable, List, Mapping
from src import metrics
import datasets
import logging
import numpy as np
import torch
import regex as re

logger = logging.getLogger(__name__)


class AbstractTask(abc.ABC):
    name = NotImplemented
    config = NotImplemented
    prefix = NotImplemented
    preprocessor: Callable = NotImplemented
    metric = NotImplemented
    metric_names = NotImplemented
    split_map = None
    labels_list = None
    split_to_data_split: Mapping[str, str] = {
        "train": "train",
        "validation": "validation",
        "test": "test"
    }
    small_datasets_without_all_splits = [
        "cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc", "superglue-wic", 
        "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb", "superglue-boolq", "xsum", "scitail"
    ]
    large_data_without_all_splits = [
        "qqp", "qnli", "superglue-record", "sst2", "squad", "snli", "anli", "amazon_polarity", 
        "yelp_polarity", "winogrande", "newsqa", "searchqa", "triviaqa", "nq", "hotpotqa"
    ]

    def __init__(self, config, seed=42):
        self.config = config
        self.seed = seed

    def get_max_target_length(self, tokenizer, default_max_length):
        if self.labels_list is not None:
            return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       add_prefix: bool = False,
                       prefix: str = None,
                       extra_fields={}):
        src_prefix = self.name if prefix is None else prefix
        sources = [src_prefix] + sources if add_prefix else sources
        return {'source': ' '.join(sources),
                'target': ' '.join(targets),
                'task': self.name}

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
            indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, self.config, split=split)

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def map_dataset(self, dataset, add_prefix):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           remove_columns=dataset.column_names)

    def get(self, split, add_prefix=True, n_obs=None, split_validation_test=False, lang=None, file_name=None):
        if split_validation_test and self.name in self.small_datasets_without_all_splits and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            if lang is not None:
                dataset = self.load_dataset(split=mapped_split, lang_code=lang)
            if file_name is not None:
                dataset = datasets.load_dataset(
                    'csv', data_files={split:file_name})[split]
            else:
                dataset = self.load_dataset(split=mapped_split)
            indices = self.get_split_indices(
                split, dataset, validation_size=len(dataset)//2)
            dataset = self.subsample(dataset, n_obs, indices)
        elif split_validation_test and self.name in self.large_data_without_all_splits and split != "test":
            if lang is not None:
                dataset = self.load_dataset(split="train", lang_code=lang)
            if file_name is not None:
                dataset = datasets.load_dataset(
                    'csv', data_files={split:file_name})[split]
            else:
                dataset = self.load_dataset(split="train")
            indices = self.get_split_indices(
                split, dataset, validation_size=1000)
            dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split[split]
            if lang is not None:
                dataset = self.load_dataset(split=mapped_split, lang_code=lang)

            if file_name is not None:
                dataset = datasets.load_dataset(
                    'csv', data_files={split:file_name})[split]
            else:
                dataset = self.load_dataset(split=mapped_split)
            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)
        return self.map_dataset(dataset, add_prefix)


class Squad(AbstractTask):
    name = "squad"
    metric = [metrics.squad]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split)

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers']).split("\t")
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)


class NaturalQuestions(AbstractTask):
    name = "nq"
    metric = [metrics.squad]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('', split=split)

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers'][0]).split("\t")
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)

    
class HotpotQA(AbstractTask):
    name = "hotpotqa"
    metric = [metrics.squad]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('', split=split)

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers'][0]).split("\t")
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)


class SearchQA(AbstractTask):
    name = "searchqa"
    metric = [metrics.squad]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('', split=split)

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers'][0]).split("\t")
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)
   

class NewsQA(AbstractTask):
    name = "newsqa"
    metric = [metrics.squad]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('', split=split)

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers'][0]).split("\t")
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)


class DROP(AbstractTask):
    name = "drop"
    metric = [metrics.squad]

    def load_dataset(self, split):
        return datasets.load_dataset("drop", split=split)

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers_spans']['spans'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['passage'])
        source = ["question:", question,
                  "context:", context]
        target = [answer]
        return self.seq2seq_format(source, target, add_prefix)


class PIQA(AbstractTask):
    name = "piqa"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('piqa', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['goal'], "choice1:",
                     example["sol1"][0], "choice2:", example["sol2"][0]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class CommonsenseQA(AbstractTask):
    name = "commonsense_qa"
    labels_list = ["0", "1", "2", "3", "4"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('commonsense_qa', split=split)

    def preprocessor(self, example, add_prefix=True):
        label2id = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}
        src_texts = ["question:", example['question'], "choice1:", example["choices"]["text"][0], "choice2:", example["choices"]["text"][1],
                     "choice3:", example["choices"]["text"][2], "choice4:", example["choices"]["text"][3], "choice5:", example["choices"]["text"][4]]
        tgt_texts = [label2id[example["answerKey"]]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SocialIQA(AbstractTask):
    name = "social_i_qa"
    labels_list = ["0", "1", "2"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "validation": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('social_i_qa', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'], "context:", example["context"], "|| choice0:",
                     example["answerA"][0], "|| choice1:", example["answerB"][0], "|| choice2:", example["answerC"][0]]
        tgt_texts = [str(int(example["label"]) - 1)]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SciTail(AbstractTask):
    name = "scitail"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('scitail', "snli_format", split=split)

    def preprocessor(self, example, add_prefix=True):
        label2id = {"entailment": "0", "neutral": "1"}
        src_texts = ["premise:", example['sentence1'],
                     "hypothesis:", example["sentence2"]]
        tgt_texts = [label2id[example["gold_label"]]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class COLA(AbstractTask):
    name = "cola"
    labels_list = ["0", "1"]
    metric = [metrics.matthews_corrcoef]
    metric_names = ["matthews_correlation"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2',
                                     split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class YelpPolarity(AbstractTask):
    name = "yelp_polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split):
        print(split)
        return datasets.load_dataset('yelp_polarity')[split]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class Amazon_Polarity(AbstractTask):
    name = "amazon_polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('amazon_polarity', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", "<title> {0} <context> {1}".format(
            example['title'], example['context'])]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class STSB(AbstractTask):
    name = "stsb"
    labels_list = [str(np.round(label, decimals=1))
                   for label in np.arange(0, 5.2, 0.2)]
    metric = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    metric_names = ["pearson", "spearmanr"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QQP(AbstractTask):
    name = "qqp"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp',
                                     split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question1:", example['question1'],
                     "question2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SNLI(AbstractTask):
    name = "snli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('snli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis: ", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MultiNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('multi_nli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'],
                     "sentence:", example["sentence"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class RTE(AbstractTask):
    name = "rte"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte',
                                     split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEBoolQ(AbstractTask):
    name = "superglue-boolq"
    labels_list = ['0', '1']
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'boolq', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"],
                     "passage:", example["passage"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUERTE(AbstractTask):
    name = "superglue-rte"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'rte', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUECB(AbstractTask):
    name = "superglue-cb"
    labels_list = ['0', '1', '2']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy]
    metric_names = ["f1_multiclass", "accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'cb', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUECOPA(AbstractTask):
    name = "superglue-copa"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'copa', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "choice1:", example["choice1"],
                     "choice2:", example["choice2"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEMultiRC(AbstractTask):
    name = "superglue-multirc"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.f1_score_with_invalid,
              ]
    metric_names = ["f1", "em"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'multirc', split=split)

    def remove_markup(self, text):
        """Removes the HTML markup."""
        text = re.sub('<br>', ' ', text)
        text = re.sub('<(/)?b>', '', text)
        return text

    def preprocessor(self, example, add_prefix=True):
        group = example['idx']['question']
        src_texts = ["question:", self.remove_markup(example["question"]),
                     "answer:", self.remove_markup(example["answer"]),
                     "paragraph:", self.remove_markup(example["paragraph"])]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields={"group": group})


class SuperGLUEWIC(AbstractTask):
    name = "superglue-wic"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'wic', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example["sentence1"],
                     "sentence2:", example["sentence2"],
                     "word:", example["word"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEWSCFixed(AbstractTask):
    name = "superglue-wsc.fixed"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'wsc.fixed', split=split)

    def _mark_span(self, text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub('N', str(span_idx), pattern_tmpl)
        pattern = re.sub('W', span_str, pattern)
        return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

    def preprocessor(self, example, add_prefix=True):
        text = example['text']
        text = self._mark_span(
            text, example['span1_text'], example['span1_index'], '*')
        span2_index = example['span2_index'] + 2 * \
            int(example['span1_index'] < example['span2_index'])
        text = self._mark_span(text, example['span2_text'], span2_index, '#')
        src_texts = ["text:", text]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUERecord(AbstractTask):
    name = "superglue-record"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.squad]
    metric_names = ["squad"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'record', split=split)

    def preprocessor(self, batch, add_prefix=True):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            passage = ex['passage']
            passage = re.sub(
                r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            if add_prefix:
                inputs = self.name + " " + inputs
            num_answers = len(ex["answers"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["source"].extend([inputs] * num_duplicates)
            new_batch["target"].extend(
                ex["answers"] if num_answers > 0 else ["<unk>"])
            new_batch["task"].extend([self.name] * num_duplicates)
            new_batch["extra_fields"].extend(
                [{"answers": ex["answers"]}]*num_duplicates)
        return new_batch

    def map_dataset(self, dataset, add_prefix=True):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           batched=True, remove_columns=dataset.column_names)


class WinoGrande(AbstractTask):
    name = "winogrande"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('winogrande', "winogrande_xl", split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example["sentence"],
                     "option0:", example["option1"],
                     "option1:", example["option1"]]
        tgt_texts = [str(int(example["answer"]) - 1)]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class PAWS(AbstractTask):
    name = "paws"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('paws', 'labeled_final', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


TASK_MAPPING = OrderedDict(
    [
        ('squad', Squad),
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('qqp', QQP),
        ('stsb', STSB),
        ('superglue-boolq', SuperGLUEBoolQ),
        ('superglue-rte', SuperGLUERTE),
        ('superglue-cb', SuperGLUECB),
        ('superglue-copa', SuperGLUECOPA),
        ('superglue-multirc', SuperGLUEMultiRC),
        ('superglue-wic', SuperGLUEWIC),
        ('superglue-wsc.fixed', SuperGLUEWSCFixed),
        ('superglue-record', SuperGLUERecord),
        ('multi_nli', MultiNLI),
        ('snli', SNLI),
        ('piqa', PIQA),
        ('drop', DROP),
        ('newsqa', NewsQA),
        ('searchqa', SearchQA),
        ('triviaqa', Squad),
        ('nq', NaturalQuestions),
        ('hotpotqa', HotpotQA),
        ("social_i_qa", SocialIQA),
        ("commonsense_qa", CommonsenseQA),
        ("winogrande", WinoGrande),
        ("scitail", SciTail),
        ('yelp_polarity', YelpPolarity),
        ('amazon_polarity', Amazon_Polarity),
        ('paws', PAWS),
    ]
)


class AutoTask:
    @classmethod
    def get(self, task, config, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )


def round_stsb_target(label):
    return np.round((label * 5) / 5, decimals=1)


def pad_punctuation(text):
    text = re.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text


num_labels_mapping = {
    "mnli": {"0": 0, "1": 0, "2": 0},
    "qqp": {"0": 0, "1": 0},
    "qnli": {"0": 0, "1": 0},
    "sst2": {"0": 0, "1": 0},
    "mrpc": {"0": 0, "1": 0},
    "stsb": {"0": 0, "1": 0},
    "rte": {"0": 0, "1": 0},
    "cola": {"0": 0, "1": 0},
    "superglue-cb": {"0": 0, "1": 0, "2": 0},
    "superglue-boolq": {"0": 0, "1": 0},
    "superglue-wic": {"0": 0, "1": 0},
    "superglue-wsc.fixed": {"0": 0, "1": 0},
    "superglue-multirc": {"0": 0, "1": 0},
    "scitail": {"0": 0, "1": 0},
}
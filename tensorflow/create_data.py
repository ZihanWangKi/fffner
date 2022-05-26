import json
import os
import sys

import collections
import pickle

import transformers
from tqdm import tqdm
import numpy as np
import tensorflow as tf

class NERDataset:
    def __init__(self, words_path, labels_path, tokenizer, is_train, no_parentheses,
                 prediction_together, cls_on_token, not_mask, label_to_entity_type_index):
        self.words_path = words_path
        self.labels_path = labels_path
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.no_parentheses = no_parentheses
        self.prediction_together = prediction_together
        self.cls_on_token = cls_on_token
        self.not_mask = not_mask
        self.left_bracket_1 = self.tokenize_word(" [")[0]
        self.right_bracket_1 = self.tokenize_word(" ]")[0]
        self.left_bracket_2 = self.tokenize_word(" (")[0]
        self.right_bracket_2 = self.tokenize_word(" )")[0]
        self.mask_id = self.tokenizer.mask_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.data = []  # may be very large for test
        self.id_to_sentence_infos = dict()
        self.id_counter = 0
        self.all_tokens = []
        self.all_labels = []
        self.max_seq_len_in_data = 0
        self.max_len = 128
        self.label_to_entity_type_index = label_to_entity_type_index

    def iter_read(self):
        with open(self.words_path) as f1, open(self.labels_path) as f2:
            for si, (l1, l2) in enumerate(zip(f1, f2)):
                tokens = l1.strip().split(' ')
                labels = l2.strip().split(' ')
                tokens = ["(" if token == "[" else token for token in tokens]
                tokens = [")" if token == "]" else token for token in tokens]
                yield tokens, labels

    def tokenize_word(self, word):
        result = self.tokenizer(word, add_special_tokens=False)
        return result['input_ids']

    def tokenize_word_list(self, word_list):
        return [self.tokenize_word(word) for word in word_list]

    def process_to_input(self, input_ids, cls_token_pos, span_start_pos, is_entity_label, entity_type_label, sid,
                         span_start, span_end):
        self.id_counter += 1
        self.id_to_sentence_infos[self.id_counter] = {
            "sid": sid,
            "span_start": span_start,
            "span_end": span_end,
        }
        seqlen = len(input_ids)
        self.max_seq_len_in_data = max(self.max_seq_len_in_data, seqlen)
        assert seqlen <= self.max_len
        return {
            'input_ids': input_ids,
            'attention_mask': [1] * seqlen,
            'cls_token_pos': cls_token_pos,
            'span_start_pos': span_start_pos,
            'is_entity_label': 1 if is_entity_label else 0,
            'entity_type_label': entity_type_label,
            'sentence_id': sid,
            'span_start': span_start,
            'span_end': span_end,
            'id': self.id_counter,
        }

    def process_word_list_and_spans_to_inputs(self, sid, word_list, spans):
        tokenized_word_list = self.tokenize_word_list(word_list)
        total_len = sum(len(x) for x in tokenized_word_list)
        projected_len = 2 + 3 + 2 + 3 + total_len
        if projected_len > self.max_len:
            print(f"projected_len {projected_len} too long")
            return
        for span_start, span_end, span_type, span_label in spans:
            assert span_type == 'mask'
            input_ids = []
            input_ids.append(self.cls_token_id)
            for ids in tokenized_word_list[: span_start]:
                input_ids.extend(ids)

            if not self.cls_on_token:
                if not self.prediction_together:
                    if not self.no_parentheses:
                        input_ids.append(self.left_bracket_1)
                    cls_token_pos = len(input_ids)
                    input_ids.append(self.mask_id if not self.not_mask else 8487)
                    if not self.no_parentheses:
                        input_ids.append(self.right_bracket_1)
            else:
                cls_token_pos = len(input_ids)

            if not self.no_parentheses:
                input_ids.append(self.left_bracket_1)
            for ids in tokenized_word_list[span_start: span_end + 1]:
                input_ids.extend(ids)
            if not self.no_parentheses:
                input_ids.append(self.right_bracket_1)

            if not self.cls_on_token:
                if not self.no_parentheses:
                    input_ids.append(self.left_bracket_1)

                span_start_pos = len(input_ids)
                if self.prediction_together:
                    cls_token_pos = len(input_ids)

                input_ids.append(self.mask_id if not self.not_mask else 2828)
                if not self.no_parentheses:
                    input_ids.append(self.right_bracket_1)
            else:
                span_start_pos = len(input_ids) - 1

            for ids in tokenized_word_list[span_end + 1:]:
                input_ids.extend(ids)
            input_ids.append(self.sep_token_id)
            is_entity_label = span_label in self.label_to_entity_type_index
            entity_type_label = self.label_to_entity_type_index.get(span_label, 0)
            yield self.process_to_input(input_ids, cls_token_pos, span_start_pos,
                                        is_entity_label, entity_type_label,
                                        sid, span_start, span_end)

    def bio_labels_to_spans(self, bio_labels):
        spans = []
        for i, label in enumerate(bio_labels):
            if label.startswith("B-"):
                spans.append([i, i, label[2:]])
            elif label.startswith("I-"):
                if len(spans) == 0:
                    print("Error... I-tag should not start a span")
                    spans.append([i, i, label[2:]])
                elif spans[-1][1] != i - 1 or spans[-1][2] != label[2:]:
                    print("Error... I-tag not consistent with previous tag")
                    spans.append([i, i, label[2:]])
                else:
                    spans[-1][1] = i
            elif label.startswith("O"):
                pass
            else:
                assert False, bio_labels
        spans = list(filter(lambda x: x[2] in self.label_to_entity_type_index.keys(), spans))
        return spans

    def collate_fn(self, batch):
        # print(batch)
        batch = self.tokenizer.pad(
            batch,
            padding='max_length',
            max_length=self.max_len,
        )
        return batch

    def prepare(self,
                negative_multiplier=10,
                biased_negatives=0):
        desc = "prepare data for training" if self.is_train else "prepare data for testing"
        total_missed_entities = 0
        total_entities = 0
        min_span_len = 1
        max_span_len = -1
        for sid, (tokens, labels) in tqdm(enumerate(self.iter_read()), desc=desc):
            self.all_tokens.append(tokens)
            self.all_labels.append(labels)
            entity_spans = self.bio_labels_to_spans(labels)
            entity_spans_dict = {(start, end): ent_type for start, end, ent_type in entity_spans}
            num_entities = len(entity_spans_dict)
            num_negatives = int((len(tokens) + num_entities * 10) * negative_multiplier)
            num_negatives = min(num_negatives, len(tokens) * (len(tokens) + 1) // 2)
            min_words = 1
            max_words = len(tokens)
            total_entities += len(entity_spans)

            spans = []
            if biased_negatives == 1:
                is_token_entity_prefix = [0] * (len(tokens) + 1)
                for start, end, _ in entity_spans:
                    for i in range(start, end + 1):
                        is_token_entity_prefix[i + 1] = 1
                for i in range(len(tokens)):
                    is_token_entity_prefix[i + 1] += is_token_entity_prefix[i]

                possible_negative_spans = []
                possible_negative_spans_probs = []
                for n_words in range(min_words, max_words + 1):
                    for i in range(len(tokens) - n_words + 1):
                        j = i + n_words - 1
                        ent_type = entity_spans_dict.get((i, j), 'O')
                        if not self.is_train or ent_type != 'O':
                            spans.append((i, j, 'mask', ent_type))
                        else:
                            possible_negative_spans.append((i, j, 'mask', ent_type))
                            intersection_size = 1. * (is_token_entity_prefix[j + 1] - is_token_entity_prefix[i] + 1) / (j + 1 - i)
                            if biased_negatives == 1:
                                possible_negative_spans_probs.append(2.718 ** intersection_size)
                            else:
                                possible_negative_spans_probs.append(intersection_size)

                if len(possible_negative_spans) > 0 and num_negatives > 0:
                    possible_negative_spans_probs = np.array(possible_negative_spans_probs) / np.sum(possible_negative_spans_probs)
                    additional_negative_span_indices = np.random.choice(len(possible_negative_spans),
                                                                        num_negatives,
                                                                        replace=biased_negatives == 1, p=possible_negative_spans_probs)
                    spans.extend([possible_negative_spans[x] for x in additional_negative_span_indices])
            elif biased_negatives == 0:
                total_spans = (max_words - min_words + 1) * len(tokens)
                sample_prob = num_negatives / total_spans  # could be > 1 if unfortunate
                for n_words in range(min_words, max_words + 1):
                    for i in range(len(tokens) - n_words + 1):
                        j = i + n_words - 1
                        ent_type = entity_spans_dict.get((i, j), 'O')
                        if not self.is_train or ent_type != 'O' or random.uniform(0, 1) <= sample_prob:
                            spans.append((i, j, 'mask', ent_type))
            else:
                assert False
            for instance in self.process_word_list_and_spans_to_inputs(sid, tokens, spans):
                self.data.append(instance)
        print(f"{total_missed_entities}/{total_entities} are missed due to length")
        print(f"Total {self.__len__()} instances")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    path_to_data_folder = "../dataset"
    dataset_name = sys.argv[1]
    dataset = os.path.join(path_to_data_folder, dataset_name)
    train_file = "few_shot_5_0"
    test_file = "test"
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    entity_map = json.load(open(os.path.join(dataset, "entity_map.json")))
    label_to_entity_type_index = {k: i for i, k in enumerate(list(entity_map.keys()))}
    train_dataset = NERDataset(words_path=os.path.join(dataset, train_file + ".words"),
                               labels_path=os.path.join(dataset, train_file + ".ner"),
                               tokenizer=tokenizer, is_train=True, no_parentheses=False,
                               prediction_together=False, cls_on_token=False, not_mask=False,
                               label_to_entity_type_index=label_to_entity_type_index)
    eval_dataset = NERDataset(words_path=os.path.join(dataset, test_file + ".words"),
                               labels_path=os.path.join(dataset, test_file + ".ner"),
                               tokenizer=tokenizer, is_train=False, no_parentheses=False,
                               prediction_together=False, cls_on_token=False, not_mask=False,
                              label_to_entity_type_index=label_to_entity_type_index)
    train_dataset.prepare(negative_multiplier=3, biased_negatives=1)
    print(train_dataset.data[0])
    train_data = train_dataset.collate_fn(train_dataset.data)
    print(train_data.keys())
    eval_dataset.prepare(negative_multiplier=3, biased_negatives=1)
    eval_data = eval_dataset.collate_fn(eval_dataset.data)


    def file_based_convert_examples_to_features(examples,
                                                output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""
        tf.io.gfile.makedirs(os.path.dirname(output_file))
        writer = tf.io.TFRecordWriter(output_file)

        for ex_index in range(len(examples["input_ids"])):
            if ex_index % 10000 == 0:
                print(f"Writing example {ex_index} of {len(examples['input_ids'])}")
                print(examples["input_ids"][ex_index])

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            def create_float_feature(values):
                f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(examples["input_ids"][ex_index])
            features["input_mask"] = create_int_feature(examples["attention_mask"][ex_index])
            features["segment_ids"] = create_int_feature([0] * len(examples["attention_mask"][ex_index]))
            features["cls_token_pos"] = create_int_feature([examples["cls_token_pos"][ex_index]])
            features["span_start_pos"] = create_int_feature([examples["span_start_pos"][ex_index]])
            features["is_entity_label"] = create_int_feature([examples["is_entity_label"][ex_index]])
            features["entity_type_label"] = create_int_feature([examples["entity_type_label"][ex_index]])
            features["example_id"] = create_int_feature([examples["id"][ex_index]])
            features["sentence_id"] = create_int_feature([examples["sentence_id"][ex_index]])
            features["span_start"] = create_int_feature([examples["span_start"][ex_index]])
            features["span_end"] = create_int_feature([examples["span_end"][ex_index]])
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()


    file_based_convert_examples_to_features(train_data, f"{dataset_name}_{train_file}.tf_record")
    file_based_convert_examples_to_features(eval_data, f"{dataset_name}_{test_file}.tf_record")

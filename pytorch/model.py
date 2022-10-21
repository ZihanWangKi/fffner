import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torch.nn.parallel
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

import transformers
from tqdm import tqdm
from collections import defaultdict
import pickle
import json

PATH_TO_DATA_DIR = "../dataset"

def get_p_r_f(truth, pred):
    n_pred = len(pred)
    n_truth = len(truth)
    n_correct = len(set(pred) & set(truth))
    precision = 1. * n_correct / n_pred if n_pred != 0 else 0.0
    recall = 1. * n_correct / n_truth if n_truth != 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0.0 else 0.0
    return {
        "n_pred": n_pred,
        "n_truth": n_truth,
        "n_correct": n_correct,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, words_path, labels_path, tokenizer, is_train,
                 label_to_entity_type_index,
                 ablation_not_mask, ablation_no_brackets, ablation_span_type_together):
        self.words_path = words_path
        self.labels_path = labels_path
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.label_to_entity_type_index = label_to_entity_type_index
        self.ablation_no_brackets = ablation_no_brackets
        self.ablation_span_type_together = ablation_span_type_together
        self.ablation_not_mask = ablation_not_mask

        self.left_bracket_1 = self.tokenize_word(" [")[0]
        self.right_bracket_1 = self.tokenize_word(" ]")[0]
        self.left_bracket_2 = self.tokenize_word(" (")[0]
        self.right_bracket_2 = self.tokenize_word(" )")[0]
        self.mask_id = self.tokenizer.mask_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        self.data = []  # may be pretty large for test
        self.id_to_sentence_infos = dict()
        self.id_counter = 0
        self.all_tokens = []
        self.all_labels = []
        self.max_seq_len_in_data = 0

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
        return {
            'input_ids': input_ids,
            'attention_mask': [1] * seqlen,
            'cls_token_pos': cls_token_pos,
            'span_start_pos': span_start_pos,
            'is_entity_label': 1 if is_entity_label else 0,
            'entity_type_label': entity_type_label,
            'id': self.id_counter,
        }

    def process_word_list_and_spans_to_inputs(self, sid, word_list, spans):
        tokenized_word_list = self.tokenize_word_list(word_list)
        for span_start, span_end, span_type, span_label in spans:
            assert span_type == 'mask'
            input_ids = []
            input_ids.append(self.cls_token_id)
            for ids in tokenized_word_list[: span_start]:
                input_ids.extend(ids)

            if not self.ablation_span_type_together:
                if not self.ablation_no_brackets:
                    input_ids.append(self.left_bracket_1)
                cls_token_pos = len(input_ids)
                input_ids.append(self.mask_id if not self.ablation_not_mask else 8487)
                if not self.ablation_no_brackets:
                    input_ids.append(self.right_bracket_1)


            if not self.ablation_no_brackets:
                input_ids.append(self.left_bracket_1)
            for ids in tokenized_word_list[span_start: span_end + 1]:
                input_ids.extend(ids)
            if not self.ablation_no_brackets:
                input_ids.append(self.right_bracket_1)

            if not self.ablation_no_brackets:
                input_ids.append(self.left_bracket_1)

            span_start_pos = len(input_ids)
            if self.ablation_span_type_together:
                cls_token_pos = len(input_ids)

            input_ids.append(self.mask_id if not self.ablation_not_mask else 2828)
            if not self.ablation_no_brackets:
                input_ids.append(self.right_bracket_1)

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
        batch = self.tokenizer.pad(
            batch,
            padding=True,
            max_length=512,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return batch

    def prepare(self,
                negative_multiplier=3.):
        desc = "prepare data for training" if self.is_train else "prepare data for testing"
        total_missed_entities = 0
        total_entities = 0
        for sid, (tokens, labels) in tqdm(enumerate(self.iter_read()), desc=desc):
            self.all_tokens.append(tokens)
            self.all_labels.append(labels)
            entity_spans = self.bio_labels_to_spans(labels)
            entity_spans_dict = {(start, end): ent_type for start, end, ent_type in entity_spans}
            num_entities = len(entity_spans_dict)
            num_negatives = int((len(tokens) + num_entities * 10) * negative_multiplier)
            num_negatives = min(num_negatives, len(tokens) * (len(tokens) + 1) // 2)
            min_words = 1
            max_words = len(tokens) # this can be set lower if you believe the maximum entity length is small & you want smaller dataset -> faster training
            total_entities += len(entity_spans)

            spans = []
            if self.is_train:
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
                            possible_negative_spans_probs.append(2.718 ** intersection_size)

                if len(possible_negative_spans) > 0 and num_negatives > 0:
                    possible_negative_spans_probs = np.array(possible_negative_spans_probs) / np.sum(possible_negative_spans_probs)
                    additional_negative_span_indices = np.random.choice(len(possible_negative_spans),
                                                                        num_negatives,
                                                                        replace=True, p=possible_negative_spans_probs)
                    spans.extend([possible_negative_spans[x] for x in additional_negative_span_indices])
            else:
                for n_words in range(min_words, max_words + 1):
                    for i in range(len(tokens) - n_words + 1):
                        j = i + n_words - 1
                        ent_type = entity_spans_dict.get((i, j), 'O')
                        spans.append((i, j, 'mask', ent_type))

            for instance in self.process_word_list_and_spans_to_inputs(sid, tokens, spans):
                self.data.append(instance)
        print(f"{total_missed_entities}/{total_entities} are ignored due to length")
        print(f"Total {self.__len__()} instances")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Transform(torch.nn.Module):
    def __init__(self, hidden_size, target_size, dropout_rate):
        super().__init__()
        self.dense1 = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.GELU()
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.dense2 = torch.nn.Linear(hidden_size, target_size)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class FFFNERModel(LightningModule):
    def __init__(self, dataset_name, train_path_suffix, val_path_suffix, test_path_suffix,
                 workers, batch_size, validation_batch_size,
                 lr, pretrained_lm, dropout,
                 negative_multiplier,
                 ablation_not_mask,
                 ablation_no_brackets,
                 ablation_span_type_together,
                 **kwargs,):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.train_path_suffix = train_path_suffix
        self.val_path_suffix = val_path_suffix
        self.test_path_suffix = test_path_suffix
        self.train_path_prefix = os.path.join(PATH_TO_DATA_DIR, self.dataset_name, self.train_path_suffix)
        self.val_path_prefix = os.path.join(PATH_TO_DATA_DIR, self.dataset_name, self.val_path_suffix)
        self.test_path_prefix = os.path.join(PATH_TO_DATA_DIR, self.dataset_name, self.test_path_suffix)

        assert os.path.exists(self.train_path_prefix + ".ner") and os.path.exists(self.val_path_prefix + ".ner") and \
               os.path.exists(self.test_path_prefix + ".ner")

        self.workers = workers
        self.batch_size = batch_size
        self.validation_batch_size = self.batch_size if validation_batch_size is None else validation_batch_size
        self.lr = lr
        self.pretrained_lm = pretrained_lm
        self.dropout = dropout

        self.negative_multiplier = negative_multiplier
        self.ablation_not_mask = ablation_not_mask
        self.ablation_no_brackets = ablation_no_brackets
        self.ablation_span_type_together = ablation_span_type_together


        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_lm,
            num_labels=2, # dub
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.pretrained_lm,
            use_fast=True
        )

        # loading dataset info
        with open(os.path.join(PATH_TO_DATA_DIR, dataset_name, f"entity_map.json")) as f:
            config = json.load(f)
        self.label_to_entity_type_index = {k: i for i, k in enumerate(list(config.keys()))}
        self.entity_type_names = list(config.values())


        self.val_dataset_obj = None
        self.test_dataset_obj = None
        self.num_epochs_passed = 0

        ### Testing forward & obtaining hidden state size
        entity_input = self.tokenizer(self.entity_type_names)
        entity_input["cls_token_pos"] = [1] * len(self.entity_type_names)
        batch = self.tokenizer.pad(
            entity_input,
            padding=True,
            max_length=512,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        self.model.eval()
        with torch.no_grad():
            out = self.forward(
                batch["input_ids"],
                batch["attention_mask"],
                batch["cls_token_pos"],
                batch["cls_token_pos"],
            )
        self.model.train()
        mask_token_repr = out["mask_token_repr"]
        mask_token_repr = mask_token_repr.detach().clone()
        hidden_size = mask_token_repr.size()[1]

        self.cls_transform = Transform(hidden_size, 2, dropout)
        self.cls_loss_fct = torch.nn.CrossEntropyLoss()

        self.type_transform = Transform(hidden_size, len(self.entity_type_names), dropout)
        self.type_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.train_steps = (self.hparams.max_epochs * len(self.train_dataloader()))


    def set_outputdir(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.result_save_dir = dir

    def forward(self, input_ids, attention_mask, cls_token_pos, span_start_pos):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True)
        if "bart" in self.pretrained_lm:
            hidden_states = output.decoder_hidden_states[-1]
        else:
            hidden_states = output.hidden_states[-1]
        n_batchs, n_pos, n_emb = hidden_states.size()
        mask_token_repr = hidden_states.gather(1, cls_token_pos.unsqueeze(1).unsqueeze(2).expand(
            (n_batchs, 1, n_emb))).squeeze(1)
        span_token_repr = hidden_states.gather(1, span_start_pos.unsqueeze(1).unsqueeze(2).expand(
            (n_batchs, 1, n_emb))).squeeze(1)
        return {
            "mask_token_repr": mask_token_repr,
            "span_token_repr": span_token_repr,
        }

    def forward_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        cls_token_pos = batch["cls_token_pos"]
        span_start_pos = batch["span_start_pos"]
        is_entity_label = batch["is_entity_label"]
        entity_type_label = batch["entity_type_label"]

        output = self.forward(input_ids, attention_mask, cls_token_pos, span_start_pos)
        mask_token_repr = output["mask_token_repr"]
        span_token_repr = output["span_token_repr"]

        is_entity_logits = self.cls_transform(mask_token_repr)
        is_entity_loss = self.cls_loss_fct(is_entity_logits.view(-1, 2), is_entity_label.view(-1))
        entity_type_logits = self.type_transform(span_token_repr)
        entity_type_loss = self.type_loss_fct(entity_type_logits, entity_type_label)

        entity_type_loss = entity_type_loss[is_entity_label.type(torch.bool)].sum()
        if is_entity_label.sum() > 0:
            entity_type_loss /= is_entity_label.sum()

        loss = is_entity_loss + entity_type_loss

        acc1, = self.__accuracy(is_entity_logits, is_entity_label, topk=(1,))
        if is_entity_label.sum() > 0:
            acc2, = self.__accuracy(entity_type_logits[is_entity_label.type(torch.bool)],
                                    entity_type_label[is_entity_label.type(torch.bool)], topk=(1,))
        else:
            acc2 = 100.

        return {
            "acc1": acc1,
            "acc2": acc2,
            "loss": loss,
            "is_entity_label": is_entity_label,
            "is_entity_logits": is_entity_logits,
            "entity_type_label": entity_type_label,
            "entity_type_logits": entity_type_logits,
            "id": batch["id"],
        }

    def training_step(self, batch, batch_idx):
        ret = self.forward_step(batch)
        self.log("train_acc1", ret["acc1"], on_step=False, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc2", ret["acc2"], on_step=False, prog_bar=True, on_epoch=True, logger=True)
        return ret["loss"]

    def training_epoch_end(self, *args, **kwargs):
        self.num_epochs_passed += 1

    def validation_step(self, batch, batch_idx):
        ret = self.forward_step(batch)
        self.log("val_acc1", ret["acc1"], on_step=False, prog_bar=True, on_epoch=True, logger=True)
        self.log("val_acc2", ret["acc2"], on_step=False, prog_bar=True, on_epoch=True, logger=True)
        return ret

    def to_list(self, T):
        return T.cpu().detach().clone().numpy().tolist()

    def naive_metric(self, results):
        gt_list = []
        pd_list = []
        for sid, span_list in results.items():
            for span_start, span_end, is_entity_label, is_entity_logit, entity_type_label, entity_type_logit in span_list:
                if is_entity_label == 1:
                    gt_list.append((sid, span_start, span_end, entity_type_label))
                if is_entity_logit[1] >= is_entity_logit[0]:
                    pd_list.append((sid, span_start, span_end, np.argmax(entity_type_logit).item()))
        print("~" * 88)
        print("with type", get_p_r_f(gt_list, pd_list))
        print("without type", get_p_r_f([tuple(x[: -1]) for x in gt_list], [tuple(x[: -1]) for x in pd_list]))
        print("~" * 88)

    def full_metric(self, results):
        def resolve(length, spans, prediction_confidence):
            used = [False] * length
            spans = sorted(spans, key=lambda x: prediction_confidence[(x[0], x[1])], reverse=True)
            real_spans = []
            for span_start, span_end, ent_type in spans:
                fill = False
                for s in range(span_start, span_end + 1):
                    if used[s]:
                        fill = True
                        break
                if not fill:
                    real_spans.append((span_start, span_end, ent_type))
                    for s in range(span_start, span_end + 1):
                        used[s] = True
            return real_spans

        def softmax(x):
            x = np.array(x)
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        ground_truth = []
        prediction_1 = [] # raw predictions
        prediction_2 = [] # resolving conflicts
        for key in sorted(list(results.keys())):
            single_result = results[key]
            gt_entities = []
            predictied_entities = []
            prediction_confidence_span = {}
            prediction_confidence_type = {}
            length = 0
            for span_start, span_end, ground_truth_span, prediction_span, ground_truth_type, prediction_type in single_result:
                if ground_truth_span == 1:
                    gt_entities.append((span_start, span_end, ground_truth_type))
                if prediction_span[1] > prediction_span[0]:
                    predictied_entities.append((span_start, span_end, np.argmax(prediction_type).item()))
                prediction_confidence_span[(span_start, span_end)] = max(softmax(prediction_span))
                prediction_confidence_type[(span_start, span_end)] = max(softmax(prediction_type))
                length = max(length, span_end)
            length += 1
            ground_truth.extend([(key, *x) for x in gt_entities])
            prediction_1.extend([(key, *x) for x in predictied_entities])
            resolved_predicted = resolve(length, predictied_entities, prediction_confidence_span)
            prediction_2.extend([(key, *x) for x in resolved_predicted])

        raw = get_p_r_f(ground_truth, prediction_1)
        resolved = get_p_r_f(ground_truth, prediction_2)
        print("~" * 88)
        print("raw_predictions", raw)
        print("resolved_predictions", resolved)
        print("~" * 88)
        return {
            "raw_f1": raw["f1"],
            "raw_precision": raw["precision"],
            "raw_recall": raw["recall"],
            "resolved_f1": resolved["f1"],
            "resolved_precision": resolved["precision"],
            "resolved_recall": resolved["recall"],
        }

    def eval_end(self, outputs, dataset_obj):
        ids = torch.cat([x['id'] for x in outputs])
        is_entity_labels = torch.cat([x['is_entity_label'] for x in outputs])
        is_entity_logits = torch.cat([x['is_entity_logits'] for x in outputs])
        entity_type_labels = torch.cat([x['entity_type_label'] for x in outputs])
        entity_type_logits = torch.cat([x['entity_type_logits'] for x in outputs])
        ids = self.to_list(ids)
        is_entity_labels = self.to_list(is_entity_labels)
        is_entity_logits = self.to_list(is_entity_logits)
        entity_type_labels = self.to_list(entity_type_labels)
        entity_type_logits = self.to_list(entity_type_logits)
        assert len(ids) == len(is_entity_labels) == len(is_entity_logits) == len(entity_type_labels) == len(
            entity_type_logits)

        results = defaultdict(list)
        for id, is_entity_label, is_entity_logit, entity_type_label, entity_type_logit in zip(ids, is_entity_labels,
                                                                                              is_entity_logits,
                                                                                              entity_type_labels,
                                                                                              entity_type_logits):
            info = dataset_obj.id_to_sentence_infos[id]
            results[info["sid"]].append((info["span_start"], info["span_end"],
                                         is_entity_label, is_entity_logit, entity_type_label,
                                         entity_type_logit))
        return results, {
            "ids": ids,
            "is_entity_labels": is_entity_labels,
            "is_entity_logits": is_entity_logits,
            "entity_type_labels": entity_type_labels,
            "entity_type_logits": entity_type_logits,
        }

    def validation_epoch_end(self, outputs):
        results, predictions = self.eval_end(outputs, self.val_dataset_obj)

        evaluation_result = self.full_metric(results)
        with open(os.path.join(self.result_save_dir, "dev_predictions.pk"), "wb") as f:
            pickle.dump({
                "results": results,
                "tokens": self.val_dataset_obj.all_tokens,
                "labels": self.val_dataset_obj.all_labels,
            }, f)

        self.log("val_metric", (evaluation_result["raw_f1"] + evaluation_result["resolved_f1"]) * 1000000 + self.num_epochs_passed)
        return predictions

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, outputs):
        results, predictions = self.eval_end(outputs, self.test_dataset_obj)
        self.full_metric(results)

        with open(os.path.join(self.result_save_dir, "predictions.pk"), "wb") as f:
            pickle.dump({
                "results": results,
                "tokens": self.test_dataset_obj.all_tokens,
                "labels": self.test_dataset_obj.all_labels,
            }, f)

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


    def configure_optimizers(self):
        def get_parameter_names(model, forbidden_layer_types):
            """
            Returns the names of the model parameters that are not inside a forbidden layer.
            """
            result = []
            for name, child in model.named_children():
                result += [
                    f"{name}.{n}"
                    for n in get_parameter_names(child, forbidden_layer_types)
                    if not isinstance(child, tuple(forbidden_layer_types))
                ]
            # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
            result += list(model._parameters.keys())
            return result

        decay_parameters = get_parameter_names(self, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        optimizer = torch.optim.AdamW([{
                "params": [p for n, p in self.named_parameters() if n in decay_parameters],
                "weight_decay": 0.02,
                "lr": self.lr,
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
                "lr": self.lr,
            }
        ])
        num_training_steps = self.train_steps
        num_warmup_steps = 0
        last_epoch = -1

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        scheduler = {
            'scheduler': lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch),
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        print("Access train dataloader")
        train_dataset = NERDataset(
            words_path=self.train_path_prefix + ".words",
            labels_path=self.train_path_prefix + ".ner",
            tokenizer=self.tokenizer,
            is_train=True,
            label_to_entity_type_index=self.label_to_entity_type_index,
            ablation_not_mask=self.ablation_not_mask,
            ablation_no_brackets=self.ablation_no_brackets,
            ablation_span_type_together=self.ablation_span_type_together,
        )
        train_dataset.prepare(negative_multiplier=self.negative_multiplier)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=train_dataset.collate_fn
        )
        return train_loader

    def val_dataloader(self):
        if self.val_dataset_obj is None:
            print("Access val dataloader")
            val_dataset = NERDataset(
                words_path=self.val_path_prefix + ".words",
                labels_path=self.val_path_prefix + ".ner",
                tokenizer=self.tokenizer,
                is_train=False,
                label_to_entity_type_index=self.label_to_entity_type_index,
                ablation_not_mask=self.ablation_not_mask,
                ablation_no_brackets=self.ablation_no_brackets,
                ablation_span_type_together=self.ablation_span_type_together,
            )
            val_dataset.prepare()
            self.val_dataset_obj = val_dataset

        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset_obj,
            batch_size=self.validation_batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.val_dataset_obj.collate_fn
        )
        return val_loader

    def test_dataloader(self):
        print("Access test dataloader")
        test_dataset = NERDataset(
            words_path=self.test_path_prefix + ".words",
            labels_path=self.test_path_prefix + ".ner",
            tokenizer=self.tokenizer,
            is_train=False,
            label_to_entity_type_index=self.label_to_entity_type_index,
            ablation_not_mask=self.ablation_not_mask,
            ablation_no_brackets=self.ablation_no_brackets,
            ablation_span_type_together=self.ablation_span_type_together,
        )
        test_dataset.prepare()
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.validation_batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=test_dataset.collate_fn
        )
        self.test_dataset_obj = test_dataset
        return test_loader

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("FFFNERModel")

        parser.add_argument(
            "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            default=32,
            type=int,
            metavar="N",
            help="mini-batch size (default: 256), this is the total batch size of all GPUs on the current node"
                 " when using Data Parallel or Distributed Data Parallel",
        )
        parser.add_argument(
            "--validation_batch_size", default=128, type=int, help="validation batch size",
        )
        parser.add_argument(
            "--lr", "--learning-rate", default=2e-5, type=float, metavar="LR", help="initial learning rate", dest="lr"
        )
        parser.add_argument("--pretrained_lm", default="bert-base-uncased", type=str)
        parser.add_argument("--dataset_name", default="conll2003")
        parser.add_argument("--train_path_suffix", default="few_shot_5_0",
                            type=str)
        parser.add_argument("--val_path_suffix", default="few_shot_5_0",
                            type=str)
        parser.add_argument("--test_path_suffix", default="test", type=str)

        parser.add_argument("--negative_multiplier", default=3., type=float)
        parser.add_argument("--dropout", default=0.1, type=float)

        parser.add_argument("--ablation_not_mask", action='store_true')
        parser.add_argument("--ablation_no_brackets", action='store_true')
        parser.add_argument("--ablation_span_type_together", action="store_true")

        return parent_parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.accelerator == "ddp":
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        filename="{epoch:02d}-{val_metric:.2f}",
        save_top_k=1,
        mode="max",
    )
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    model = FFFNERModel(**vars(args))
    model.set_outputdir(trainer.logger.log_dir)
    if not args.evaluate:
        trainer.fit(model)
    model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.set_outputdir(trainer.logger.log_dir)
    trainer.test(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument(
        "-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set"
    )
    parent_parser.add_argument("--seed", type=int, default=42, help="seed for initializing training.")
    parser = FFFNERModel.add_model_specific_args(parent_parser)

    parser.set_defaults(profiler="simple", deterministic=False,
                        max_epochs=30, check_val_every_n_epoch=1,
                        log_every_n_steps=10)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run_cli()

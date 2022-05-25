# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sentence prediction (classification) task."""
import dataclasses
from typing import List, Union, Optional
from collections import defaultdict

from absl import logging
import numpy as np
import orbit
from scipy import stats
from sklearn import metrics as sklearn_metrics
import tensorflow as tf

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from official.nlp.data import data_loader_factory
from official.nlp.modeling import models
from official.nlp.tasks import utils

from official.projects.fffner import fffner_classifier

METRIC_TYPES = frozenset(
    ['accuracy', 'matthews_corrcoef', 'pearson_spearman_corr'])


@dataclasses.dataclass
class FFFNerModelConfig(base_config.Config):
  """A classifier/regressor configuration."""
  num_classes_1: int = 0
  num_classes_2: int = 0
  use_encoder_pooler: bool = True
  encoder: encoders.EncoderConfig = encoders.EncoderConfig()


@dataclasses.dataclass
class FFFNerPredictionConfig(cfg.TaskConfig):
  """The model config."""
  # At most one of `init_checkpoint` and `hub_module_url` can
  # be specified.
  init_checkpoint: str = ''
  init_cls_pooler: bool = False
  hub_module_url: str = ''
  metric_type: str = 'accuracy'
  # Defines the concrete model config at instantiation time.
  model: FFFNerModelConfig = FFFNerModelConfig()
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()


@task_factory.register_task_cls(FFFNerPredictionConfig)
class FFFNerTask(base_task.Task):
  """Task object for sentence_prediction."""

  def __init__(self, params: cfg.TaskConfig, logging_dir=None, name=None):
    super().__init__(params, logging_dir, name=name)
    if params.metric_type not in METRIC_TYPES:
      raise ValueError('Invalid metric_type: {}'.format(params.metric_type))
    self.metric_type = params.metric_type
    self.label_field_1 = 'is_entity_label'
    self.label_field_2 = 'entity_type_label'

  def build_model(self):
    if self.task_config.hub_module_url and self.task_config.init_checkpoint:
      raise ValueError('At most one of `hub_module_url` and '
                       '`init_checkpoint` can be specified.')
    if self.task_config.hub_module_url:
      encoder_network = utils.get_encoder_from_hub(
          self.task_config.hub_module_url)
    else:
      encoder_network = encoders.build_encoder(self.task_config.model.encoder)
    encoder_cfg = self.task_config.model.encoder.get()
    if self.task_config.model.encoder.type == 'xlnet':
      assert False, "Not supported yet"
    else:
      return fffner_classifier.FFFNerClassifier(
        # encoder_network.inputs
          network=encoder_network,
          num_classes_1=self.task_config.model.num_classes_1,
          num_classes_2=self.task_config.model.num_classes_2,
          initializer=tf.keras.initializers.TruncatedNormal(
              stddev=encoder_cfg.initializer_range),
          use_encoder_pooler=self.task_config.model.use_encoder_pooler)

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    label_ids_1 = labels[self.label_field_1]
    label_ids_2 = labels[self.label_field_2]
    loss_1 = tf.keras.losses.sparse_categorical_crossentropy(
      label_ids_1, tf.cast(model_outputs[0], tf.float32), from_logits=True) #TODO
    loss_2 = tf.keras.losses.sparse_categorical_crossentropy(
      label_ids_2, tf.cast(model_outputs[1], tf.float32), from_logits=True)
    loss = loss_1 + loss_2

    if aux_losses:
      loss += tf.add_n(aux_losses)
    return tf_utils.safe_mean(loss)

  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for sentence_prediction task."""
    if params.input_path == 'dummy':

      def dummy_data(_):
        dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
        x = dict(
            input_word_ids=dummy_ids,
            input_mask=dummy_ids,
            input_type_ids=dummy_ids,
            cls_token_pos=tf.zeros((1, 1), dtype=tf.int32),
            span_start_pos=tf.ones((1, 1), dtype=tf.int32)
        )

        x[self.label_field_1] = tf.zeros((1, 1), dtype=tf.int32)
        x[self.label_field_2] = tf.zeros((1, 1), dtype=tf.int32)
        return x

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    return data_loader_factory.get_data_loader(params).load(input_context)

  def build_metrics(self, training=None):
    del training
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy_1'),
        tf.keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy_2'),
    ]
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    for metric in metrics:
      if metric.name == 'cls_accuracy_1':
        metric.update_state(labels[self.label_field_1], model_outputs[0])
      if metric.name == 'cls_accuracy_2':
        metric.update_state(labels[self.label_field_2], model_outputs[1])

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    # looks sus
    compiled_metrics.update_state(labels[self.label_field_1], model_outputs[0])
    compiled_metrics.update_state(labels[self.label_field_2], model_outputs[1])

  def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
    features, labels = inputs, inputs
    outputs = self.inference_step(features, model)
    loss = self.build_losses(
      labels=labels, model_outputs=outputs, aux_losses=model.losses)
    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    if model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics or []})
      logs.update({m.name: m.result() for m in model.metrics})
    logs.update({
      'sentence_prediction_1': outputs[0],
      'sentence_prediction_2': outputs[1],
      'labels_1':
        labels[self.label_field_1],
      'labels_2':
        labels[self.label_field_2],
      'id': labels['example_id'],
      'sentence_id': labels['sentence_id'],
      'span_start': labels['span_start'],
      'span_end': labels['span_end']
    })
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    # if self.metric_type == 'accuracy':
    #   return None
    if state is None:
      state = {'sentence_prediction_1': [], 'sentence_prediction_2': [], 'labels_1': [], 'labels_2': [], 'ids': [],
               'sentence_id': [], 'span_start': [], 'span_end': []}
    state['sentence_prediction_1'].append(
        np.concatenate([v.numpy() for v in step_outputs['sentence_prediction_1']],
                       axis=0))
    state['sentence_prediction_2'].append(
      np.concatenate([v.numpy() for v in step_outputs['sentence_prediction_2']],
                     axis=0))
    state['labels_1'].append(
        np.concatenate([v.numpy() for v in step_outputs['labels_1']], axis=0))
    state['labels_2'].append(
      np.concatenate([v.numpy() for v in step_outputs['labels_2']], axis=0))
    state['ids'].append(
      np.concatenate([v.numpy() for v in step_outputs['id']], axis=0))
    state['sentence_id'].append(
      np.concatenate([v.numpy() for v in step_outputs['sentence_id']], axis=0))
    state['span_start'].append(
      np.concatenate([v.numpy() for v in step_outputs['span_start']], axis=0))
    state['span_end'].append(
      np.concatenate([v.numpy() for v in step_outputs['span_end']], axis=0))
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    sentence_prediction_1 = np.concatenate(aggregated_logs['sentence_prediction_1'], axis=0)
    sentence_prediction_1 = np.reshape(sentence_prediction_1, (-1, self.task_config.model.num_classes_1))
    sentence_prediction_2 = np.concatenate(aggregated_logs['sentence_prediction_2'], axis=0)
    sentence_prediction_2 = np.reshape(sentence_prediction_2, (-1, self.task_config.model.num_classes_2))
    labels_1 = np.concatenate(aggregated_logs['labels_1'], axis=0)
    labels_1 = np.reshape(labels_1, -1)
    labels_2 = np.concatenate(aggregated_logs['labels_2'], axis=0)
    labels_2 = np.reshape(labels_2, -1)

    ids = np.concatenate(aggregated_logs['ids'], axis=0)
    ids = np.reshape(ids, -1)
    sentence_id = np.concatenate(aggregated_logs['sentence_id'], axis=0)
    sentence_id = np.reshape(sentence_id, -1)
    span_start = np.concatenate(aggregated_logs['span_start'], axis=0)
    span_start = np.reshape(span_start, -1)
    span_end = np.concatenate(aggregated_logs['span_end'], axis=0)
    span_end = np.reshape(span_end, -1)

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

    def softmax(x):
      x = np.array(x)
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum(axis=0)

    per_sid_results = defaultdict(list)
    for id, sent_id, sp_start, sp_end, is_entity_label, is_entity_logit, entity_type_label, entity_type_logit in zip(
            ids, sentence_id, span_start, span_end, labels_1, sentence_prediction_1, labels_2, sentence_prediction_2):
      if sent_id > 0:
        per_sid_results[sent_id].append(
          (sp_start, sp_end, is_entity_label, is_entity_logit, entity_type_label, entity_type_logit))
    ground_truth = []
    prediction_1 = []
    prediction_2 = []
    for key in sorted(list(per_sid_results.keys())):
      results = per_sid_results[key]
      gt_entities = []
      predictied_entities = []
      prediction_confidence = {}
      prediction_confidence_type = {}
      length = 0
      for span_start, span_end, ground_truth, prediction, ground_truth_type, prediction_type in results:
        if ground_truth == 1:
          gt_entities.append((span_start, span_end, ground_truth_type))
        if prediction[1] > prediction[0]:
          predictied_entities.append((span_start, span_end, np.argmax(prediction_type).item()))
        prediction_confidence[(span_start, span_end)] = max(softmax(prediction))
        prediction_confidence_type[(span_start, span_end)] = max(softmax(prediction_type))
        length = max(length, span_end)
      length += 1
      ground_truth.extend([(key, *x) for x in gt_entities])
      prediction_1.extend([(key, *x) for x in predictied_entities])
      resolved_predicted = resolve(length, predictied_entities, prediction_confidence)
      prediction_2.extend([(key, *x) for x in resolved_predicted])

    raw = get_p_r_f(gt, p1)
    resolved = get_p_r_f(gt, p2)
    return {
      "raw_f1": raw["f1"],
      "raw_precision": raw["precision"],
      "raw_recall": raw["recall"],
      "resolved_f1": resolved["f1"],
      "resolved_precision": resolved["precision"],
      "resolved_recall": resolved["recall"],
    }


  def initialize(self, model):
    """Load a pretrained checkpoint (if exists) and then train from iter 0."""
    ckpt_dir_or_file = self.task_config.init_checkpoint
    logging.info('Trying to load pretrained checkpoint from %s',
                 ckpt_dir_or_file)
    if ckpt_dir_or_file and tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      logging.info('No checkpoint file found from %s. Will not load.',
                   ckpt_dir_or_file)
      return

    pretrain2finetune_mapping = {
        'encoder': model.checkpoint_items['encoder'],
    }
    print(pretrain2finetune_mapping)
    if self.task_config.init_cls_pooler:
      # This option is valid when use_encoder_pooler is false.
      pretrain2finetune_mapping[
          'next_sentence.pooler_dense'] = model.checkpoint_items[
              'sentence_prediction.pooler_dense']
    ckpt = tf.train.Checkpoint(**pretrain2finetune_mapping)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

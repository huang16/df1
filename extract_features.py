# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Extract pre-computed feature vectors from BERT."""

# modify extract_features.py to be one step of data pre-process pipline
# need a conf-obj and a sentence list
# return a sentence-vector list

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re
import os
import pickle as cPickle
import time

from bert import modeling
from bert import tokenization
import tensorflow as tf
import numpy as np


class ExtractConf(object):
    # too poor to use tpu
    # hope update one day
    def __init__(self,
                 max_seq_length=128,
                 sentence_field=None,
                 id_field=None,
                 input_list=None,
                 label_dict=None,
                 checkpoint_folder=None,
                 do_lower_case=True,
                 batch_size=32,
                 layers='-1,-2,-3,-4',
                 use_one_hot_embeddings=False,
                 bert_folder='~/Models/chinese_wwm_ext_L-12_H-768_A-12/',
                 output_folder='./output/',
                 use_tpu=False,
                 master=None,
                 num_tpu_cores=8):
        self.max_seq_length = max_seq_length
        self.sentence_field = sentence_field
        self.id_field = id_field
        self.input_list = input_list
        self.label_dict=label_dict
        self.checkpoint_folder = checkpoint_folder
        self.do_lower_case = do_lower_case
        self.batch_size = batch_size
        self.layers = [int(x) for x in layers.split(",")]
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.bert_config_file = os.path.expanduser(bert_folder + 'bert_config.json')
        self.vocab_file = os.path.expanduser(bert_folder + 'vocab.txt')
        self.init_checkpoint = os.path.expanduser(bert_folder + 'bert_model.ckpt')
        self.output_folder = output_folder
        self.use_tpu = use_tpu
        self.master = master
        self.num_tpu_cores = num_tpu_cores


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.string),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_type_ids":
                tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {
            "unique_id": unique_ids,
        }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.unique_id))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_list, sentence_field='sentence', id_field=None):
    """Read a list of `InputExample`s from a list of input_list."""
    # what if input is a sentence, may put it in []
    # use id override unique_id if exist
    examples = []
    unique_id = 0
    for seq in input_list:
        line = tokenization.convert_to_unicode(seq[sentence_field])
        if not line:
            break
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        if id_field is not None:
            unique_id = seq[id_field]
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
        else:
            unique_id += 1
    print('len examples: %d'%len(examples))
    return examples


def extract(config):
    tf.logging.set_verbosity(tf.logging.INFO)
    layer_indexes = config.layers
    bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=config.vocab_file,
        do_lower_case=config.do_lower_case
    )
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=config.master,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=config.num_tpu_cores,
            per_host_input_for_training=is_per_host
        )
    )
    examples = read_examples(config.input_list, config.sentence_field, config.id_field)
    features = convert_examples_to_features(
        examples=examples,
        seq_length=config.max_seq_length,
        tokenizer=tokenizer
    )
    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=config.init_checkpoint,
        layer_indexes=layer_indexes,
        use_tpu=config.use_tpu,
        use_one_hot_embeddings=config.use_one_hot_embeddings
    )
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=config.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=config.batch_size
    )
    input_fn = input_fn_builder(
        features=features,
        seq_length=config.max_seq_length
    )
    now = time.localtime()
    output_filename = 'extract_features_time_%d_%d_%d_%d_size_%d' % (now.tm_mon,
                                                                     now.tm_mday,
                                                                     now.tm_hour,
                                                                     now.tm_min,
                                                                     len(features))
    with open(output_filename, 'wb') as output:
        all_feature = []
        invalid_key=0
        for result in estimator.predict(input_fn, yield_single_examples=True):
            unique_id = result['unique_id']
            if config.id_field is None:
                unique_id = int(unique_id)
            # feature = unique_id_to_feature[unique_id]
            output_feature = {'id': unique_id}
            sentence_features = []
            for (i, index) in enumerate(layer_indexes):
                sentence_features.append(result["layer_output_%d" % i])
            sentence_features = np.array(sentence_features)
            output_feature['matrix'] = sentence_features
            if config.label_dict is not None:
                try:
                    output_feature['label']=config.label_dict[unique_id]
                except KeyError:
                    output_feature['label'] =-1
                    invalid_key+=1
            all_feature.append(output_feature)
        print(len(all_feature))
        cPickle.dump(all_feature, output)
        print('invalid key:%d'%invalid_key)

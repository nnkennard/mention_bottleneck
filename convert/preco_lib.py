"""Resplit preco to make a test set."""

import json
import sys
import os
import random

import convert_lib

def get_records_from_preco_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()


def preprocess(data_dir):

  preco_orig_dir = os.path.join(data_dir, "original", "PreCo_1.0")
  preco_dir = os.path.join(data_dir, "original", "preco")
  
  convert_lib.create_dir(preco_dir)

  resplit_datasets = {}

  resplit_datasets[
      convert_lib.DatasetSplit.test] = get_records_from_preco_file(
      os.path.join(preco_orig_dir, "dev.jsonl"))

  temp_original_train = get_records_from_preco_file(
      os.path.join(preco_orig_dir, "train.jsonl"))
  random.seed(43)
  random.shuffle(temp_original_train)
  total_train = len(temp_original_train)
  boundary = int(0.8 * total_train)
  resplit_datasets[
      convert_lib.DatasetSplit.train] = temp_original_train[:boundary]
  resplit_datasets[
      convert_lib.DatasetSplit.dev] = temp_original_train[boundary:]

  for split_name, records in resplit_datasets.items():
    # Write resplit
    with open(os.path.join(preco_dir, split_name + ".jsonl"), 'w') as f:
      f.write("".join(records))
 

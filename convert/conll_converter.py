import convert_lib
import collections
import os
import re

CONLL = convert_lib.DatasetName.conll

CONLL_FIELD_MAP = {
  convert_lib.LabelSequences.WORD: 3,
  convert_lib.LabelSequences.POS: 4, 
  convert_lib.LabelSequences.PARSE: 5, 
  convert_lib.LabelSequences.SPEAKER: 9, 
  convert_lib.LabelSequences.COREF: -1,
}

def build_coref_span_map(coref_col, offset):
  span_starts = collections.defaultdict(list)
  complete_spans = []
  for i, orig_label in enumerate(coref_col):
    if orig_label == '-': # no coref label
      continue
    else:
      labels = orig_label.split("|") # split for multiple (nested) case
      for label in labels:
        if label.startswith("("): # Span start
          if label.endswith(")"): # Single-token span
            complete_spans.append((i, i, label[1:-1]))
          else:
            span_starts[label[1:]].append(i) # Register span start for later
        elif label.endswith(")"):
          ending_cluster = label[:-1] # Which cluster is ending here
          assert len(span_starts[ending_cluster]) in [1, 2]
          # Sometimes it's closing a nested span but apparently never more than
          # two levels for the same entity
          start_idx = span_starts[ending_cluster].pop(-1)
          # The one added latest is the match
          complete_spans.append((start_idx, i, ending_cluster))

  span_dict = collections.defaultdict(list)
  for start, end, cluster in complete_spans:
    span_dict[cluster].append((offset + start, offset + end))
    # offset is the token offset of the sentence within the document
  return span_dict

def split_parse_label(label):
  curr_chunk = ""
  chunks = []
  for c in label:
    if c in "()": # A chunk is everything up to a paren
      if curr_chunk:
        chunks.append(curr_chunk)
      curr_chunk = c
    else:
      curr_chunk += c
  chunks.append(curr_chunk)
  return chunks


def build_parse_span_map(parse_col, offset):
  span_starts = collections.defaultdict(list)
  stack = []
  label_map = {}
  for i, orig_label in enumerate(parse_col):
    labels = split_parse_label(orig_label) # Chunking around parens
    for label in labels:
      if label.startswith("("): # Register start of a label
        stack.insert(0, [label, i + offset]) # Goes on the top of the stack
        # ^ build up label in [0], remember start (with offset) in [1]
      elif label.endswith(")"): # End of chunk, hopefully start was registered
        span_prefix, start_idx = stack.pop(0)
        assert (span_prefix, i) not in label_map # This is an unclosed span
        label_map[
            (start_idx, i + offset)] = span_prefix + label # Label is suffix
      else:
        stack[0][0] += label # This is part of the label we're currently collecting

  return label_map

def ldd_append(ldd, to_append):
  for k, v in to_append.items():
    ldd[k] += v
  return ldd

def get_lines_from_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()

MARKABLES_REGEX = r"\(NP\**\)"

POS_MARKABLES = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "PRP", "PRP$"]

def get_pos_markables(pos_sequence):
  pos_markables = []
  for i, pos in enumerate(pos_sequence):
    if pos in POS_MARKABLES:
      pos_markables.append((i, i))
  return pos_markables

def add_sentence(curr_doc, curr_sent, doc_coref_map, doc_parse_map,
                 sentence_offset):
  curr_doc.speakers.append(curr_sent[convert_lib.LabelSequences.SPEAKER])
  curr_doc.sentences.append(curr_sent[convert_lib.LabelSequences.WORD])
  curr_doc.pos.append(curr_sent[convert_lib.LabelSequences.POS])

  coref_span_map = build_coref_span_map(
      curr_sent[convert_lib.LabelSequences.COREF], sentence_offset)
  doc_coref_map = ldd_append(doc_coref_map, coref_span_map)

  parse_span_map = build_parse_span_map(
      curr_sent[convert_lib.LabelSequences.PARSE], sentence_offset)
  doc_parse_map = ldd_append(doc_parse_map, parse_span_map)

  coref_spans = sum(coref_span_map.values(), [])
  singletons = [span
      for span, label in parse_span_map.items() if (
        re.match(MARKABLES_REGEX, label) and span not in coref_spans)]
  singletons += get_pos_markables(curr_sent[convert_lib.LabelSequences.POS])
  singletons = list(set(singletons))
  # For now, singletons are NPs, PRPs and VBs that aren't in clusters

  sentence_offset += len(curr_sent[convert_lib.LabelSequences.WORD])

  return doc_coref_map, doc_parse_map, sentence_offset


def create_dataset(filename, field_map):
 
  dataset = convert_lib.Dataset(CONLL)

  document_counter = 0
  sentence_offset = 0

  curr_doc = None
  curr_doc_id = None
  curr_sent = collections.defaultdict(list)
  doc_coref_map = collections.defaultdict(list)
  doc_parse_map = collections.defaultdict(list)

  for line in get_lines_from_file(filename):

    if line.startswith("#begin"):
      assert curr_doc is None 
      curr_doc_id = line.split()[2][1:-2].replace("/", "-")
      part = str(int(line.split()[-1]))
      curr_doc = convert_lib.Document(curr_doc_id, part)
      sentence_offset = 0
    
    elif line.startswith("#end"):
      curr_doc.clusters = list(doc_coref_map.values())
      dataset.documents.append(curr_doc)
      doc_coref_map = collections.defaultdict(list)
      doc_parse_map = collections.defaultdict(list)
      curr_doc = None

    elif not line.strip():
      if curr_sent:
        doc_coref_map, doc_parse_map, sentence_offset = add_sentence(
          curr_doc, curr_sent, doc_coref_map, doc_parse_map, sentence_offset)
        curr_sent = collections.defaultdict(list)

    else: # Empty line signifies the end of a sentence
      fields = line.replace("/.", ".").split()
      for field_name, field_index in field_map.items():
        curr_sent[field_name].append(fields[field_index])

  return dataset


def convert(data_home):
  input_directory = os.path.join(data_home, "original", CONLL)
  output_directory = os.path.join(data_home, "processed", CONLL)
  convert_lib.create_dir(output_directory)
  conll_datasets = {}
  for split in convert_lib.DatasetSplit.ALL:
    input_filename = os.path.join(input_directory, "conll12_" + split + ".txt")
    converted_dataset = create_dataset(input_filename, CONLL_FIELD_MAP)
    convert_lib.write_converted(converted_dataset, output_directory + "/" + split)

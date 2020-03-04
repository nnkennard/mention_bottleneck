import conll_lib
import convert_lib

import os
import re
import sys
     
def main():

  data_home = sys.argv[1]

  for new_dataset_name in convert_lib.DatasetName.alternate_conlls:
    input_directory = os.path.join(
        data_home, "original", convert_lib.DatasetName.conll)
    output_directory = os.path.join(
        data_home, "original", new_dataset_name)
    convert_lib.create_dir(output_directory)

    for split in convert_lib.DatasetSplit.ALL:
      input_filename = os.path.join(input_directory, "conll12_" + split + ".txt")
      output_filename = os.path.join(output_directory, "conll12_" + split + ".txt")
      dataset = conll_lib.listify_conll_dataset(input_filename)
      fn = FN_MAP[new_dataset_name]
      new_dataset = conll_add_singletons(dataset, fn)
      write_conll_to_file(new_dataset, output_filename)

if __name__ == "__main__":
  main()

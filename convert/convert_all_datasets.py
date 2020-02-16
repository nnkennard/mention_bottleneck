import sys
import preco_converter
#import conll_converter


def main():
  data_home = sys.argv[1]
  preco_converter.convert(data_home) 

if __name__ == "__main__":
  main()

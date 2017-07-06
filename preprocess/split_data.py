'''
splits parallel data into train, dev, and test sets
example run:
$ python3 split_data.py /nfs/topaz/lcheung/data/europarl-de-en/europarl-v7.de-en.en /nfs/topaz/lcheung/data/europarl-de-en/europarl-v7.de-en.de
'''

import sys, re, random

def choose():
  TEST_PROB_LEVEL = 0.07
  DEV_PROB_LEVEL = 0.25
  if random.random() < TEST_PROB_LEVEL:                    
    return 2
  elif random.random() < DEV_PROB_LEVEL:                   
    return 1
  else:                                                    
    return 0

def main():
  assert(len(sys.argv) == 3)

  src_name = sys.argv[1]
  trg_name = sys.argv[2]

  in_src = open(src_name, 'r')
  in_trg = open(trg_name, 'r')

  out_src_train = open('%s.train' % src_name, 'w')
  out_src_dev = open('%s.dev' % src_name, 'w')
  out_src_test = open('%s.test' % src_name, 'w')

  out_trg_train = open('%s.train' % trg_name, 'w')
  out_trg_dev = open('%s.dev' % trg_name, 'w')
  out_trg_test = open('%s.test' % trg_name, 'w')

  for src, trg in zip(in_src, in_trg):
    choice = choose() 

    if choice is 0: # train
      out_src_train.write(src)
      out_trg_train.write(trg)
    elif choice is 1: # dev
      out_src_dev.write(src)
      out_trg_dev.write(trg)
    else: # test
      out_src_test.write(src)
      out_trg_test.write(trg)

if __name__ == '__main__':
  main()

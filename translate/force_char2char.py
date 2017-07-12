import argparse
import sys
import os
import time
import tempfile

reload(sys)
sys.setdefaultencoding('utf-8')

sys.path.insert(0, "/nfs/topaz/lcheung/code/dl4mt-c2c/char2char") # change appropriately

import numpy
import cPickle as pkl
from mixer import *

from nmt import pred_probs
from prepare_data import prepare_data
from data_iterator import TextIterator
from char_base import build_sampler, build_model, init_params

def main(model_dir, model_pkl, model_grads, dict_src, dict_trg, hyp_filename, 
        saveto, n_words_src, n_words):

  print 'Loading model.'

  model_file = os.path.join(model_dir, model_pkl)
  with open(model_file, 'rb') as f:
    model_options = pkl.load(f)

  param_file = os.path.join(model_dir, model_grads) 
  params = init_params(model_options)
  params = load_params(param_file, params)
  tparams = init_tparams(params)

  # load dictionary and invert
  with open(dict_src, 'rb') as f:
    word_dict = pkl.load(f)
  word_idict = dict()
  for kk, vv in word_dict.iteritems():
    word_idict[vv] = kk
  with open(dict_trg, 'rb') as f:
    word_dict_trg = pkl.load(f)
  word_idict_trg = dict()
  for kk, vv in word_dict_trg.iteritems():
    word_idict_trg[vv] = kk

  #temp_dir = os.path.dirname(hyp_filename)
  temp_dir = model_dir # TODO better solution for this 
  hyp_src_fname = os.path.join(temp_dir, '%s.src.tmp' % hyp_filename)
  hyp_trg_fname = os.path.join(temp_dir, '%s.trg.tmp' % hyp_filename)
 
  hyp_src = open(hyp_src_fname, 'w')
  hyp_trg = open(hyp_trg_fname, 'w')
  with open(hyp_filename, 'r') as f:
    for line in f:
      toks = line.strip().split('\t')
      hyp_src.write('%s\n' % toks[0].strip())
      hyp_trg.write('%s\n' % toks[1].strip())
  hyp_src.close()
  hyp_trg.close()

  test = TextIterator(source=hyp_src_fname,
                      target=hyp_trg_fname,
                      source_dict=dict_src,
                      target_dict=dict_trg,
                      n_words_source=n_words_src,
                      n_words_target=n_words,
                      source_word_level=0,
                      target_word_level=0,
                      batch_size=1,
                      sort_size=1) #?? dunno what this param does

  print 'Building model...\n',
  trng, use_noise, \
      x, x_mask, y, y_mask, \
      opt_ret, \
      cost = \
      build_model(tparams, model_options)
  inps = [x, x_mask, y, y_mask]

  '''
  # TODO maybe don't need this
  f_init, f_next = build_sampler(tparams, model_options, trng, use_noise)
  '''

  print 'Building f_log_probs...'
  f_log_probs = theano.function(inps, cost, profile=profile)
  use_noise.set_value(0.)

  test_scores = pred_probs(f_log_probs,
                           prepare_data,
                           model_options,
                           test,
                           5)
  print test_scores.mean()

  os.remove(hyp_src_fname)
  os.remove(hyp_trg_fname)

  test_scores = [ str(f) for f in test_scores ]

  with open(saveto, 'w') as f:
    f.write(u'\n'.join(test_scores).encode('utf-8'))
    f.write(u'\n')

  print "Done", saveto

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dir', type=str) # dir with models
    parser.add_argument('-model_pkl', type=str) # prefix name 
    parser.add_argument('-model_grads', type=str) # prefix name 
    parser.add_argument('-saveto', type=str, ) # absolute path where the translation should be saved
    parser.add_argument('-dict_src', type=str, default=None)
    parser.add_argument('-dict_trg', type=str, default=None)
    parser.add_argument('-hyp', type=str, default=None)
    parser.add_argument('-n_words_src', type=int, default=304, help="298 for FI-EN")
    parser.add_argument('-n_words', type=int, default=302, help="292 for FI-EN")

    args = parser.parse_args()

    print "src dict:", args.dict_src
    print "trg dict:", args.dict_trg
    print "source:", args.hyp
    print "dest :", args.saveto

    print args

    time1 = time.time()
    main(args.model_dir, args.model_pkl, args.model_grads, args.dict_src, 
         args.dict_trg, args.hyp, args.saveto,
         args.n_words_src, args.n_words)
    time2 = time.time()
    duration = (time2-time1)/float(60)
    print("Translation took %.2f minutes" % duration)

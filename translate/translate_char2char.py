import argparse
import sys
import os
import time

reload(sys)
sys.setdefaultencoding('utf-8')

sys.path.insert(0, "/nfs/topaz/lcheung/code/dl4mt-c2c/char2char") # change appropriately

import numpy
import cPickle as pkl
from mixer import *

def translate_model(jobqueue, resultqueue, model, options, k, normalize, build_sampler, gen_sample, init_params, model_id, silent):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    use_noise = theano.shared(numpy.float32(0.))
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    def _translate(seq):
        use_noise.set_value(0.)
        # sample given an input sequence and obtain scores
        # NOTE : if seq length too small, do something about it
        # beam size is 5 by default
        sample, score = gen_sample(tparams, f_init, f_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   options, trng=trng, k=k, maxlen=500,
                                   stochastic=False, argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample]) 
            score = score / lengths

        sidx = numpy.argmin(score)
        return sample[sidx]

    while jobqueue:
        req = jobqueue.pop(0)

        idx, x = req[0], req[1]
        if not silent:
            print "sentence", idx, model_id
        seq = _translate(x)
        #print 'Seq', seq, 'Score:', score

        resultqueue.append((idx, seq))
    return

def main(model, dictionary, dictionary_target, source_file, saveto, k=5,
         normalize=False, encoder_chr_level=False,
         decoder_chr_level=False, utf8=False, 
          model_id=None, silent=False):

    from char_base import (build_sampler, gen_sample, init_params)

    pkl_file = model.split('.')[0] + '.pkl'
    with open(pkl_file, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk


    # create input and output queues for processes
    jobqueue = []
    resultqueue = []

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            print cc
            for w in cc:
                if w == 0:
                    break
                if utf8:
                    ww.append(word_idict_trg[w].encode('utf-8'))
                else:
                    ww.append(word_idict_trg[w])
            if decoder_chr_level:
                capsw.append(''.join(ww))
            else:
                capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname):
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                # idx : 0 ... len-1 
                pool_window = options['pool_stride']

                if encoder_chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()

                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
                x = [2] + x + [3]

                # len : 77, pool_window 10 -> 3 
                # len : 80, pool_window 10 -> 0
                #rem = pool_window - ( len(x) % pool_window )
                #if rem < pool_window:
                #    x += [0]*rem

                while len(x) % pool_window != 0:
                    x += [0]

                x = [0]*pool_window + x + [0]*pool_window

                jobqueue.append((idx, x))

        return idx+1

    def _retrieve_jobs(n_samples, silent):
        trans = [None] * n_samples

        for idx in xrange(n_samples):
            resp = resultqueue.pop(0)
            trans[resp[0]] = resp[1] # (sequence, score)
            if numpy.mod(idx, 10) == 0:
                if not silent:
                    print 'Sample ', (idx+1), '/', n_samples, ' Done decoding using', model_id
        return trans

    print 'Translating ', source_file, '...'
    n_samples = _send_jobs(source_file)
    print "jobs sent"

    translate_model(jobqueue, resultqueue, model, options, k, normalize, build_sampler, gen_sample, init_params, model_id, silent)
    raw_outputs = _retrieve_jobs(n_samples, silent)
    trans = _seqs2words(raw_outputs)
    print "translations retrieved"

    with open(saveto, 'w') as f:
        f.write(u'\n'.join(trans).encode('utf-8'))
        f.write(u'\n')

    print "Done", saveto

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=20) # beam width
    parser.add_argument('-n', action="store_true", default=True) # normalize scores for different hypothesis based on their length (to penalize shorter hypotheses, longer hypotheses are already penalized by the BLEU measure, which is precision of sorts).
    parser.add_argument('-enc_c', action="store_true", default=True) # is encoder character-level?
    parser.add_argument('-dec_c', action="store_true", default=True) # is decoder character-level?
    parser.add_argument('-utf8', action="store_true", default=True)
    parser.add_argument('-model', type=str) # absolute path to a model (.npz file)
    parser.add_argument('-saveto', type=str, ) # absolute path where the translation should be saved
    parser.add_argument('-silent', action="store_true", default=False) # suppress progress messages
    parser.add_argument('-dict_src', type=str, default=None)
    parser.add_argument('-dict_trg', type=str, default=None)
    parser.add_argument('-source', type=str, default=None)

    args = parser.parse_args()

    char_base = args.model.split("/")[-1]

    print "src dict:", args.dict_src
    print "trg dict:", args.dict_trg
    print "source:", args.source
    print "dest :", args.saveto

    print args

    time1 = time.time()
    main(args.model, args.dict_src, args.dict_trg, args.source,
         args.saveto, k=args.k, normalize=args.n, encoder_chr_level=args.enc_c,
         decoder_chr_level=args.dec_c,
         utf8=args.utf8,
         model_id=char_base,
         silent=args.silent,
        )
    time2 = time.time()
    duration = (time2-time1)/float(60)
    print("Translation took %.2f minutes" % duration)

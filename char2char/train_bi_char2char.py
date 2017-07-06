import os
import sys
import argparse
import string
from collections import OrderedDict
from wmt_path import *
from char_base import *
from nmt import train
from conv_tools import *
from prepare_data import *

def main(job_id, args):
    source_dataset = args.src_train
    target_dataset = args.trg_train
    valid_source_dataset = args.src_dev
    valid_target_dataset = args.trg_dev
    source_dictionary = args.src_dict
    target_dictionary = args.trg_dict

    print args.model_path, args.model_name
    print source_dataset
    print target_dataset
    print valid_source_dataset
    print valid_target_dataset
    print source_dictionary
    print target_dictionary
    validerr = train(
        highway=args.highway,

        max_epochs=args.max_epochs,
        patience=args.patience,

        dim_word_src=args.dim_word_src,
        dim_word=args.dim_word,

        conv_width=args.conv_width,
        conv_nkernels=args.conv_nkernels,

        pool_window=args.pool_window,
        pool_stride=args.pool_stride,

        model_path=args.model_path,
        save_file_name=args.model_name,
        re_load=args.re_load,
        re_load_old_setting=args.re_load_old_setting,

        enc_dim=args.enc_dim,
        dec_dim=args.dec_dim,

        n_words_src=args.n_words_src,
        n_words=args.n_words,
        decay_c=args.decay_c,
        lrate=args.learning_rate,
        optimizer=args.optimizer,
        maxlen=args.maxlen,
        maxlen_trg=args.maxlen_trg,
        maxlen_sample=args.maxlen_sample,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        sort_size=args.sort_size,
        validFreq=args.validFreq,
        dispFreq=args.dispFreq,
        saveFreq=args.saveFreq,
        sampleFreq=args.sampleFreq,
        pbatchFreq=args.pbatchFreq,
        clip_c=args.clip_c,

        datasets=[source_dataset, target_dataset],
        valid_datasets=[valid_source_dataset, valid_target_dataset],
        dictionaries=[source_dictionary, target_dictionary],

        dropout_gru=args.dropout_gru,
        dropout_softmax=args.dropout_softmax,
        source_word_level=args.source_word_level,
        target_word_level=args.target_word_level,
        save_every_saveFreq=1,
        use_bpe=0,
        quit_immediately=args.quit_immediately,
        init_params=init_params,
        build_model=build_model,
        build_sampler=build_sampler,
        gen_sample=gen_sample,
        prepare_data=prepare_data,
    )
    return validerr

if __name__ == '__main__':

    import sys, time

    parser = argparse.ArgumentParser()
    # parser.add_argument('-translate', type=str, default="de_en", help="de_en / cs_en / fi_en / ru_en")
    parser.add_argument('-src_train', required=True, type=str, default=None, help='train src file')
    parser.add_argument('-src_dev', required=True, type=str, default=None, help='dev src file')
    parser.add_argument('-trg_train', required=True, type=str, default=None, help='train dev file')
    parser.add_argument('-trg_dev', required=True, type=str, default=None, help='dev trg file')
    parser.add_argument('-src_dict', required=True, type=str, default=None, help='src dict file')
    parser.add_argument('-trg_dict', required=True, type=str, default=None, help='trg dict file')
    parser.add_argument('-model_path', required=True, type=str, default=None, help='dir of model')

    parser.add_argument('-highway', type=int, default=4)

    parser.add_argument('-conv_width', type=str, default="1-2-3-4-5-6-7-8")
    parser.add_argument('-conv_nkernels', type=str, default="200-200-250-250-300-300-300-300")

    parser.add_argument('-pool_window', type=int, default=5)
    parser.add_argument('-pool_stride', type=int, default=5)

    parser.add_argument('-enc_dim', type=int, default=512)
    parser.add_argument('-dec_dim', type=int, default=1024)

    parser.add_argument('-dim_word', type=int, default=512)
    parser.add_argument('-dim_word_src', type=int, default=128)

    parser.add_argument('-batch_size', type=int, default=64, help="")
    parser.add_argument('-valid_batch_size', type=int, default=64, help="")

    parser.add_argument('-dropout_gru', type=int, default=0, help="")
    parser.add_argument('-dropout_softmax', type=int, default=0, help="")

    parser.add_argument('-maxlen', type=int, default=450, help="")
    parser.add_argument('-maxlen_trg', type=int, default=500, help="")
    parser.add_argument('-maxlen_sample', type=int, default=500, help="")

    parser.add_argument('-re_load', action="store_true", default=False)
    parser.add_argument('-re_load_old_setting', action="store_true", default=False)
    parser.add_argument('-quit_immediately', action="store_true", default=False, help="if true, will not proceed training, only print the size of the model.")

    parser.add_argument('-max_epochs', type=int, default=1000000000000, help="")
    parser.add_argument('-patience', type=int, default=-1, help="")
    parser.add_argument('-learning_rate', type=float, default=0.0001, help="")

    parser.add_argument('-n_words_src', type=int, default=304, help="298 for FI-EN")
    parser.add_argument('-n_words', type=int, default=302, help="292 for FI-EN")

    parser.add_argument('-optimizer', type=str, default="adam", help="")
    parser.add_argument('-decay_c', type=int, default=0, help="")
    parser.add_argument('-clip_c', type=int, default=1, help="")

    parser.add_argument('-saveFreq', type=int, default=5000, help="")
    parser.add_argument('-sampleFreq', type=int, default=5000, help="")
    parser.add_argument('-dispFreq', type=int, default=1000, help="")
    parser.add_argument('-validFreq', type=int, default=5000, help="")
    parser.add_argument('-pbatchFreq', type=int, default=5000, help="")
    parser.add_argument('-sort_size', type=int, default=20, help="")

    parser.add_argument('-source_word_level', type=int, default=0, help="")
    parser.add_argument('-target_word_level', type=int, default=0, help="")

    args = parser.parse_args()

    args.model_name = "bi-char2char"

    args.conv_width = [ int(x) for x in args.conv_width.split("-") ]
    args.conv_nkernels = [ int(x) for x in args.conv_nkernels.split("-") ]

    print "Model path:", args.model_path

    print args
    main(0, args)

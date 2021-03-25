"""

    Main script for inferring correspondences across domains by using the
    Gromov-Wasserstein distanceself.

    Parts of the machinery to load / evaluate word embeddings where built upon
    the very thorough codebase by Artetxe https://github.com/artetxem

"""
import sys
import os
import argparse
import collections
from collections import defaultdict
from time import time
import pickle

import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pylab as plt

import pdb
import argparse
import ot

from src.bilind import custom_gromov_bilind
import src.embeddings as embeddings
# import numpy.linalg.norm as norm
from numpy.linalg import norm as norm

def dump_results(outdir, args, optim_args, acc, BLI):
    results = {'acc': acc, 'args': vars(args), 'optim_args':  vars(optim_args), 'G': BLI.coupling}
    if BLI.mapping is not None:
        results['P'] = BLI.mapping
    np.save(os.path.join(outdir, "coupling"), BLI.coupling)
    dump_file = os.path.join(outdir, "results.pkl")
    pickle.dump(results, open(dump_file, "wb"))

def load_results(outdir, BLI):
    dump_file = os.path.join(outdir, "results.pkl")
    results = pickle.load(open(dump_file, "rb"))
    BLI.mapping  = results['P']
    BLI.coupling = results['G']
    return BLI

def parse_args():
    parser = argparse.ArgumentParser(description='Word embedding alignment with GW',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    ### General Task Options
    general = parser.add_argument_group('General task options')
    general.add_argument('--debug',action='store_true',
                    help='trigger debugging mode (saving to /tmp/)')
    general.add_argument('--data_dir', type=str, default='data/raw',
                         help='where word embedding data is located (i.e. path to MUSE/data dir)')
    general.add_argument('--load', action='store_true',
                         help='load previously trained model')
    general.add_argument('--encoding', type=str,
                         default='utf-8', help='embedding encoding')
    general.add_argument('--maxs', type=int, default=2000,
                         help='use only first k embeddings from source [default: 2000]')
    general.add_argument('--maxt', type=int, default=2000,
                         help='use only first k embeddings from target [default: 2000]')
    general.add_argument('--distribs', type=str, default='uniform',
                         help='p/q distributions to use [default: uniform]')
    general.add_argument('--normalize_vecs', type=str, default='both',
                         choices=['mean','both','whiten','whiten_zca','none'], help='whether to normalize embeddings')
    general.add_argument('--score_type', type=str, default='coupling', choices=[
                       'coupling','transported','distance'], help='what variable to use as the basis for translation scores')
    general.add_argument('--adjust', type=str, default='none', choices=[
                       'csls','isf','none'], help='What type of neighborhood adjustment to use')
    general.add_argument('--maxiter', type=int, default=1000, help='Max number of iterations for optimization problem')


    #### PATHS
    general.add_argument('--det_gts', type=str, required=True,
                    help='detection result pkl file')
    general.add_argument('--chkpt_path', type=str,
                         default='checkpoints', help='where to save the snapshot')
    general.add_argument('--results_path', type=str, default='out',
                         help='where to dump model config and epoch stats')
    general.add_argument('--log_path', type=str, default='log',
                         help='where to dump training logs  epoch stats (and config??)')
    general.add_argument('--summary_path', type=str, default='results/summary.csv',
                         help='where to dump model config and epoch stats')

    ### SAVING AND CHECKPOINTING
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency during train (in iters)')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='checkpoint save frequency during train (in  iters)')
    parser.add_argument('--plot_freq', type=int, default=100,
                        help='plot frequency during train (in  iters)')

    #############   Gromov-specific Optimization Args ###############
    gromov_optim = parser.add_argument_group('Gromov Wasserstein Optimization options')

    gromov_optim.add_argument('--metric', type=str, default='cosine', choices=[
                           'euclidean', 'sqeuclidean', 'cosine'], help='metric to use for computing vector distances')
    gromov_optim.add_argument('--normalize_dists', type=str, default='mean', choices = ['mean','max','median','none'],
                       help='method to normalize distance matrices')
    gromov_optim.add_argument('--no_entropy', action='store_false', default=True, dest='entropic',
                       help='do not use entropic regularized Gromov-Wasserstein')
    gromov_optim.add_argument('--entreg', type=float, default=5e-4,
                       help='entopy regularization for sinkhorn')
    gromov_optim.add_argument('--tol', type=float, default=1e-8,
                       help='stop criterion tolerance for sinkhorn')
    gromov_optim.add_argument('--gpu', action='store_true',
                       help='use CUDA/GPU for sinkhorn computation')

    args = parser.parse_args()

    if args.debug:
        args.verbose      = True
        args.chkpt_path   = '/tmp/'
        args.results_path = '/tmp/'
        args.log_path     = '/tmp/'
        args.summary_path = '/tmp/'

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))


    optimp = gromov_optim
    optimp.normalize_vecs  = args.normalize_vecs
    optimp.normalize_dists = args.normalize_dists

    optim_args = argparse.Namespace(
        **{a.dest: getattr(args, a.dest, None) for a in optimp._group_actions})
    data_args = argparse.Namespace(
        **{a.dest: getattr(args, a.dest, None) for a in general._group_actions})

    return args, optim_args

def make_path(root, args):
    if root is None:
        return None
    topdir = '_'.join([str(args.maxs)])
    method = 'gromov'
    params = {  # Subset of parameters to put in filename
                'entreg': 'ereg',
                'tol': 'tol',
    }
    subdir = [method, args.normalize_vecs, args.metric, args.distribs]
    for arg, name in params.items():
        val = getattr(args, arg)
        subdir.append(params[arg] + '_' + str(val))
    subdir = '_'.join(subdir)
    path = os.path.join(root, topdir, subdir)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def print_header(method):
    print('='*80)
    print('='*13 +'  Bilingual Lexical Induction with Gromov-Wasserstein  ' +'='*12)
    print('='*80)

def init_data():
    sigma = 1.0
    xs, xt = [], []
    s_label, t_label = [], []
    for i in range(300):
        mu = i*0.3
        vs = np.random.normal(mu, sigma, 2048)
        vt = np.random.normal(mu, sigma, 2048)
        xs.append(vs)
        xt.append(vt)
        s_label.append(i)
        t_label.append(i)

    xs, xt = np.array(xs), np.array(xt)

    return xs, xt, s_label, t_label

def main():
    """

        Pass outpath=checkpoints/bla to solve()
        Save progress plots, and current G and P in a pkl file:
        (it, G_t, P_t, lambda_G, lambda_P, ent_reg, ....)

        Add restarting from checkpoint:

        Saving to out:
            - history plot
            - final scores
            - tranlsations?
            - model? Popt Gopt
    """
    args, optim_args = parse_args()
    if args.det_gts:
        xs, xt = [], []
        s_label, t_label = [], []
        detect_result = pickle.load(open(args.det_gts, "rb" ))
        for idx, image_id in enumerate(detect_result.keys()):
            current_xs, current_xt = [], []
            current_s_label, current_t_label = [], []
            results = detect_result[image_id]
            if image_id == "train_0027_004035_004":
                continue
            # print(image_id)
            # print(results)
            for result in results:
                if result["cls"] == "body":
                    current_xs.append(result["feature"])
                    current_s_label.append(result["body_id"])
                else:
                    current_xt.append(result["feature"])
                    current_t_label.append(result["body_id"])

            if len(current_t_label) == 0:
                continue
            
            # sort accord to label
            current_t_label = np.array(current_t_label)
            current_xt = np.array(current_xt)
            inds = current_t_label.argsort()
            sorted_cur_xt = current_xt[inds]
            sorted_t_label = current_t_label[inds]

            sorted_cur_xs = []
            sorted_s_label = []
            used_index = np.zeros(shape=len(current_xs), )
            for each_index in sorted_t_label:
                _tmp_idx = np.where(current_s_label == each_index)
                tmp_idx = _tmp_idx[0][0]
                sorted_cur_xs.append(current_xs[tmp_idx])
                sorted_s_label.append(each_index)
                used_index[tmp_idx] = 1.0

            unused_feature = []
            for tmp_idx, each_index in enumerate(used_index):
                if each_index == 1.0:
                    continue
                unused_feature.append(current_xs[tmp_idx])

            for each_unused_feature in unused_feature:
                min_distance = 1000.0
                this_index = -5.0
                for u_idx, (label, each_used_feature) in enumerate(zip(sorted_s_label, sorted_cur_xs)):
                    distance = norm(each_unused_feature-each_used_feature)
                    if distance < min_distance:
                        min_distance = distance
                        this_index = label
                tmp_idx = np.where(sorted_t_label==this_index)[0][0]
                closest_target_feature = sorted_cur_xt[tmp_idx]
                fake_closest_target_feature = np.random.normal(0.0, 0.01, 1024) + closest_target_feature
                fake_closest_target_feature = np.expand_dims(fake_closest_target_feature, axis=0)
                sorted_cur_xs.append(each_used_feature)
                sorted_cur_xt = np.concatenate((sorted_cur_xt, fake_closest_target_feature), axis=0)
                sorted_s_label.append(-1.0)
                sorted_t_label = np.append(sorted_t_label, -1.0)
            
            sorted_cur_xs = np.array(sorted_cur_xs)
            # offset label
            current_offset = len(xs)

            sorted_s_label = [x+current_offset+1 for x in sorted_s_label]
            sorted_t_label = [x+current_offset+1 for x in sorted_t_label]
            sorted_s_label = np.array(sorted_s_label)
            sorted_t_label = np.array(sorted_t_label)

            if isinstance(xs, list) or isinstance(xt, list):
                xs, xt = sorted_cur_xs, sorted_cur_xt
            else:
                xs = np.concatenate((xs, sorted_cur_xs), axis=0)
                xt = np.concatenate((xt, sorted_cur_xt), axis=0)

            print("xs.shape, xt.shape: ", xs.shape, xt.shape)
            s_label = np.concatenate((s_label, sorted_s_label), axis=0)
            t_label = np.concatenate((t_label, sorted_t_label), axis=0)

    else:
        # Read Word Embeddings

        xs, xt, s_label, t_label = init_data()

    outdir   = make_path(args.results_path, args)
    chkptdir = make_path(args.chkpt_path, args)

    print('Saving checkpoints to: {}'.format(chkptdir))
    print('Saving results to: {}'.format(outdir))

    BLI = custom_gromov_bilind(xs, xt, s_label, t_label, 
                        metric = args.metric, normalize_vecs = args.normalize_vecs,
                        normalize_dists = args.normalize_dists,
                        score_type = args.score_type, adjust = args.adjust,
                        distribs = args.distribs)
    BLI.init_optimizer(**vars(optim_args)) # FIXME: This is ugly. Get rid of it

    if (not args.load) or (not os.path.exists(os.path.join(outdir, "results.pkl"))):
        if args.load:
            print('Could not load!!!')
        print('Will train from scratch')
        start = time()
        BLI.fit(maxiter=args.maxiter, plot_every=args.plot_freq,
                print_every=args.print_freq, verbose=True, save_plots = outdir)
        plt.close('all')
        print('Total elapsed time: {}s'.format(time() - start))
        if outdir:
            BLI.solver.plot_history(save_path=os.path.join(outdir, 'history.pdf'))
            acc = 0
            print("saving outdir")
            dump_results(outdir, args, optim_args, acc, BLI)
        else:
            BLI.solver.plot_history()
        print('Done!')
    else:
        print('Will load pre-solved solution from: ', outdir)
        BLI = load_results(outdir, BLI)


    ### Infer mapping from coupling - there's many ways to do this.
    print(BLI.mapping.shape)
    acc_file = os.path.join(outdir, 'accuracies.tsv')
    acc_dict = {}
    print('Results on test dictionary for fitting vectors: (via bary projection)')
    acc_dict['bary'] = BLI.test_accuracy(verbose=True, score_type = 'barycentric')

    if outdir:
        # print('Saving accuacy results')
        # with open(acc_file, 'w') as f:
        #     for k,acc in acc_dict.items():
        #         f.write('\t'.join([k] + ['{:4.2f}'.format(100*v) for v in acc.values()]) + '\n')
        # print('Saving in-vocabulary translations and mapped vectors')
        # translation_file = os.path.join(outdir, "translations_transductive.tsv")
        # BLI.dump_translations(src2trg, translation_file)
        #BLI.export_mapped(iov_mode='matched', outf=outdir, suffix = 'match') # Only needed for debug/analysis purposes
        BLI.export_mapped(iov_mode='projection', outf=outdir, suffix = 'proj')


    ### STEP 2: Out-of sample vectors
    print('************')
    print('Compute now for all vectors')
    # Read Word Embeddings
    argsc = argparse.Namespace(**vars(args))
    argsc.maxs = max(200000, args.maxs)
    argsc.maxt = max(200000, args.maxt)

    #BLI.normalize_embeddings()

    print('Projecting and dumping....')
    if BLI.mapping is not None:
        BLI.export_mapped(iov_mode='projection', outf=outdir, suffix = 'proj-proj')
        #BLI.export_mapped(iov_mode='barycentric', outf=outdir, suffix = 'bary-proj')
        #BLI.export_mapped(iov_mode='matched', outf=outdir, suffix = 'match-proj')
        print('Done!')


if __name__ == "__main__":
    main()

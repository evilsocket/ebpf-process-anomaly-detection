#!/usr/bin/env python3
# 
# Copyleft Simone 'evilsocket' Margaritelli
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import argparse


def learn_action(args):
    import os
    import time

    from lib import MAX_SYSCALLS
    from lib.ebpf import Probe

    if os.path.exists(args.data):
        print("%s already exists" % args.data)
        quit()
    
    probe = Probe(args.pid)

    print("learning from process %d (%s), saving data to %s ..." % (args.pid, probe.comm, args.data))

    # prepare histogram per cpu array with sum reducer
    histo_map = probe.start()
    prev = [0.0] * MAX_SYSCALLS
    prev_t = 0
    samples = 0

    with open(args.data, 'w+t') as fp:
        # write csv header
        fp.write('sample_time,%s\n' % ','.join(['sys_%d' % s for s in range(0, MAX_SYSCALLS)]))
        # polling
        while 1:
            # get single histogram from per-cpu arrays
            histogram = [histo_map[s] for s in range(0, MAX_SYSCALLS)]
            # if any change happened
            if histogram != prev:
                # compute the rate of change for every syscall
                deltas = [ 1.0 - (prev[s] / histogram[s]) if histogram[s] != 0.0 else 0.0 for s in range(0, MAX_SYSCALLS)]
                prev = histogram
                # append to csv file
                fp.write("%f,%s\n" % (time.time(), ','.join(map(str, deltas))))
                fp.flush()
                samples += 1

            # report number of samples saved to file every second
            now = time.time()
            if now - prev_t >= 1.0:
                print("%d samples saved ..." % samples)
                prev_t = now
            
            time.sleep(args.time / 1000.0)

def train_action(args):
    from lib.ml import AutoEncoder

    ae = AutoEncoder(args.model)
    # train the autoencoder and get error threshold
    (_, threshold) = ae.train(args.data, args.epochs, args.batch_size)

    print('error threshold=%f' % threshold)

def run_action(args):
    import time
    from collections import Counter

    from lib import MAX_SYSCALLS
    from lib.ebpf import Probe
    from lib.ml import AutoEncoder
    from lib.platform import SYSCALLS

    probe = Probe(args.pid)
    ae = AutoEncoder(args.model, load=True)

    print("monitoring process %d (%s) ..." % (args.pid, probe.comm))

    # prepare histogram per cpu array with sum reducer
    histo_map = probe.start()
    prev = [0.0] * MAX_SYSCALLS

    # polling
    while 1:
        # get single histogram from per-cpu arrays
        histogram = [histo_map[s] for s in range(0, MAX_SYSCALLS)]
        # if any change happened
        if histogram != prev:
            # compute the rate of change for every syscall
            deltas = [ 1.0 - (prev[s] / histogram[s]) if histogram[s] != 0.0 else 0.0 for s in range(0, MAX_SYSCALLS)]
            prev = histogram
            # run the model and get per-feature error and cumulative error
            (_, feat_errors, error) = ae.predict([deltas])  
            # if cumulative error is greater than the threshold we have an anomaly
            if error > args.max_error:
                print("error = %f - max = %f - top 3:" % (error, args.max_error))
                # get top 3 per-feature anomalies
                errors = {idx: err for idx, err in enumerate(feat_errors)}
                k = Counter(errors)
                top3 = k.most_common(3)
                for (idx, err) in top3:
                    name = SYSCALLS.get(idx, 'syscall_%d' % idx)
                    print("  %s = %f" % (name, err))
            
        time.sleep(args.time / 1000.0)
        

parser = argparse.ArgumentParser(description='Process activity anomaly detection with eBPF and unsupervised learning.')

actions = parser.add_mutually_exclusive_group()
actions.add_argument('--learn', action='store_true', help='start capturing data from a running process for later training')
actions.add_argument('--train', action='store_true', help='train a model on previously captured data')
actions.add_argument('--run', action='store_true', help='run a trained model with a running process')

parser.add_argument("--pid", type=int, default=0, help="target pid")
parser.add_argument("--data", default="data.csv", help="activity csv file")
parser.add_argument("--model", default="model.h5", help="model file")
parser.add_argument("--time", type=int, default=100, help="polling time in milliseconds")
parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=16, help="training batch size")
parser.add_argument("--max-error", type=float, default=0.09, help="autoencoder error threshold")

args = parser.parse_args()

# default to learn action
if not args.learn and not args.train and not args.run:
    args.learn = True

if args.learn:
    learn_action(args)
elif args.train:
    train_action(args)
elif args.run:
    run_action(args)

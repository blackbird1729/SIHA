from synthetic_sim import ChargedParticlesSim, SpringSim,SpringSimrd
import time
import numpy as np
import argparse
import multiprocessing as mp
import os

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=200000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=50000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=50000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=10000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=10,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--num-workers', type=int, default=16,
                    help='Number of worker processes to use.')

args = parser.parse_args()
np.random.seed(args.seed)

if args.simulation == 'springs':
    sim_class = SpringSim
    suffix = '_springs'
elif args.simulation == 'charged':
    sim_class = ChargedParticlesSim
    suffix = '_charged'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls)
# suffix += 'rdbig'
print(f"Simulation suffix: {suffix}")


def run_single_sim(args_tuple):
    sim_class, n_balls, length, sample_freq, seed = args_tuple
    np.random.seed(seed)  # 每个任务独立的随机种子
    # n_balls = np.random.randint(5, 11)
    sim = sim_class(noise_var=0.0, n_balls=n_balls)
    loc, vel, edges = sim.sample_trajectory(T=length, sample_freq=sample_freq)
    return loc, vel, edges




def generate_dataset_parallel(num_sims, length, sample_freq, base_seed=0):
    print(f"Using {args.num_workers} worker(s) to generate {num_sims} simulations...")

    task_args = [
        (sim_class, args.n_balls, length, sample_freq, base_seed + i)
        for i in range(num_sims)
    ]

    with mp.Pool(processes=args.num_workers) as pool:
        results = pool.map(run_single_sim, task_args)

    loc_all, vel_all, edges_all = zip(*results)
    return np.stack(loc_all), np.stack(vel_all), np.stack(edges_all)




if __name__ == "__main__":
    print("Generating training simulations...")
    loc_train, vel_train, edges_train = generate_dataset_parallel(
        args.num_train, args.length, args.sample_freq, base_seed=args.seed)

    print("Generating validation simulations...")
    loc_valid, vel_valid, edges_valid = generate_dataset_parallel(
        args.num_valid, args.length, args.sample_freq, base_seed=args.seed + args.num_train)

    print("Generating test simulations...")
    loc_test, vel_test, edges_test = generate_dataset_parallel(
        args.num_test, args.length_test, args.sample_freq, base_seed=args.seed + args.num_train + args.num_valid)


    np.save('loc_train' + suffix + '.npy', loc_train)
    np.save('vel_train' + suffix + '.npy', vel_train)
    np.save('edges_train' + suffix + '.npy', edges_train)

    np.save('loc_valid' + suffix + '.npy', loc_valid)
    np.save('vel_valid' + suffix + '.npy', vel_valid)
    np.save('edges_valid' + suffix + '.npy', edges_valid)

    np.save('loc_test' + suffix + '.npy', loc_test)
    np.save('vel_test' + suffix + '.npy', vel_test)
    np.save('edges_test' + suffix + '.npy', edges_test)

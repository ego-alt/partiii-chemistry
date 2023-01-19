import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
import matplotlib.animation as animation
import pickle
import os

FILEDIR = "./output/"
FFMPEG_PATH = "/usr/local/bin/ffmpeg"


class Writer:
    def __init__(self, img_frames, ffmpeg_path=FFMPEG_PATH):
        self.ffmpeg_path = ffmpeg_path
        self.img_frames = img_frames
        self.first_frame = None

    def save_video(self, movie_path):
        fig, ax = plt.subplots()
        cmap = mc.ListedColormap(["black", "red", "blue", "yellow"])
        self.first_frame = ax.imshow(self.img_frames[0], cmap=cmap, interpolation='nearest', animated=True)
        if self.ffmpeg_path:
            plt.rcParams["animation.ffmpeg_path"] = self.ffmpeg_path
        anim = animation.FuncAnimation(fig, self.animate, frames=len(self.img_frames))
        anim.save(FILEDIR + movie_path, writer=animation.FFMpegWriter(fps=60))
        print("Animation complete")
        plt.close()

    def animate(self, frame):
        next_frame = self.img_frames[frame]
        self.first_frame.set_array(next_frame)
        return next_frame

    def save_state(self, pairings, pickle_path):
        state = {"N": self.img_frames[0].shape,
                 "parameters": pairings,
                 "evolution": self.img_frames}
        with open(FILEDIR + pickle_path, "wb") as file:
            pickle.dump(state, file)


def count_like_pairs(grid):
    # Calculates the difference ./. numbers of like pairs and unlike pairs
    leftshifted, downshifted = grid[:, 1:], grid[1:, :]
    horizontal_n = np.sum(grid[:, :-1] == leftshifted)  # Number of favourable left-right interactions
    vertical_n = np.sum(grid[:-1, :] == downshifted)  # Number of favourable up-down interactions
    return horizontal_n + vertical_n


class Reader:
    def __init__(self, pickle_path):
        self.pickle_path = pickle_path
        with open(self.pickle_path, "rb") as f:
            data = pickle.load(f)
        self.length, _ = data["N"]
        self.parameters = data["parameters"]
        self.time_evol = np.asarray(data["evolution"])
        self.time_steps = (len(self.time_evol) - 1) * 10 + 1
        self.cells = {0: "Dead", 1: "Type A", 2: "Type B", 3: "Type C"}
        self.total_n = 2 * (self.length - 1) * self.length  # Total no. of adjacent pairs in the system

    def populations(self):
        plt.title("Populations against time")
        plt.ylabel("Number of cells")
        plt.xlabel("Time steps")
        pop = list(map(self.count, self.cells.keys()))
        for ind, n in enumerate(pop):
            plt.plot(range(0, self.time_steps, 10), n, label=self.cells[ind])
        plt.legend()
        plt.savefig(f"{self.pickle_path.replace('.pickle', '')}.png")
        plt.clf()

    def energy(self):
        plt.title("System energy against time")
        plt.ylabel("Energy (J / kT)")
        plt.xlabel("Time steps")
        plt.plot(range(0, self.time_steps, 10), self.measure_energy())
        plt.savefig(f"{self.pickle_path.replace('.pickle', '')}_energy.png")
        plt.clf()

    def measure_energy(self):
        alive_or_dead = self.time_evol > 0
        energy = np.asarray([self.total_n - count_like_pairs(i) for i in alive_or_dead])
        return energy

    def count(self, i):
        n = [np.count_nonzero(arr == i) for arr in self.time_evol]
        return n

    def count_particles(self):
        n = np.asarray([np.count_nonzero(arr > 0) for arr in np.asarray(self.time_evol)])
        return n

    def count_coexistence(self, i):
        n = np.asarray([len(np.unique(frame)) == i for frame in self.time_evol])
        return n


def plot_crowdedness(dir_list, label_list, name):
    plt.title("Measure of crowdedness at different mobilities")
    plt.ylabel("$n_{empty neighbours}/N$")
    plt.xlabel("Time steps")
    for ind, dir in enumerate(dir_list):
        x, num = 0, 0
        for file in os.scandir(dir):
            file = os.path.join(dir, file.name)
            if ".pickle" in file:
                z = Reader(file)
                x += z.measure_energy() / z.count_particles()
                num += 1
        plt.plot(range(0, 10001, 10), x / num, label=label_list[ind])
        plt.legend()
    plt.savefig(name)


def plot_fraction(dir_list, label_list, time_steps, name, normalise=None):
    plt.title("Probability of coexistence between three species")
    plt.ylabel("Fraction of trials $f_{coexistence}$")
    plt.xlabel("Time steps")
    plt.ylim(0, 1)
    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for ind, dir in enumerate(dir_list):
        reduction = []
        for file in os.scandir(dir):
            file = os.path.join(dir, file.name)
            if ".pickle" in file:
                z = Reader(file)
                reduction.append(z.count_coexistence(4)[:time_steps])
        if normalise:  # Specify required labels and colors with [ind] here
            plt.xlim(0, 1)
            plt.plot(np.arange(0, time_steps) / time_steps, sum(reduction) / len(reduction))
        else:
            plt.plot(np.arange(0, time_steps), sum(reduction) / len(reduction))
            plt.legend()
    plt.savefig(name)


def plot_min(dir_list, label_list, name):
    plt.title("Population of the least common species over time")
    plt.ylabel("Number of particles")
    plt.xlabel("Time steps")
    for ind, dir in enumerate(dir_list):
        x, num = 0, 0
        for file in os.scandir(dir):
            file = os.path.join(dir, file.name)
            if ".pickle" in file:
                z = Reader(file)
                minimum = np.asarray([np.min(np.unique(i, return_counts=True)[1][1:])
                                      if len(np.unique(i, return_counts=True)[1]) == 4 else 0
                                      for i in z.time_evol])
                x += minimum
                num += 1
                # num += minimum > 0
        # num = np.where(num == 0, 1, num)
        plt.plot(range(0, 20001, 10), x / num, label=label_list[ind])
        plt.legend()
    plt.savefig(name)


def time_correlation(frame):
    pass


def spatial_correlation(frame, type):
    plt.title("$<r_{A,x} r_{A,x+d}>$ for species A")
    plt.ylabel("$<r_{A,x} r_{A,x+d}>$")
    plt.xlabel("Distance between cells $d$")
    # Pairwise distances between all type A particles
    N, N = frame.shape  # Always handling square lattices
    n = np.zeros(N)
    x_coord, y_coord = np.where(frame == type)
    a, b = np.tril_indices(len(x_coord), 1)
    x_dist = x_coord[a] - x_coord[b]
    y_dist = y_coord[a] - y_coord[b]
    # Count interactions between particles of type A in the same column
    vertical = np.where(x_dist == 0)
    tot_dist_v = np.sqrt(np.square(x_dist[vertical]) + np.square(y_dist[vertical]))
    remove = np.where(x_dist != 0)
    x_dist, y_dist = x_dist[remove], y_dist[remove]
    # Repeat the process for particles of type A in the same row (without double-counting)
    horizon = np.where(y_dist == 0)
    tot_dist_h = np.sqrt(np.square(x_dist[horizon]) + np.square(y_dist[horizon]))
    dist, count = np.unique(np.concatenate((tot_dist_h, tot_dist_v)), return_counts=True)
    n[dist.astype(int)] = count
    # Plot the average product as a function of distance
    plt.plot(range(0, N), n / length_count(N))


def length_count(N):
    a, b = np.tril_indices(N, N)
    z_total = np.zeros(N)
    for ind, x in enumerate(a):
        z = np.zeros(N)
        y = b[ind]
        z[:N - x] += 1
        z[:x + 1] += 1
        z[:N - y] += 1
        z[:y + 1] += 1
        z[0] = 1
        z_total += z
    z_total[1:] = z_total[1:] / 2
    return z_total


def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


"""
    def pairs(self):
        alive_or_dead = self.time_evol > 0
        identical_pairs = np.asarray([count_like_pairs(i) for i in np.asarray(self.time_evol)])
        similar_pairs = np.asarray([count_like_pairs(i) for i in alive_or_dead])

        plt.title("Number of interaction pairs against time")
        sel_pairs = similar_pairs - identical_pairs
        rep_pairs = self.total_n - similar_pairs
        plt.plot(range(self.time_steps + 1), rep_pairs, label="Reproduction")
        plt.plot(range(self.time_steps + 1), sel_pairs, label="Selection")
        plt.legend()
        plt.savefig(f"{self.pickle_path.replace('.pickle', '')}_pairs.png")
        plt.clf()
"""

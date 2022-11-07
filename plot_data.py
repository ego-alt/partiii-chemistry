import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import pickle

FILEDIR = "./output/"
FFMPEG_PATH = "/usr/local/bin/ffmpeg"


class Writer:
    def __init__(self, img_frames, ffmpeg_path=FFMPEG_PATH):
        self.ffmpeg_path = ffmpeg_path
        self.img_frames = img_frames
        self.first_frame = None

    def save_video(self, movie_path):
        fig, ax = plt.subplots()
        cmap = ListedColormap(["black", "red", "blue", "yellow"])
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
                 "time_steps": len(self.img_frames) - 1,
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

        self.cells = {0: "Dead", 1: "Type A", 2: "Type B", 3: "Type C"}
        self.length, _ = data["N"]
        self.time_steps = data["time_steps"]
        self.time_evol = np.asarray(data["evolution"])
        self.pop = list(map(self.count, self.cells.keys()))
        self.total_n = 2 * (self.length - 1) * self.length  # Total no. of adjacent pairs in the system

    def populations(self):
        plt.title("Populations against time")
        plt.ylabel("Number of cells")
        plt.xlabel("Time steps")
        for ind, n in enumerate(self.pop):
            plt.plot(range(self.time_steps + 1), n, label=self.cells[ind])
        plt.legend()
        plt.savefig(f"{self.pickle_path.replace('.pickle', '')}.png")
        plt.show()

    def count(self, i):
        n = [np.count_nonzero(arr == i) for arr in self.time_evol]
        return n

    def frequencies(self):
        fig, axs = plt.subplots(1, 4, figsize=(30, 10), tight_layout=True)
        for ind, n in enumerate(self.pop):
            axs[ind].hist(n)
            axs[ind].set(title=f"{self.cells[ind]}",
                         xlabel="Number of cells", ylabel="Frequency")
        fig.suptitle("Distribution of cell populations")
        fig.savefig(f"{self.pickle_path.replace('.pickle', '')}_freq.png")

    def energy(self):
        alive_or_dead = self.time_evol > 0
        energy = [self.total_n - 2 * count_like_pairs(i) for i in alive_or_dead]

        plt.title("System energy against time")
        plt.ylabel("Energy (J / kT)")
        plt.xlabel("Time steps")
        plt.plot(range(self.time_steps + 1), energy)
        plt.savefig(f"{self.pickle_path.replace('.pickle', '')}_energy.png")
        plt.show()

    def pairs(self):
        alive_or_dead = self.time_evol > 0
        identical_pairs = np.asarray([count_like_pairs(i) for i in self.time_evol])
        similar_pairs = np.asarray([count_like_pairs(i) for i in alive_or_dead])

        plt.title("Number of interaction pairs against time")
        sel_pairs = similar_pairs - identical_pairs
        rep_pairs = self.total_n - similar_pairs
        plt.plot(range(self.time_steps + 1), rep_pairs, label="Reproduction")
        plt.plot(range(self.time_steps + 1), sel_pairs, label="Selection")
        plt.legend()
        plt.savefig(f"{self.pickle_path.replace('.pickle', '')}_pairs.png")
        plt.show()

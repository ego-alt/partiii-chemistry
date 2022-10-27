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


class Reader:
    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        self.pickle_path = pickle_path
        self.cells = {0: "Dead", 1: "Type A", 2: "Type B", 3: "Type C"}
        self.time_steps = data["time_steps"]
        self.time_evol = data["evolution"]

    def populations(self):
        pop = list(map(self.count, self.cells.keys()))
        plt.title("Populations against time")
        plt.ylabel("Number of cells")
        plt.xlabel("Time steps")
        for ind, n in enumerate(pop):
            plt.plot(range(self.time_steps + 1), n, label=self.cells[ind])
        plt.legend()
        plt.savefig(f"{self.pickle_path.replace('.pickle', '')}.png")
        plt.show()

    def count(self, i):
        n = [np.count_nonzero(arr == i) for arr in self.time_evol]
        return n

    def energy(self):
        pass

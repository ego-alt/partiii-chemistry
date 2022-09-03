import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation


# Based on Reichenbach, Mobilia & Frey (2007)
# "Mobility promotes and jeopardizes biodiversity in rock–paper–scissors games"
class Grid:
    def __init__(self, n, max_steps):
        self.length = n
        self.cell_status = [1, 2, 3]  # 0=dead, 1=type A, 2=type B, 3=type C
        self.grid = self.initial_grid()
        self.max_steps = max_steps
        self.current_step_count = 0
        self.img_frames = []
        self.first_frame = None

        exchan_var = 0.8
        reprod_var = select_var = 1
        self.total_prob = exchan_var + reprod_var + select_var
        exchan_prob = self.probability(exchan_var)  # Average probability of exchange in the next time step dt
        reprod_prob = self.probability(reprod_var)  # Average probability of reproduction in the next time step dt
        select_prob = self.probability(select_var)  # Average probability of selection in the next time step dt

        self.diff_prob = [exchan_prob, reprod_prob, select_prob]
        self.dispatcher = [self.exchange, self.reproduction, self.selection]

    def call(self):
        # Reiterates each time step until the step limit has been reached
        while self.current_step_count < self.max_steps:
            self.one_time_step()
            self.current_step_count += 1
            print(self.current_step_count)

    def initial_grid(self):
        # Initialises the randomly populated grid
        grid = np.random.randint(self.cell_status[0], self.cell_status[-1] + 1, size=(self.length, self.length))
        return grid

    def probability(self, variable):
        # Calculates the probability of exchange, reproduction or selection
        probability = variable / self.total_prob
        return probability

    def one_time_step(self):
        reaction_count = 0
        num_cells = self.length * self.length
        while reaction_count < num_cells:
            # Randomly chooses cell coordinates
            x = random.randrange(len(self.grid))
            y = random.randrange(len(self.grid))
            cell_coord = [x, y]

            # Randomly chooses one of four neighbours (up, down, left, right)
            x_or_y = random.randint(0, 1)
            neighbour_coord = cell_coord.copy()
            neighbour_coord[x_or_y] = (neighbour_coord[x_or_y] + random.choice([-1, 1])) % self.length

            reaction = random.choices(self.dispatcher, weights=self.diff_prob)[0]
            reaction(cell_coord, neighbour_coord)
            reaction_count += 1

        self.img_frames.append(np.copy(self.grid))

    def exchange(self, cell_coord, neighbour_coord):
        # Swaps positions with a neighbouring cell
        x, y = cell_coord[0], cell_coord[1]
        new_x, new_y = neighbour_coord[0], neighbour_coord[1]
        self.grid[x][y], self.grid[new_x][new_y] = self.grid[new_x][new_y], self.grid[x][y]

    def selection(self, cell_coord, neighbour_coord):
        x, y = cell_coord[0], cell_coord[1]
        new_x, new_y = neighbour_coord[0], neighbour_coord[1]
        if self.grid[x][y]:
            cell_index = (self.cell_status.index(self.grid[x][y]) + 1) % len(self.cell_status)
            if self.grid[new_x][new_y] == self.cell_status[cell_index]:
                self.grid[new_x][new_y] = 0

    def reproduction(self, cell_coord, neighbour_coord):
        x, y = cell_coord[0], cell_coord[1]
        new_x, new_y = neighbour_coord[0], neighbour_coord[1]
        if self.grid[x][y]:
            if self.grid[new_x][new_y] == 0:
                self.grid[new_x][new_y] = self.grid[x][y]

    def save_image(self):
        for ind, grid in enumerate(self.img_frames):
            plt.savefig(f"frame_{ind}.png")

    def save_video(self):
        fig, ax = plt.subplots()
        plt.rcParams["animation.ffmpeg_path"] = "/usr/local/bin/ffmpeg"
        cmap = ListedColormap(["black", "red", "blue", "yellow"])

        self.first_frame = ax.imshow(self.img_frames[0], cmap=cmap, interpolation='nearest', animated=True)
        anim = animation.FuncAnimation(fig, self.animate, frames=len(self.img_frames))
        anim.save("movie.mp4", writer=animation.FFMpegWriter(fps=60))
        plt.close()

    def animate(self, frame):
        next_frame = self.img_frames[frame]
        self.first_frame.set_array(next_frame)
        return next_frame


test = Grid(400, 1000)
test.call()
test.save_video()


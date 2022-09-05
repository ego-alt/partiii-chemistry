import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation


# Based on Reichenbach, Mobilia & Frey (2007)
# "Mobility promotes and jeopardizes biodiversity in rock–paper–scissors games"
class Grid:
    def __init__(self, n, max_steps, one_time_step):
        self.length = n
        self.cell_status = [1, 2, 3]  # 1=type A, 2=type B, 3=type C
        self.max_steps = max_steps
        self.current_step_count = 0  # Counts the number of time steps
        self.img_frames = []  # Stores the grid array at every time step
        self.first_frame = None  # Initialises the first frame for animation later

        exchan_var = 0.005
        reprod_var = select_var = 1
        self.total_prob = exchan_var + reprod_var + select_var

        exchan_prob = self.probability(exchan_var)  # Average probability of exchange in the next time step dt
        reprod_prob = self.probability(reprod_var)  # Average probability of reproduction in the next time step dt
        select_prob = self.probability(select_var)  # Average probability of selection in the next time step dt
        self.diff_prob = [exchan_prob, reprod_prob, select_prob]
        self.one_time_step = one_time_step

    def call(self):
        # Reiterates each time step until the step limit has been reached
        while self.current_step_count < self.max_steps:
            self.one_time_step()
            self.current_step_count += 1
            if self.current_step_count % 10 == 0:
                print(f"Total number of steps: {self.current_step_count}")

    def exchange(self, grid, cell_coord, neighbour_coord):
        # Swaps positions with a neighbouring cell
        x, y = cell_coord[0], cell_coord[1]
        new_x, new_y = neighbour_coord[0], neighbour_coord[1]
        grid[x][y], grid[new_x][new_y] = grid[new_x][new_y], grid[x][y]

    def selection(self, grid, cell_coord, neighbour_coord):
        # Destroys a neighbouring cell via the "rock-paper-scissors" mechanic
        # Type A destroys B, type B destroys C, type C destroys A
        x, y = cell_coord[0], cell_coord[1]
        new_x, new_y = neighbour_coord[0], neighbour_coord[1]
        cell_index = (self.cell_status.index(grid[x][y]) + 1) % len(self.cell_status)
        if grid[new_x][new_y] == self.cell_status[cell_index]:
            grid[new_x][new_y] = 0

    def reproduction(self, grid, cell_coord, neighbour_coord):
        # Generates a new cell of the same type in the neighbouring position
        x, y = cell_coord[0], cell_coord[1]
        new_x, new_y = neighbour_coord[0], neighbour_coord[1]
        if grid[new_x][new_y] == 0:
            grid[new_x][new_y] = grid[x][y]

    def probability(self, variable):
        # Calculates the probability of exchange, reproduction or selection
        probability = variable / self.total_prob
        return probability

    def save_image(self):
        # Saves each successive iteration of the grid as a .png
        for ind, grid in enumerate(self.img_frames):
            cmap = ListedColormap(["black", "red", "blue", "yellow"])
            plt.imshow(grid, cmap=cmap, interpolation='nearest')
            plt.savefig(f"frame_{ind}.png")

    def save_video(self):
        # Generates a .mp4 from the time evolution of the grid
        fig, ax = plt.subplots()
        plt.rcParams["animation.ffmpeg_path"] = "/usr/local/bin/ffmpeg"
        cmap = ListedColormap(["black", "red", "blue", "yellow"])

        self.first_frame = ax.imshow(self.img_frames[0], cmap=cmap, interpolation='nearest', animated=True)
        anim = animation.FuncAnimation(fig, self.animate, frames=len(self.img_frames))
        anim.save("output/movie.mp4", writer=animation.FFMpegWriter(fps=60))
        plt.close()

    def animate(self, frame):
        next_frame = self.img_frames[frame]
        self.first_frame.set_array(next_frame)
        return next_frame


class PlainGrid(Grid):
    def __init__(self, n, max_steps):
        Grid.__init__(self, n, max_steps, self.one_time_step)
        self.grid = self.initial_grid()
        self.dispatcher = [self.exchange, self.reproduction, self.selection]

    def initial_grid(self):
        # Initialises the randomly populated grid
        grid = np.random.randint(self.cell_status[0], self.cell_status[-1] + 1, size=(self.length, self.length))
        return grid

    def one_time_step(self):
        reaction_count = 0
        num_cells = self.length * self.length
        while reaction_count < num_cells:
            # Randomly chooses cell coordinates
            x = random.randrange(len(self.grid))
            y = random.randrange(len(self.grid))
            cell_coord = [x, y]

            if self.grid[x][y]:
                # Randomly chooses one of four neighbours: up, down, left, right
                x_or_y = random.randint(0, 1)
                neighbour_coord = cell_coord.copy()
                neighbour_coord[x_or_y] = (neighbour_coord[x_or_y] + random.choice([-1, 1])) % self.length

                # Randomly chooses one event: exchange, selection, reproduction
                # Only valid if a cell occupies the chosen coordinates, ie value != 0
                reaction = random.choices(self.dispatcher, weights=self.diff_prob)[0]
                reaction(self.grid, cell_coord, neighbour_coord)
                reaction_count += 1

        self.img_frames.append(np.copy(self.grid))


class BorderGrid(Grid):
    def __init__(self, n, max_steps):
        Grid.__init__(self, n, max_steps, self.one_time_step)
        self.grid = self.initial_grid()
        self.dispatcher = [self.exchange, self.reproduction, self.selection]

    def initial_grid(self):
        # Initialises the randomly populated grid
        grid = np.random.randint(self.cell_status[0], self.cell_status[-1] + 1, size=(self.length - 1, self.length - 1))
        grid = np.pad(grid, pad_width=1, mode='constant', constant_values=0)  # Initialises an empty border
        return grid

    def one_time_step(self):
        reaction_count = 0
        num_cells = (self.length - 1) * (self.length - 1)
        while reaction_count < num_cells:
            # Randomly chooses cell coordinates
            x = random.randrange(1, len(self.grid) - 1)
            y = random.randrange(1, len(self.grid) - 1)
            cell_coord = [x, y]
            border_edge = {1: 1, (len(self.grid) - 2): -1}

            if self.grid[x][y]:
                # Randomly chooses one of four (central) or three (beside the fixed border) neighbours
                x_or_y = random.randint(0, 1)
                neighbour_coord = cell_coord.copy()
                if neighbour_coord[x_or_y] in border_edge:
                    neighbour_coord[x_or_y] += border_edge[neighbour_coord[x_or_y]]
                else:
                    neighbour_coord[x_or_y] += random.choice([-1, 1])

                # Randomly chooses one event: exchange, selection, reproduction
                # Only valid if a cell occupies the chosen coordinates, ie value != 0
                reaction = random.choices(self.dispatcher, weights=self.diff_prob)[0]
                reaction(self.grid, cell_coord, neighbour_coord)
                reaction_count += 1

        self.img_frames.append(np.copy(self.grid))


test = BorderGrid(200, 1000)
test.call()
test.save_video()


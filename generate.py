import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation


# "Mobility promotes and jeopardizes biodiversity in rock–paper–scissors games"

class Grid:
    # Defines the function for reiterating time steps, interaction methods, and image/video processing
    def __init__(self, n, max_steps, one_time_step, ising_on):
        self.length = n
        self.cell_status = [1, 2, 3]  # 1=type A, 2=type B, 3=type C
        self.max_steps = max_steps
        self.current_step_count = 0  # Counts the number of time steps
        self.img_frames = []  # Stores the grid array at every time step
        self.first_frame = None  # Initialises the first frame for animation later
        self.ising_on = ising_on  # TRUE/FALSE for biased diffusion

        # Parameters set in Reichenbach, Mobilia & Frey (2007)
        self.reprod_param = 1
        self.selec_param = 1
        self.exchan_param = 0.005  # M = 2 * exchange_param / max_steps = 1e-4

        # Newly introduced parameters
        self.death_param = 0.6
        self.ising_param = 0.003
        parameters = [self.exchan_param, self.reprod_param, self.selec_param, self.death_param]

        # Average probability of an interaction in the next time step dt
        total_prob = sum(parameters)
        self.diff_prob = [i / total_prob for i in parameters]
        self.one_time_step = one_time_step

        self.dispatcher = [self.exchange, self.reproduction, self.selection, self.death]

    def call(self):
        # Reiterates each time step until the step limit has been reached
        while self.current_step_count < self.max_steps:
            self.one_time_step()
            self.current_step_count += 1
            if self.current_step_count % 10 == 0:
                print(f"Total number of steps: {self.current_step_count}")

    def call_neighbours(self, cell_coord):
        x, y = cell_coord[0], cell_coord[1]
        neighbours = [(x, (y + 1) % self.length),
                      (x, (y - 1) % self.length),
                      ((x + 1) % self.length, y),
                      ((x - 1) % self.length, y)]
        return neighbours

    def exchange(self, grid, cell, neighbour):
        # Swaps positions with a neighbouring cell
        if self.ising_on:
            ising_neighbours = self.call_neighbours(neighbour)
            num_empty = len([grid[i] for i in ising_neighbours if grid[i] == 0])
            num_threshold = - (-len(ising_neighbours) // 2)

            a = 2 * num_empty - len(ising_neighbours)
            diff_threshold = np.exp(-2 * a * self.ising_param)

            if num_empty < num_threshold or np.random.rand() < diff_threshold:
                grid[cell], grid[neighbour] = grid[neighbour], grid[cell]

        else:
            grid[cell], grid[neighbour] = grid[neighbour], grid[cell]

    def selection(self, grid, cell, neighbour):
        # Destroys a neighbouring cell via the "rock-paper-scissors" mechanic
        # Type A destroys B, type B destroys C, type C destroys A
        cell_index = (self.cell_status.index(grid[cell]) + 1) % len(self.cell_status)
        if grid[neighbour] == self.cell_status[cell_index]:
            grid[neighbour] = 0

    def reproduction(self, grid, cell, neighbour):
        # Generates a new cell of the same type in the neighbouring position
        if grid[neighbour] == 0:
            grid[neighbour] = grid[cell]

    def death(self, grid, cell, _):
        # Cell leaves a vacant position after death
        grid[cell] = 0

    def replacement(self, grid, cell, neighbour):
        # Replaces a neighbouring cell via the "rock-paper-scissors" mechanic
        # Based off dominance-replacement in Szczesny, Mobilia & Rucklidge (2014)
        cell_index = (self.cell_status.index(grid[cell]) + 1) % len(self.cell_status)
        if grid[neighbour] == self.cell_status[cell_index]:
            grid[neighbour] = grid[cell]

    def mutation(self, grid, cell, _):
        # Mutates one cell type into the other
        cell_index = (self.cell_status.index(grid[cell]) + 1) % len(self.cell_status)
        grid[cell] = self.cell_status[cell_index]

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
    # Defines the initial grid set-up, additional interaction rules, and each time step
    def __init__(self, n, max_steps, ising_on=False):
        Grid.__init__(self, n, max_steps, self.one_time_step, ising_on)
        self.grid = self.initial_grid()

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
            cell_coord = (x, y)

            if self.grid[x][y]:  # Only valid if a cell occupies the chosen coordinates, ie value != 0
                # Randomly chooses one of four neighbours: up, down, left, right
                neighbour_coord = random.choice(self.call_neighbours(cell_coord))
                # Randomly chooses one event: exchange, selection, reproduction, death
                reaction = random.choices(self.dispatcher, weights=self.diff_prob)[0]
                reaction(self.grid, cell_coord, neighbour_coord)
                reaction_count += 1

        self.img_frames.append(np.copy(self.grid))


"""class BorderGrid(Grid):
    def __init__(self, n, max_steps):
        Grid.__init__(self, n, max_steps, self.one_time_step)
        self.grid = self.initial_grid()

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

        self.img_frames.append(np.copy(self.grid))"""

test = PlainGrid(200, 50, ising_on=True)
test.call()
test.save_video()

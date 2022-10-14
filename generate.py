import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from tqdm import tqdm

# "Mobility promotes and jeopardizes biodiversity in rock–paper–scissors games." Reichenbach, Mobilia & Frey (2007)

REPRODUCTION_PARAM = 0  # A + 0 --> A + A
SELECTION_PARAM = 0  # A + B --> A + 0
EXCHANGE_PARAM = 1  # M = 2 * exchange_param / number of sites
# Newly introduced parameters
ISING_PARAM = 1.7
DEATH_PARAM = 0
# Additional parameters set in Szczesny, Mobilia & Rucklidge (2014)
REPLACEMENT_PARAM = 0  # A + B --> A + A
MUTATION_PARAM = 0  # A --> B

SEED = 12345
FILENAME = "./output/state.pickle"
PERIODIC_BOUNDARY_ON = False
ISING_ON = True  # TRUE/FALSE for biased diffusion


class Grid:
    # Defines the function for reiterating time steps, interaction methods, and image/video processing
    def __init__(self, n, max_steps, one_time_step, dispatcher):
        self.length = n
        self.num_cells = n ** 2
        self.max_steps = max_steps
        self.current_step_count = 0  # Counts the number of time steps
        self.cell_status = [1, 2, 3]  # 1= type A(red), 2= type B(blue), 3= type C(yellow)

        self.current_cell = ()
        self.current_neighbours = []
        self.img_frames = []  # Stores the grid array at every time step
        self.first_frame = None  # Initialises the first frame for animation later

        pairings = {"exchange": EXCHANGE_PARAM,
                    "selection": SELECTION_PARAM,
                    "reproduction": REPRODUCTION_PARAM,
                    "death": DEATH_PARAM,
                    "replacement": REPLACEMENT_PARAM,
                    "mutation": MUTATION_PARAM}

        # Average probability of an interaction in the next time step dt
        diff_prob = [pairings[d.__name__] for d in dispatcher]
        total_prob = sum(diff_prob)
        self.diff_prob = np.divide(diff_prob, total_prob)
        self.one_time_step = one_time_step

    def call(self):
        # Reiterates each time step until the step limit has been reached
        for i in tqdm(range(self.max_steps), desc="Progress bar"):
            self.one_time_step()
            self.current_step_count = i

        print("Simulation complete, now loading animation...")
        self.save_video()
        self.save_state()

    def call_neighbours(self, cell_coord):
        x, y = cell_coord[0], cell_coord[1]
        if PERIODIC_BOUNDARY_ON:
            self.current_neighbours = [(x, (y + 1) % self.length),
                                       (x, (y - 1) % self.length),
                                       ((x + 1) % self.length, y),
                                       ((x - 1) % self.length, y)]
        else:
            neighbours = [(x, (y + 1)), (x, (y - 1)), ((x + 1), y), ((x - 1), y)]
            self.current_neighbours = [(x, y) for (x, y) in neighbours
                                       if 0 <= x < self.length and 0 <= y < self.length]
        return self.current_neighbours

    def calc_neighbours(self, grid, neighbours):
        bias = len([i for i in neighbours if grid[i] > 0])
        return bias

    def exchange(self, grid, cell, neighbour):
        # Swaps positions with a neighbouring cell
        calc_original = self.calc_neighbours(grid, self.current_neighbours)
        grid[cell], grid[neighbour] = grid[neighbour], grid[cell]
        if ISING_ON and grid[cell] == 0:
            calc_new = self.calc_neighbours(grid, self.call_neighbours(neighbour))
            num_J = calc_original - calc_new
            prob = min(1, np.exp(- num_J * ISING_PARAM))
            if prob < np.random.rand():
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
        anim.save("./output/movie.mp4", writer=animation.FFMpegWriter(fps=60))
        plt.close()

    def animate(self, frame):
        next_frame = self.img_frames[frame]
        self.first_frame.set_array(next_frame)
        return next_frame

    def save_state(self):
        total_cells = self.length ** 2
        state = {}
        prop_full, prop_dead = [], []

        for grid in self.img_frames:
            ar_unique, i = np.unique(grid, return_counts=True)
            prop_full.append(i[1:] / total_cells)
            prop_dead.append(i[0] / total_cells)

        state["param"] = {
            "seed": SEED,
            "system_size": total_cells,
            "time_steps": self.max_steps
        }

        state["data"] = {
            "prop_full": prop_full,
            "prop_dead": prop_dead,
            "time_evol": self.img_frames
        }

        with open(FILENAME, "wb") as file:
            pickle.dump(state, file)


class PlainGrid(Grid):
    # Defines the initial grid set-up, additional interaction rules, and each time step
    def __init__(self, n, max_steps):
        self.dispatcher = [self.exchange]
        Grid.__init__(self, n, max_steps, self.one_time_step, self.dispatcher)
        self.grid = self.initial_grid()

    def initial_grid(self):
        np.random.seed(SEED)  # Initialises the randomly populated grid with a fixed seed
        grid = np.random.choice(4, size=(self.length, self.length), p=[0.76, 0.08, 0.08, 0.08])
        # grid = np.pad(grid, pad_width=1, mode='constant', constant_values=0)  # Adds an empty border
        return grid

    def one_time_step(self):
        reaction_count = 0
        # while reaction_count < self.num_cells:
        while reaction_count < self.num_cells:
            # Randomly chooses cell coordinates
            cell_coord = (random.randrange(self.length), random.randrange(self.length))
            if self.grid[cell_coord]:  # Only valid if a cell occupies the chosen coordinates, ie value != 0
                # Randomly chooses one of four neighbours: up, down, left, right
                neighbour_coord = random.choice(self.call_neighbours(cell_coord))
                # Randomly chooses one event: exchange, selection, reproduction, death
                reaction = np.random.choice(len(self.dispatcher), p=self.diff_prob)
                self.dispatcher[reaction](self.grid, cell_coord, neighbour_coord)
                reaction_count += 1

        self.img_frames.append(np.copy(self.grid))


test = PlainGrid(40, 2500)
test.call()

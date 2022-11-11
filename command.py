"""Run in Python REPL"""
# import julia
# julia.install()
# from julia.api import Julia
# jl = Julia(compiled_modules=False)

from julia import Pkg

Pkg.activate("Project")
from julia import Project
from plot_data import Reader, Writer
import matplotlib.pyplot as plt
import numpy as np

Project.pair_handler(("exchange", 0.05), ("ising", 0), ("death", 0), ("reproduction", 1), ("selection", 1))

for j in [0.52, 0.54, 0.56, 0.58, 0.60, 0.62]:
        Project.pair_handler(("death", j))
        x = Project.call(100, 10000, seed=456789)
        y = Writer(x)
        # y.save_video(f"e0.5d{j * 10}.mp4")
        y.save_state(Project.pair_handler(write=True), f"e0.0005d{j * 10}[2].pickle")
        z = Reader(f"./output/e0.0005d{j * 10}[2].pickle")
        z.populations()

a = Reader("./output/e0.0005d5.6[2].pickle").time_evol
b = Reader("./output/e0.005d5.6[2].pickle").time_evol
c = Reader("./output/e0.05d5.6[2].pickle").time_evol
c_test = np.concatenate(([c[0]], c[1::10]))

cases = [a, b, c_test]
case_labels = [0.0005, 0.005, 0.05]

plt.title("Populations of dead cells against time")
plt.ylabel("Number of dead cells")
plt.xlabel("Time steps")
for ind, i in enumerate(cases):
        n = [np.count_nonzero(arr == 0) for arr in i]
        plt.plot(range(0, 10001, 10), n, label=case_labels[ind])

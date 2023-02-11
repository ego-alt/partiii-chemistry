"""Run in Python REPL"""
# import julia
# julia.install()
# from julia.api import Julia
# jl = Julia(compiled_modules=False)

from julia import Pkg
Pkg.activate("Project")
from julia import Project
from plot_data import Reader, Writer

exchange = 0.5
Project.pair_handler(("exchange", exchange), ("ising", 0), ("death", 0),
                     ("reproduction", 1), ("selection", 1))


def run_loop(death_params, size, time_steps, seeds):
    for i in seeds:
        for j in death_params:
            Project.pair_handler(("death", j))
            x = Project.call(size, time_steps, seed=i)
            y = Writer(x)
            # y.save_video(f"e{exchange}d{j * 10}.mp4")
            y.save_state(Project.pair_handler(write=True), f"e{exchange}d{j * 10}.pickle")
            z = Reader(f"./output/e{exchange}d{j * 10}.pickle")
            z.populations()

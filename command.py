"""Run in Python REPL"""
# import julia
# julia.install()
# from julia.api import Julia
# jl = Julia(compiled_modules=False)

from julia import Pkg
Pkg.activate("Project")
from julia import Project
from plot_data import Reader, Writer

Project.pair_handler(("exchange", 1), ("ising", 0.6), ("reproduction", 0), ("selection", 0))

for i in [0.05, 0.5, 5]:
    Project.pair_handler(("exchange", i))
    x = Project.call(100, 4000)
    y = Writer(x)
    y.save_video(f"e{i}d6.mp4")
    y.save_state(Project.pair_handler(write=True), f"e{i}d6.pickle")

"""Run in Python REPL"""
# import julia
# julia.install()

from julia import Pkg
Pkg.activate("Project")
from julia import Project
from plot_data import Reader, Writer
import pickle

FILENAME = "./output/state.pickle"

with open(FILENAME, 'rb') as f:
    data = pickle.load(f)
    print(data["var"]["time_steps"])

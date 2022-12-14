using Pkg
# Pkg.add("StatsBase")
# Pkg.add("DataStructures")
# Pkg.add("JLD2")
# Pkg.add("ProgressBars")
using StatsBase, DataStructures, JLD2, ProgressBars
using Random, Plots, FixedPointNumbers

# One-liners
x ± y = (x-y, x+y)
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

# Definition of interaction events
function call_neighbours(x, y, n)
    neighbours = [(x, mod1(y + 1, n)), 
                  (x, mod1(y - 1, n)), 
                  (mod1(x + 1, n), y), 
                  (mod1(x - 1, n), y)]
    if ! periodic_boundary_on
        neighbours = [(w, z) for (w, z) in neighbours if w in x ± 1 || z in y ± 1] # Remove neighbours which require a wraparound
    else
        return neighbours
    end
end

function calc_neighbours(grid::Array, neighbours::Vector)
    num_full = sum([1 for (x, y) in neighbours if grid[x, y] > 0]) # Count the number of filled neighbours
end

function death(grid::Array, x::Int, y::Int, args...)
    grid[x, y] = 0
end

# Consider the 3 neighbours around cell A and the 3 neighbours around dead cell 0 (A <--> 0)
# n1 filled neighbours for A --> s(i) . ∑s(i)s(j) = n1 - (3 - n1)
# n2 filled neighbours for 0 --> s(i) . ∑s(i)s(j) = (3 - n2) - n2
# Original configuration: no. of stabilising pair interactions = 2 * n1 - 2 * n2
# New configuration: no. of stabilising pair interactions = 2 * n2 - 2 * n1

function exchange(grid::Array, x::Int, y::Int, new_x::Int, new_y::Int, ising_on::Bool)
    if ! ising_on || grid[new_x, new_y] > 0 # If the neighbour selected is dead: --> immediately perform the exchange
        grid[x, y], grid[new_x, new_y] = grid[new_x, new_y], grid[x, y]
    else
        n, n = size(grid)
        calc_original = calc_neighbours(grid, call_neighbours(x, y, n)) # Calculate nearest neighbours (non-periodic)
        calc_new = calc_neighbours(grid, call_neighbours(new_x, new_y, n)) - 1 # Remove 1 to account for the swap

        if rand() < ising(calc_original, calc_new) # If random number < P(accepting the exchange): --> perform the exchange
            grid[x, y], grid[new_x, new_y] = grid[new_x, new_y], grid[x, y]
        end        
    end
end

function ising(calc_original::Int, calc_new::Int)
    # Calculate the change in stabilising pair interactions
    Δ = 4 * (calc_original - calc_new) # Compute change in no. of stabilising pair interactions
    prob = min(1, exp(- Δ * pairings[ising])) # If Δ < 0, i.e. stabilisation increases: choose 1 --> immediately perform the exchange; 
    #                                           If Δ > 0, i.e. stabilisation decreases: choose exp(-ve), i.e. < 1
end

function reproduction(grid::Array, x::Int, y::Int, new_x::Int, new_y::Int)
    if grid[new_x, new_y] == 0
        grid[new_x, new_y] = grid[x, y]
    end
end

function selection(grid::Array, x::Int, y::Int, new_x::Int, new_y::Int)
    if grid[new_x, new_y] == grid[x, y] % 3 + 1
        grid[new_x, new_y] = 0
    end
end

# Bulk flow of Monte Carlo simulations
pairings = Dict{Function, Real}(death => 0, 
                                exchange => 1, 
                                ising => 1.8, 
                                reproduction => 0, 
                                selection => 0)
periodic_boundary_on = false

function to_colour(i::Int)
    if i == 0
        return colorant"black"
    else
        return colorant"red"
    end
end

function initial_grid(n::Int, default_seed::Int=123456)
    Random.seed!(default_seed)
    sample(0:3, Weights([6, 1, 1, 1]), (n, n))
end

function call(n::Int, max_steps::Int, filename::String="")
    tot_cells = n * n
    grid = initial_grid(n)
    img_frames = [to_colour.(grid)]
    # img_frames = [grid]
    new_pairings = [(a, b) for (a, b) in pairings if b > 0 && a != ising]
    active_events, probabilities = unzip(new_pairings)
    probabilities = probabilities / sum(probabilities)
    ising_on = pairings[ising] > 0
    
    for time_step ∈ ProgressBar(1:max_steps)
        for cell_step ∈ 1:tot_cells
            x, y = rand(1:n, 2)
            if grid[x, y] > 0
                new_x, new_y = rand(call_neighbours(x, y, n))
                event = sample(active_events, Weights(probabilities))
                ! (event == exchange) ? event(grid, x, y, new_x, new_y) : 
                                        event(grid, x, y, new_x, new_y, ising_on)
            end
        end
        ! isempty(filename) && push!(img_frames, to_colour.(grid))
        # ! isempty(filename) && push!(img_frames, grid)
    end

    # Saving the raw data for visualisation later
    println("Simulation complete, compiling data...")
    if ! isempty(filename)
        img_frames = cat(dims=3, img_frames...)
        save(filename * ".gif", img_frames, fps=5) 
        jldsave(filename * ".jld2"; grid_length=n, time_steps=max_steps, parameters=pairings, time_evol=img_frames)
        println("Animation saved")
    end
    return to_colour.(grid)
end

# Extract data from .jld2 files
function summarise(filename::String)
    file_data = load(filename)
    n, max_steps = file_data["grid_length"], file_data["time_steps"]
    parameters = file_data["parameters"]
    all_frames = file_data["time_evol"]
    tot_cells = n * n

    println("########################################")
    println("$n x $n grid after $max_steps time steps.")
    println("########################################")
    println("Parameters (", length(parameters), " entries):")
    for (k, v) in parameters
        println("$k => $v")
    end
    println("########################################")
    last_frame = all_frames[:, :, end]
    empty, c, b, a = values(counter(last_frame))
    empty_p, c_p, b_p, a_p = round.([empty, c, b, a] * 100 / tot_cells, sigdigits=3)

    println("Final numbers of different particles:")
    println("Species A => $a ($a_p%)")
    println("Species B => $b ($b_p%)")
    println("Species C => $c ($c_p%)")
    println("Empty => $empty ($empty_p%)")
    println("########################################")
end

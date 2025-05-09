from simulation.simulation import Simulation

if __name__ == "__main__":
    sim = Simulation(num_households=5)
    sim.run(steps=3)

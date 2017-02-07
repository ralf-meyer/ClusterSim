import warnings
import numpy as _np

class Atom(object):
    def __init__(self, atom_type, xyz):
        # Todo: use either string or numeric atom type 
        # and do the convertion here
        self.atom_type = atom_type
        self.xyz = xyz

class Cluster(object):
    def __init__(self):
        self.atoms = []

    def calc_neighbors(self):
        neighbor_list = []

    def analyze(self):
        # Calculates the energy of the current geometry
        pass

    def optimize(self):
        # Function that optimizes the current geometry
        # returns the optimized energy
        pass

    def read(self, filename):
        if filename[-4:] == ".xyz":
            self.read_xyz(filename)
        else:
            warnings.warn("Unknown file extension!")        

    def read_xyz(self, filename):
        with open(filename, "r") as f_in:
            for i, line in enumerate(f_in):
                if i == 0:
                    # set counter for number of lines to be read
                    counter = int(line)
                    # Skip comment line
                    next(f_in)
                elif counter >= 0:
                    sp = line.split()
                    self.atoms.append(Atom(sp[0], _np.array([float(s) for s in sp[1:4]])))
                    counter -= 1
                if counter == 0:
                    break

     


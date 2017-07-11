import numpy as _np
import copy

class MarkovChainMonteCarlo(object):
    kb = 8.617 * 10**(-5)  # in eV/K
    
    def __init__(self, lmp):
        self.lmp = lmp

    def setup(self):
        self.neighbors, self.coord_curr = self.calc_neighbors()
        
        self.lmp.command("minimize 1.0e-4 0 100 1000")
        e_start = self.lmp.get_thermo("pe")
    
        self.naccept = 0
        
        self.energies = [e_start]
        self.e_curr = [e_start]
        self.e_min = e_start
        self.pos_min = copy.deepcopy(self.lmp.gather_atoms("x", 1, 3))
        self.types_min = copy.deepcopy(self.lmp.gather_atoms("type", 0, 1))
        self.coord_nrs = [copy.copy(self.coord_curr)]
        self.plot_current_geo(filename = "min_auto")
    

    def step(self, T, exchange = "neighbors", eta = 1, log_coords = True):
        pos_old = self.lmp.gather_atoms("x", 1, 3)
        types_old = self.lmp.gather_atoms("type", 0, 1)
        types = copy.copy(types_old)
    
        found = False 
        if exchange == "neighbors":
            # randomly choose a pair of neighboring atoms of different type             
            for ind1 in _np.random.permutation(self.lmp.get_natoms()):                
                for ind2 in _np.random.permutation(self.neighbors[ind1]):
                    if not types[ind1] == types[ind2]:
                        found = True
                        break
                if found:
                    break
        elif exchange == "tailored":    
            # randomly choose a pair of different type based on their coordination number
            # TODO rewrite failproof:
            tot_coord = _np.sum(eta*self.coord_curr+1)
            cum_coord = _np.cumsum(eta*self.coord_curr+1)
            for rand1 in _np.random.permutation(tot_coord):
                ind1 = _np.argmax(cum_coord >  rand1)
                for rand2 in _np.random.permutation(tot_coord):
                    ind2 = _np.argmax(cum_coord > rand2)
                    if not types[ind1] == types[ind2]:
                        found = True
                        break
                if found:
                    break
        else: # Random
            for ind1 in _np.random.permutation(self.lmp.get_natoms()):
                for ind2 in _np.random.permutation(self.lmp.get_natoms()):
                    if not types[ind1] == types[ind2]:
                        found = True
                        break
                if found:
                    break
            
        # swap their type 
        types[ind1], types[ind2] = types[ind2], types[ind1]
        self.lmp.scatter_atoms("type", 0, 1, types)
    
        # evaluate the new energy
        self.lmp.command("minimize 1.0e-4 0 100 1000")
        e_new = self.lmp.get_thermo("pe")
        self.energies.append(e_new)
    
        # accept or decline the new geometry
        if _np.random.rand() < _np.exp((self.e_curr[-1] - e_new)/(self.kb*T)): # Accept
            # Save energies etc
            c = 0
            for j in self.neighbors[ind1]: 
                if types[j] == types[ind1] and not j == ind2:
                    self.coord_curr[j] -= 1
                elif not types[j] == types[ind1] and not j == ind2:
                    self.coord_curr[j] += 1
                    c += 1
                elif not types[j] == types[ind1] and j == ind2:
                    c += 1
            self.coord_curr[ind1] = copy.copy(c)
            c = 0
            for j in self.neighbors[ind2]:
                if types[j] == types[ind2] and not j == ind1:
                    self.coord_curr[j] -= 1
                elif not types[j] == types[ind2] and not j == ind1:
                    self.coord_curr[j] += 1
                    c += 1
                elif not types[j] == types[ind2] and j == ind1:
                    c += 1
            self.coord_curr[ind2] = copy.copy(c)      
            self.naccept += 1
            self.e_curr.append(e_new) 
            if e_new < self.e_min: 
                self.e_min = e_new
                self.pos_min = copy.deepcopy(
                        self.lmp.gather_atoms("x", 1, 3))
                self.types_min = copy.deepcopy(
                        self.lmp.gather_atoms("type", 0, 1))
                self.plot_current_geo(filename = "min_auto")
        else: # Decline
            self.lmp.scatter_atoms("type", 0, 1, types_old)
            self.lmp.scatter_atoms("x", 1, 3, pos_old)
            self.e_curr.append(self.e_curr[-1])
        if log_coords:
            self.coord_nrs.append(copy.copy(self.coord_curr))

    def calc_neighbors(self):
        pos = self.lmp.gather_atoms("x",1,3)
        types = self.lmp.gather_atoms("type",0,1)
    
        neighbors = []
        coordination = []
        cut = 3.5

        for i in xrange(self.lmp.get_natoms()):
            neighs = []
            c = 0
            for j in xrange(self.lmp.get_natoms()):
                if not i == j:
                    r_ij = _np.sqrt((pos[3*i]-pos[3*j])**2 + 
                                   (pos[3*i+1]-pos[3*j+1])**2 + 
                                   (pos[3*i+2]-pos[3*j+2])**2)
                    if r_ij < cut:
                        neighs.append(j)
                        if not types[i] == types[j]:
                            c += 1
            neighbors.append(neighs)
            coordination.append(c)
        coordination = _np.array(coordination) # Allow easier slicing
        return neighbors, coordination
    
    def plot_current_geo(self, filename = "curr_geo"):
        self.lmp.command(
                "write_dump all image {}.png type type zoom 2.5 view 60 0 box no 1.0 modify backcolor white adiam 1 3.0".format(filename))
        
    def plot_min_geo(self, filename = "min_geo"):
        pos_old = self.lmp.gather_atoms("x",1,3)
        types_old = self.lmp.gather_atoms("type",0,1)
        self.lmp.scatter_atoms("x", 1, 3, self.pos_min)
        self.lmp.scatter_atoms("type", 0, 1, self.types_min)
        self.lmp.command(
                "write_dump all image {}.png type type zoom 2.5 view 60 0 box no 1.0 modify backcolor white adiam 1 3.0".format(filename))
        self.lmp.scatter_atoms("x", 1, 3, pos_old)
        self.lmp.scatter_atoms("type", 0, 1, types_old)
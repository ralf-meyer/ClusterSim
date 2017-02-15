import warnings
import numpy as _np
from scipy.constants import golden
from itertools import combinations_with_replacement

class Atom(object):
    def __init__(self, atom_type, xyz):
        # Todo: use either string or numeric atom type 
        # and do the convertion here
        self.atom_type = atom_type
        self.xyz = xyz

class Cluster(object):
    def __init__(self):
        self.atoms = []
        self.neighbors = []

    def calc_neighbors(self, R = 3.0):
        self.neighbors = []
        for a in self.atoms:
            n = []
            for j, b in enumerate(self.atoms):
                if not a is b and _np.linalg.norm(a.xyz-b.xyz) < R:
                    n.append(j)
            self.neighbors.append(n)

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

     
class Ikosaeder(object):
    basis_vectors = _np.sqrt(1 + golden**2)**(-1) * _np.array([[0, 1, golden], # 0
                     [0, 1, -golden],  # 1
                     [0, -1, golden],  # 2
                     [0, -1, -golden], # 3
                     [1, golden, 0],   # 4
                     [1, -golden, 0],  # 5
                     [-1, golden, 0],  # 6
                     [-1, -golden, 0], # 7
                     [golden, 0, 1],   # 8
                     [golden, 0, -1],  # 9
                     [-golden, 0, 1],  # 10
                     [-golden, 0, -1]])# 11

    faces = [[0, 4, 8],
             [0, 4, 6],
             [0, 6, 10],
             [0, 10, 2],
             [0, 2, 8],
             [2, 7, 5],
             [2, 5, 8],
             [8, 5, 9],
             [8, 9, 4],
             [4, 9, 1],
             [4, 1, 6],
             [6, 1, 11],
             [6, 11, 10],
             [10, 11, 7],
             [10, 7, 2],
             [3, 1, 9],
             [3, 9, 5],
             [3, 5, 7],
             [3, 7, 11],
             [3, 11, 1]]
             
    unit_vectors = _np.eye(12)
    
    def __init__(self):
        pass

    def build_layers(self, n):
        # Add center atom
        vectors = [[[0,0,0,0,0,0,0,0,0,0,0,0]]]
        # Build individual layer
        for i in range(1,n+1):
            layer = []
            # Build layer triangle by triangle
            for fa in self.faces:
                face_vecs = [self.unit_vectors[fa[0]], self.unit_vectors[fa[1]], self.unit_vectors[fa[2]]]
                for perm in combinations_with_replacement(face_vecs, i):
                    vec = list(_np.sum(perm,0))
                    # Check if vec is already in layer (happens along the edges)
                    if not vec in layer:
                        layer.append(vec)                                 
            vectors.append(layer)
        self.layers = vectors

    @staticmethod
    def get_xyz(layers, scale):
        xyzs = []
        if len(_np.shape(layers)) == 1: # multiple layers
            for layer in layers:
                for atom in layer:
                    xyzs.append(scale*_np.array(atom).dot(Ikosaeder.basis_vectors))
        else: # single layer
            for atom in layers:
                    xyzs.append(scale*_np.array(atom).dot(Ikosaeder.basis_vectors))
        return xyzs
        
    @staticmethod
    def get_magic_nr(n):
        n = n+1 # To stay consistent with the definition of number of layers
        return (10*n**3 - 15*n**2 + 11*n - 3)/3
        
    def draw_sphere(ax, center, c = "b"):
        u = _np.linspace(0, 2 * _np.pi, 72+1)
        v = _np.linspace(0, _np.pi, 36+1)
    
        x = center[0] + 1 * _np.outer(_np.cos(u), _np.sin(v))
        y = center[1] + 1 * _np.outer(_np.sin(u), _np.sin(v))
        z = center[2] + 1 * _np.outer(_np.ones(_np.size(u)), _np.cos(v))
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, linewidth = 0, color = c, alpha = 0.5)





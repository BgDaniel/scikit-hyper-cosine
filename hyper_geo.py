import numpy as np
from string import ascii_uppercase
import math

from math_helper import angle

class BeltramiKlein:
    def __init__(self, dim):
        self._dim = dim

    def contained_in(self, vector):
        if np.linalg.norm(vector) < 1.0:
            return True
        else:
            return False   

    def metric_tensor(self, x, vec, wec):
        assert len(x) == self._dim and len(vec) == self._dim and len(wec) == self._dim,\
            'Input vectors must same length as space dimension {0}'.format(self._dim)
        scalar_vec_wev = np.dot(vec, wec)
        x_abs = np.linalg.norm(x)
        x_abs_squared = x_abs * x_abs
        scalar_x_vec = np.dot(x, vec)
        scalar_x_wev = np.dot(x, wec)

        return scalar_vec_wev / (1.0 - x_abs_squared) + (scalar_x_vec * scalar_x_wev) \
            / ((1.0 - x_abs_squared) * (1.0 - x_abs_squared))

    def angle(self, x, v, w):
        return angle(v, w, self.metric_tensor, x)

    def rnd_points(self, nb_samples, hyper_norm = 1.0):
        samples = []

        for i in range(0, nb_samples):
            is_in = False

            while is_in is False:
                vec = np.random.uniform(-1.0, 1.0, self._dim)
                is_in = self.contained_in(vec)

            samples.append(vec)

        return samples

    def rnd_simplices(self, nb_samples, hyper_norm = 1.0):
        samples = []

        for i in range(0, nb_samples):
            edges = self.rnd_points(self._dim, hyper_norm)          
            samples.append(HyperSimplex(self._dim, edges))

        return samples

class BeltramiKlein3Dim(BeltramiKlein):
    def __init__(self):
        BeltramiKlein.__init__(self, 3)

    def surface_0(self, a, b):
        assert len(a) == 3 and len(b) == 3, 'Input vectors must have length 3!'        
        return math.pi - angle(a, b) - super().angle(a, - a, b - a) - super().angle(b, - b, a - b)

    def surface(self, a, b, c):
        assert len(a) == 3 and len(b) == 3 and len(c) == 3, 'Input vectors must have length 3!'         
        return math.pi - super().angle(b, a - b, c - b) - super().angle(c, a - c, b - c) - super().angle(a, b - a, c - a)


class HyperSimplex:
    def __init__(self, dim, edges):
        self._dim = dim
        self._edges = {}
        self._angles = None
        self._surfaces = None
        self._adjacent_surface = .0

        for i, edge in enumerate(edges):
            assert len(edge) == self._dim, 'Edge vectors must same length as space dimension {0}'.format(self._dim)
            self._edges[ascii_uppercase[i]] = edge

    def angles(self):
        if self._angles != None:
            return self._angles
        else:
            self._angles = {}

            for i, edge in enumerate(self._edges.values()):
                edge_0 = edge
                edge_0_name = list(self._edges.keys())[i]
                edge_1 =  list(self._edges.values())[(i + 1) % self._dim]
                edge_1_name = list(self._edges.keys())[(i + 1) % self._dim]

                self._angles[edge_0_name + edge_1_name] = angle(edge_0, edge_1)

        return self._angles

    def surfaces(self):
        if self._surfaces != None:
            return self._surfaces
        else:
            if self._dim != 3:
                raise Exception('Dimension has to be equal to 3!')

            beltramiKlein3Dim = BeltramiKlein3Dim()
            self._surfaces = {}
            
            A, B, C = list(self._edges.values())[0], list(self._edges.values())[1], list(self._edges.values())[2]
            self._surfaces['AB'] = beltramiKlein3Dim.surface_0(A, B)
            self._surfaces['BC'] = beltramiKlein3Dim.surface_0(B, C)
            self._surfaces['CA'] = beltramiKlein3Dim.surface_0(C, A)

            self._adjacent_surface = beltramiKlein3Dim.surface(A, B, C)

        return self._surfaces, self._adjacent_surface





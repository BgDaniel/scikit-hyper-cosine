import numpy as np
from string import ascii_uppercase
import math

from math_helper import angle, FindIdealPoints

class BeltramiKlein:
    def __init__(self, dim):
        self._dim = dim

    def contained_in(self, vector, hyperbolic_length = None):
        if hyperbolic_length == None:
            if np.linalg.norm(vector) < 1.0:
                return True
            else:
                return False 
        else:
            euclidean_length = math.tanh(hyperbolic_length)
            if np.linalg.norm(vector) < euclidean_length:
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

    def rnd_points(self, nb_samples, hyper_norm = None):
        samples = []

        for i in range(0, nb_samples):
            is_in = False

            while is_in is False:
                vec = np.random.uniform(-1.0, 1.0, self._dim)
                is_in = self.contained_in(vec, hyper_norm)

            samples.append(vec)

        return samples

    def rnd_simplices(self, nb_samples, hyper_norm = None):
        samples = []

        for i in range(0, nb_samples):
            edges = self.rnd_points(self._dim, hyper_norm)          
            samples.append(HyperSimplex(self._dim, edges))

        return samples

class BeltramiKlein3Dim(BeltramiKlein):
    def __init__(self):
        BeltramiKlein.__init__(self, 3)

    def surface_0(self, edges):
        assert len(edges) == 2, 'There must be two edges in the list!'
        a, b = edges[0], edges[1]
        assert len(a) == 3 and len(b) == 3, 'Input vectors must have length 3!'        
        return math.pi - angle(a, b) - super().angle(a, - a, b - a) - super().angle(b, - b, a - b)

    def surface(self, edges):
        assert len(edges) == 3, 'There must be three edges in the list!'
        a, b, c = edges[0], edges[1], edges[2]
        assert len(a) == 3 and len(b) == 3 and len(c) == 3, 'Input vectors must have length 3!'         
        return math.pi - super().angle(b, a - b, c - b) - super().angle(c, a - c, b - c) - super().angle(a, b - a, c - a)

class BeltramiKlein2Dim(BeltramiKlein):
    def __init__(self):
        BeltramiKlein.__init__(self, 2)

    def surface_0(edges):
        assert len(edges) == 1, 'There must be one edge in the list!'
        assert len(edges[0]) == 2, 'Input vectors must have length 2!'        
        return math.atanh(np.linalg.norm(edges[0]))

    def surface(edges):
        assert len(edges) == 2, 'There must be one edge in the list!'
        a, b = edges[0], edges[1]
        assert len(a) == 2 and len(b) == 2, 'Input vectors must have length 2!'         
        
        findIdealPoints = FindIdealPoints(a, b)
        r, p, q, s = findIdealPoints.find()
        rq = abs(np.linalg.norm(r - q))
        ps = abs(np.linalg.norm(p - s))
        rp = abs(np.linalg.norm(r - p))
        qs = abs(np.linalg.norm(q - s))

        return .5 * math.log((rq * ps) / (rp * qs))


class HyperSimplex:
    @property
    def get_angles(self):
        if self._angles == None:
            return self.angles()
        else:
            return self._angles

    @property
    def get_faces(self):
        if self._surfaces == None:
            return self.surfaces()
        else:
            return self._surfaces

    def __init__(self, dim, edges):
        self._dim = dim
        self._edges = {}
        self._angles = None
        self._surfaces = None
        self._adjacent_surface = None

        for i, edge in enumerate(edges):
            assert len(edge) == self._dim, 'Edge vectors must same length as space dimension {0}'.format(self._dim)
            self._edges[ascii_uppercase[i]] = edge

        self._beltramiKleinModel = None

        if self._dim != 2 and self._dim != 3:
            raise Exception('Dimension has to be equal to 2 or 3!')
        elif self._dim == 3:
            self._beltramiKleinModel = BeltramiKlein2Dim()
        else:
            self._beltramiKleinModel = BeltramiKlein3Dim()

    def angles(self):
        if self._angles != None:
            return self._angles
        else:
            self._angles = {}

            for i, edge in enumerate(self._edges.values()):
                A = edge
                name_A = list(self._edges.keys())[i]
                B =  list(self._edges.values())[(i + 1) % self._dim]
                name_B = list(self._edges.keys())[(i + 1) % self._dim]

                self._angles[name_A + name_B] = angle(A, B)

        return self._angles

    def surfaces(self):
        if self._surfaces != None:
            return self._surfaces
        else:            
            self._surfaces = {}
            
            for i, edge in enumerate(self._edges.values()):
                A = edge
                name_A = list(self._edges.keys())[i]
                B =  list(self._edges.values())[(i + 1) % self._dim]
                name_B = list(self._edges.keys())[(i + 1) % self._dim]

                self._surfaces[name_A + name_B] = self._beltramiKleinModel.surface_0(A, B)

            return self._surfaces

        def adjacent_surface(self):
            if self._adjacent_surface != None:
                return self._surfaces
            else:
                self._adjacent_surface = self._beltramiKleinModel.surface(self._edges.values())
                return self._adjacent_surface





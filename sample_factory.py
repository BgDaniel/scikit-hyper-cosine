import pandas as pd
from hyper_geo import BeltramiKlein2Dim, BeltramiKlein3Dim

class SampleCreator:
    def __init__(self, dim, upper_boundary = 1.0):
        self._dim = dim

        beltramiKleinModel = None
            
        if self._dim != 2 and self._dim != 3:
            raise Exception('Dimension has to be equal to 2 or 3!')
        elif self._dim == 3:
            beltramiKleinModel = BeltramiKlein2Dim()
        else:
            beltramiKleinModel = BeltramiKlein3Dim()
        self._upper_boundary = upper_boundary

    def create(self, nb_samples=1000, save_to=None):
        rnd_simplices = self._beltramiKlein.rnd_simplices(nb_samples, self._upper_boundary)
        data = {}
        angles_AB = []
        angles_BC= []
        angles_CA = []
        surfaces_AB = []
        surfaces_BC = []
        surfaces_CA = []
        surfaces_ABC = []
        
        for rnd_simplex in rnd_simplices:
            angles = rnd_simplex.angles()
            angles_AB.append(angles['AB'])
            angles_BC.append(angles['BC'])
            angles_CA.append(angles['CA'])
            
            surfaces, adjacent_surface = rnd_simplex.surfaces()
            surfaces_AB.append(surfaces['AB'])
            surfaces_BC.append(surfaces['BC'])
            surfaces_CA.append(surfaces['CA'])
            surfaces_ABC.append(adjacent_surface)

        data['angles_AB'] = angles_AB
        data['angles_BC'] = angles_BC
        data['angles_CB'] = angles_CA
        data['surfaces_AB'] = surfaces_AB
        data['surfaces_BC'] = surfaces_BC
        data['surfaces_CA'] = surfaces_CA
        data['surfaces_ABC'] = surfaces_ABC

        return pd.DataFrame(data)

sampleCreator = SampleCreator(0.2)
samples = sampleCreator.create(100)

print(samples.head())

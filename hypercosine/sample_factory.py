import pandas as pd
from hyper_geo import BeltramiKlein2Dim, BeltramiKlein3Dim

class SampleCreator:
    def __init__(self, dim, upper_boundary = 1.0):
        self._dim = dim

        self._beltramiKlein = None
            
        if self._dim != 2 and self._dim != 3:
            raise Exception('Dimension has to be equal to 2 or 3!')
        elif self._dim == 2:
            self._beltramiKlein = BeltramiKlein2Dim()
        else:
            self._beltramiKlein = BeltramiKlein3Dim()
        self._upper_boundary = upper_boundary

    def create(self, nb_samples=1000, save_to=None):
        rnd_simplices = self._beltramiKlein.rnd_simplices(nb_samples, self._upper_boundary)
        data = {}
        
        for rnd_simplex in rnd_simplices:
            angles = rnd_simplex.get_angles
            for k, v in angles.items():
                if k in data:
                    data[k].append(v) 
                else:
                    data[k] = [v]

            faces = rnd_simplex.get_faces
            for k, v in angles.items():
                if k in data:
                    data[k].append(v) 
                else:
                    data[k] = [v]
               
            if "adjacent face" in data:
                data["adjacent face"].append(rnd_simplex.adjacent_face) 
            else:
                data["adjacent face"] = [rnd_simplex.adjacent_face]

        return pd.DataFrame(data)

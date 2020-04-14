import math
import numpy as np

def angle(v, w, metric_tensor=None, x=None):
    assert len(v) == len(w), 'Vectors must have same length!'

    if metric_tensor == None:
        return math.acos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))
    else:
        assert len(x) == len(v), 'Base point x must have same length as v and w!'

        g_vw_x = metric_tensor(x, v, w)
        g_vv_x = metric_tensor(x, v, v)
        g_ww_x = metric_tensor(x, w, w)

        return math.acos(g_vw_x / (math.sqrt(g_vv_x * g_ww_x)))
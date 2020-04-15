import math
import numpy as np
import collections
from scipy import optimize as opt

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

class FindIdealPoints:
    def __init__(self, p, q):
        assert len(p) == 2 and len(q) == 2, 'Input vectors must have length 2!'
        assert not np.array_equal(p, q), 'p and q must be different!'  
        assert np.linalg.norm(p) < 1.0 and  np.linalg.norm(q) < 1.0, 'p and q must be contained in open unit disc!'  
        self._p = p
        self._q = q

    def find(self):
        def _lin_qp(t):
            return self._p - t * (self._q - self._p)
        
        def _f(t):
            r = np.linalg.norm(_lin_qp(t))
            return r * r - 1.0

        dist = np.linalg.norm(self._p - self._q)
        alpha = math.sqrt(2.0) / dist + 0.1
        root_p = opt.brentq(lambda s: _f(s), .0, + alpha)
        root_m = opt.brentq(lambda s: _f(s), .0, - alpha)
        v = collections.OrderedDict({ .0 : self._p, 1.0 : self._q, root_p : _lin_qp(root_p), root_m : _lin_qp(root_m)})

        return list(v.values())[0], list(v.values())[1], list(v.values())[2], list(v.values())[3] 
        
      


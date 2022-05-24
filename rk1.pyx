import numpy as np
cimport numpy as np
cimport cython

cdef class rk:

        cdef:
               np.ndarray initial, interval

        def __init__(self, initial, interval):
                self.initial = initial
                self.interval = interval

        @cython.boundscheck(False)
        @cython.wraparound(False)
        @cython.initializedcheck(False)

        cpdef solve(self, farg):
                cdef:
                        int i
                        double h
                        np.ndarray x, k1, k2, k3, k4, a

                x = np.linspace(self.interval[0], self.interval[1], 1000)
                a = np.ones([x.shape[0], self.initial.shape[0]])
                h = x[1]-x[0]
                a[0] = self.initial

                for i in range(a.shape[0]-1):
                        k1 = farg(a[i], x[i])
                        k2 = farg(a[i] + k1 * h / 2., x[i] + h / 2.)
                        k3 = farg(a[i] + k2 * h / 2., x[i] + h / 2.)
                        k4 = farg(a[i] + k3 * h, x[i] + h)
                        a[i+1] = a[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)

                return np.vstack((x,np.transpose(a)))

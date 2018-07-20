


cpdef double[:] cython_numerical_model(self, double[:] T, float z, float dt, float q, float PwCw, float pc, float Ke, int n_iterations, double[:] top_bc, double[:] bot_bc):
    '''

    :param T:
    :param z:
    :param dt:
    :param q:
    :param PwCw:
    :param pc:
    :param Ke:
    :param n_iterations:
    :param top_bc:
    :param bot_bc:
    :return:
    '''
    cdef int i
    for i in range(n_iterations):
        T[0] = top_bc[i]
        T[-1] = bot_bc[i]
        T[1:-1] = self.equation(T, z, dt, q, PwCw, pc, Ke)
    return T
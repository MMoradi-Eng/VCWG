import numpy
import math

# This class is used to invert and resolve a tri-diagonal matrix
class Invert:
    def __init__(self,nz,A,RHS):
        self.nz = nz
        self.A = A
        self.RHS = RHS

    def Output(self):
        X = numpy.zeros(self.nz)
        for i in range(self.nz-2,-1,-1):
            self.RHS[i] = self.RHS[i]-self.A[i][2]*self.RHS[i+1]/self.A[i+1][1]
            self.A[i][1] = self.A[i][1]-self.A[i][2]*self.A[i+1][0]/self.A[i+1][1]

        for i in range(1,self.nz):
            self.RHS[i] = self.RHS[i]-self.A[i][0]*self.RHS[i-1]/self.A[i-1][1]

        for i in range(0,self.nz):
            X[i] = self.RHS[i]/self.A[i][1]

        return X
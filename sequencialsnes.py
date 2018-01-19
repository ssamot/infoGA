__author__ = 'Spyridon Samothrakis ssamot@essex.ac.uk'

from snes import SNES
import numpy as np
from random import randint

class SSNES():
    def __init__(self, x0, learning_rate_mult, popsize):
        self.snes = SNES(x0,learning_rate_mult,popsize)
        self.gen()

    def gen(self):
        self.asked = self.snes.ask()
        self.scores = {n:[] for n in range(len(self.asked))}


    def predict(self):
        r = randint(0,len(self.asked)-1)
        asked = self.asked[r]
        return asked, r


    def fit(self, scores, r):

        # sort them out
        for i, score in enumerate(scores):
            self.scores[r[i]].append(score)

        told = []
        for i in range(len(self.asked)):
            told.append(np.array(self.scores[i]).mean())

        self.snes.tell(self.asked,told)
        self.gen()

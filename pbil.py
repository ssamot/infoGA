__author__ = 'Spyridon Samothrakis ssamot@essex.ac.uk'

import numpy as np

# shamelessly stollen from wikipedia

def int2bin(n):
	'From positive integer to list of binary bits, msb at index 0'
	if n:
		bits = []
		while n:
			n,remainder = divmod(n, 2)
			bits.insert(0, remainder)
		return bits
	else: return [0]


def bin2int(bits):
	'From binary bits, msb at index 0 to integer'
	i = 0
	for bit in bits:
		i = i * 2 + bit
	return i


def bin2gray(bits):
    return bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]

def gray2bin(bits):
    b = [bits[0]]
    for nextb in bits[1:]: b.append(b[-1] ^ nextb)
    return b

class PBIL():
    def __init__(self, bits, n_individuals):
        self.bits = bits
        self.n_individtuals = n_individuals
        self.samples = np.random.random((n_individuals,bits))
        self.probvec = np.ones(bits,dtype=np.float32)*0.5
        self.lr = 0.1
        self.nlr = 0.075
        self.mutProb = 0.02
        self.mutShift = 0.05
        self.best_v = np.inf
        self.best_g = self.samples[0]



    def ask(self):

        self.samples = [(np.random.binomial(1,p, self.n_individtuals))for p in self.probvec]
        self.samples = np.array(self.samples, dtype=np.float32).T
        #print self.samples.shape
        return self.samples

    def tell(self, samples, fits):
        # Update the probability vector with max and min cost genes
        fits = np.array(fits)
        f_argmin = fits.argmin()
        f_argmax = fits.argmax()

        if(fits.min() < self.best_v ):
            self.best_g = samples[fits.argmin()]
            self.best_v = fits.min()


        for i in range(len(samples[f_argmin])):
            mingene = samples[f_argmin][i]
            maxgene = samples[f_argmax][i]
            if(mingene == maxgene):
                self.probvec[i] = self.probvec[i]* (1.0 - self.lr)  +  mingene*self.lr
            else:
                lr2 = self.lr + self.nlr
                self.probvec[i] = self.probvec[i]* (1.0 - lr2) +  mingene*lr2


        # mutate

        for i in range(self.bits):
            if(np.random.random() < self.mutProb):
                self.probvec[i] = self.probvec[i]*(1.0-self.mutShift) + np.random.randint(2)*self.mutShift


class PBILSeperable():
    def __init__(self, bits, n_individuals, space):
        self.pbils = [PBIL(bits,n_individuals) for i in range(space)]
        self.space = space
        self.bits = bits
        self.n_individuals = n_individuals

    def ask(self):
        total = []
        for pbil in self.pbils:
            total.append(pbil.ask())
        total = np.array(total)
        #print total.shape, "shape"
        return total

    def tell(self, samples, fits):

        for i, pbil in enumerate(self.pbils):
            pbil.tell(samples[i], fits[i])

        self.best_g = []
        self.best_v = []
        for i in range(self.space):
            self.best_g.extend(self.pbils[i].best_g)
            self.best_v.append(self.pbils[i].best_v)
        self.best_v = np.array(self.best_v).sum()




if __name__=="__main__":

    pbil = PBILSeperable(10, 100, 1000)


    for i in range(10000):
        #print pbil.probvec
        f = []
        samples = pbil.ask()
        for sep in samples:
            s_f = []
            for sample in sep:
                s_f.append(-np.sum(sample))
            f.append(s_f)

        m = np.array(f).mean()
        for j in range(len(f)):
            f[j][0] = m
        print "Iteration",i

        print sample.shape, len(f)

        pbil.tell(samples,f)
        #bin2int(gray2bin([bool(pbil.best_g[l]) for l in range(len(pbil.best_g))]))
        print np.array(f).mean(), pbil.best_v,i,

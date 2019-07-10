import numpy as np
import itertools
import image_transforms

def DeepHunter(I, coverage, K=10):
    F = []
    T = Preprocess(I)
    B, B_id = SelectNext(T)
    while B:
        S = Sample(B)
        Ps = PowerSchedule(S, K)
        B_new = []
        for I in S:
            for i in range(1, Ps(I)+1):
                I_new = Mutate(I)
                if isFailedTest(I_new):
                    F += I_new
                elif isChanged(I, I_new):
                    B_new += I_new

        cov = Predict(coverage, B_new)
        if CoverageGain(cov):
            T += B_new
            BatchPrioritize(T, B_id)
        B, B_id = SelectNext(T)

def Preprocess(I):
    Bs = np.split(np.random.shuffle(I), 64)
    return np.ones(len(Bs)), Bs

def calc_priority(B_ci, p_min=0.01, gama=0.9):
    if B_ci < (1-p_min) * gama:
        return 1 - B_ci / gama
    else:
        return p_min

def SelectNext(T, batch_size=64):
    B_c, Bs = T
    B_p = [calc_priority(B_c[i]) for i in range(len(B_c))]
    c = np.random.choice(len(Bs), size=batch_size, replace=False, p=B_p/np.sum(B_p))
    return Bs[c], c

def Sample(B, batch_size=16):
    c = np.random.choice(len(B), size=batch_size, replace=False)
    return B[c]

class INFO:
    def __init__(self):
        self.dict = {}
    
    def __getitem__(self, i):
        _i =str(i)
        if _i in self.dict:
            return self.dict[_i]
        else:
            I0, I0_new, state = i, i, 0
            return I0, I0_new, state

    def __setitem__(self, i, s):
        _i = str(i)
        self.dict[_i] = s
        return self.dict[_i]

info = INFO() 

def PowerSchedule(S, K, beta=0.1):
    global info
    potentials = []
    for i in range(len(S)):
        I = S[i]
        I0, I0_new, state = info[I]
        p = beta * 255 * I.size - sum(np.abs(I - I0_new))
        potentials.append(p)
    potentials = np.array(potentials) / np.sum(potentials)

    def Ps(I):
        i = S.index(I)
        p = potentials[i]
        return int(p*K)
    
    return Ps


def isFailedTest(I_new):
    return False

def isChanged(I, I_new):
    return np.any(I != I_new)

def Predict(coverage, B_new):
    _, cov = coverage.step(B_new)
    return cov

def CoverageGain(cov):
    return cov > 0

def BatchPrioritize(T, B_id):
    B_c, Bs = T
    B_c[B_id] += 1


translation = list(itertools.product([getattr(image_transforms,"image_translation")], [(10+10*k,10+10*k) for k in range(10)]))
scale = list(itertools.product([getattr(image_transforms, "scale")], [(1.5+0.5*k,1.5+0.5*k) for k in range(10)]))
shear = list(itertools.product([getattr(image_transforms, "shear")], [(-1.0+0.1*k,0) for k in range(10)]))
rotation = list(itertools.product([getattr(image_transforms, "rotation")], [3+3*k for k in range(10)]))
contrast = list(itertools.product([getattr(image_transforms, "contrast")], [1.2+0.2*k for k in range(10)]))
brightness = list(itertools.product([getattr(image_transforms, "brightness")], [10+10*k for k in range(10)]))
blur = list(itertools.product([getattr(image_transforms, "blur")], [k+1 for k in range(10)]))

G = translation + scale + shear + rotation
P = contrast + brightness + blur

def Mutate(I, TRY_NUM=100):
    global info
    I0, I0_new, state = info[I]
    for i in range(1, TRY_NUM):
        if state == 0:
            t, p = randomPick(G + P)
        else:
            t, p = randomPick(P)

        I_new = t(I, p)
        if f(I0_new, I_new):
            if (t, p) in G:
                state = 1
                I0_new = t(I0, p)
            info[I_new] = (I0, I0_new, state)
            return I_new

    return I

def randomPick(A):
    c = np.random.randint(0, len(A))
    return A[c]

def f(I, I_new, beta=0.1, alpha=0.1):
    if(np.sum(np.abs(I-I_new)) < alpha * I.size):
        return np.all(I-I_new <= 255)
    else:
        return np.all(I-I_new <= beta*255)
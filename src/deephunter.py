import numpy as np
from src.utility import init_image_plots, update_image_plots

class INFO:
    def __init__(self):
        self.dict = {}

    def __getitem__(self, i):
        _i = str(i)
        if _i in self.dict:
            return self.dict[_i]
        else:
            I0, I0_new, state = np.copy(i), np.copy(i), 0
            return I0, I0_new, state

    def __setitem__(self, i, s):
        _i = str(i)
        self.dict[_i] = s
        return self.dict[_i]


class DeepHunter:
    def __init__(self, params, experiment):
        self.params = params
        self.experiment = experiment
        self.info = INFO()
        self.last_coverage_state = None
        self.input_shape = params.input_shape
        
    def run(self):
        I = self.experiment.dataset["test_inputs"]
        if self.params.image_verbose:
            self.f_current = init_image_plots(1, 1, I.shape)
            self.f_best = init_image_plots(8, 8, I.shape)
        

        F = np.array([]).reshape(0, *(self.input_shape[1:]))
        T = self.Preprocess(I)
        B, B_id = self.SelectNext(T)

        counter = 0
        while B is not None:
            S = self.Sample(B)
            if self.params.save_batch:
                counter += 1
                np.save("data/deephunter_{}".format(counter), S)

            Ps = self.PowerSchedule(S, self.params.K)
            B_new = np.array([]).reshape(0, *(self.input_shape[1:]))
            for s_i in range(len(S)):
                I = S[s_i]
                for i in range(1, Ps(s_i) + 1):
                    I_new = self.Mutate(I)
                    if self.isFailedTest(I_new):
                        F += np.concatenate((F, [I_new]))
                    elif self.isChanged(I, I_new):
                        B_new = np.concatenate((B_new, [I_new]))

            if len(B_new) > 0:
                cov = self.Predict(B_new)
                
                if self.params.verbose:
                    print("coverage increase:", cov)

                if self.params.image_verbose:
                    title = "Coverage Increase: " + str(cov)
                    update_image_plots(self.f_best, B_new, title)

                if self.CoverageGain(cov):
                    self.experiment.coverage.step(B_new, update_state=True, coverage_state=self.last_coverage_state)
                    print("coverage:", self.experiment.coverage.get_current_coverage())
                    B_c, Bs = T
                    B_c += [0]
                    Bs += [B_new]
                    self.BatchPrioritize(T, B_id)

            B, B_id = self.SelectNext(T)


    def Preprocess(self, I):
        _I = np.random.permutation(I)
        Bs = np.array_split(_I, range(self.params.batch1, len(_I), self.params.batch1))
        return list(np.zeros(len(Bs))), Bs


    def calc_priority(self, B_ci):
        if B_ci < (1 - self.params.p_min) * self.params.gamma:
            return 1 - B_ci / self.params.gamma
        else:
            return self.params.p_min

    def SelectNext(self, T):
        B_c, Bs = T
        B_p = [self.calc_priority(B_c[i]) for i in range(len(B_c))]
        c = np.random.choice(len(Bs), p=B_p / np.sum(B_p))
        return Bs[c], c


    def Sample(self, B):
        c = np.random.choice(len(B), size=self.params.batch2, replace=False)
        return B[c]


    def PowerSchedule(self, S, K):
        potentials = []
        for i in range(len(S)):
            I = S[i]
            I0, I0_new, state = self.info[I]
            p = self.params.beta * 255 * np.sum(I > 0) - np.sum(np.abs(I - I0_new))
            potentials.append(p)
        potentials = np.array(potentials) / np.sum(potentials)

        def Ps(I_id):
            p = potentials[I_id]
            return int(np.ceil(p * K))

        return Ps


    def isFailedTest(self, I_new):
        return False


    def isChanged(self, I, I_new):
        return np.any(I != I_new)


    def Predict(self, B_new):
        print("Predict B_new.shape", np.array(B_new).shape)
        self.last_coverage_state, cov = self.experiment.coverage.step(B_new, update_state=False)
        return cov


    def CoverageGain(self, cov):
        return cov > 0


    def BatchPrioritize(self, T, B_id):
        B_c, Bs = T
        B_c[B_id] += 1


    def Mutate(self, I):
        G, P = self.params.G, self.params.P
        I0, I0_new, state = self.info[I]
        for i in range(1, self.params.TRY_NUM):
            if state == 0:
                t, p = self.randomPick(G + P)
            else:
                t, p = self.randomPick(P)

            I_new = t(np.copy(I), p).reshape(*(self.input_shape[1:]))
            I_new = np.clip(I_new, 0, 255)

            if self.params.image_verbose:
                title = ""
                update_image_plots(self.f_current, I_new.reshape(*self.input_shape), title)
                    
            if self.f(I0_new, I_new):
                if (t, p) in G:
                    state = 1
                    I0_new = t(np.copy(I0), p)
                self.info[I_new] = (np.copy(I0), np.copy(I0_new), state)
                return I_new

        return I


    def randomPick(self, A):
        c = np.random.randint(0, len(A))
        return A[c]


    def f(self, I, I_new):
        if (np.sum((I - I_new) != 0) < self.params.alpha * np.sum(I > 0)):
            return np.max(np.abs(I - I_new)) <= 255
        else:
            return np.max(np.abs(I - I_new)) <= self.params.beta * 255

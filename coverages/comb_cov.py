import sys
sys.path.append('../')
sys.path.append('./lrp_toolbox/')
import os
import itertools
import numpy as np
from pyflann import *
from sklearn import cluster
from datetime import datetime
from utils import save_quantization, load_quantization
from utils import save_max_comb, load_max_comb, get_layer_outs_new)

from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

experiment_folder = 'experiments'
model_folder      = 'neural_networks'

class CombCoverage:
    def __init__(model, model_name, num_relevant_neurons, selected_class, subject_layer,
                      train_inputs, quantization_granularity)
        self.covered_combinations = ()

        self.model = model
        self.model_name = model_name
        self.num_relevant_neurons = num_relevant_neurons
        self.selected_class = selected_class
        self.subject_layer = subject_layer
        self.quantization_granularity = quantization_granularity
        self.train_inputs = train_inputs


    def get_measure_state(self):
        return self.covered_combinations

    def set_measure_state(self, covered_combinations):
        self.covered_combinations = covered_combinations

    def test(self, test_inputs):
        #########################
        #1.Find Relevant Neurons#
        #########################

        try:
            relevant_neurons = load_layerwise_relevances('%s/%s_%d_%d_%d'
                                                            %(experiment_folder,
                                                            self.model_name,
                                                            self.num_relevant_neurons,
                                                            self.elected_class,
                                                            self.subject_layer))

        except:
            # Convert keras model into txt
            model_path = model_folder + '/' + model_name
            write(model_path, model_path, 'keras_txt')

            lrpmodel = read(model_path + '.txt', 'txt')  # 99.16% prediction accuracy
            lrpmodel.drop_softmax_output_layer()  # drop softnax output layer for analysis

            relevant_neurons, least_relevant_neurons, total_R = find_relevant_neurons(
                self.model, lrpmodel, X_train, Y_train, self.subject_layer,
                self.num_relevant_neurons, 'alphabeta', 'sum')


            save_layerwise_relevances(total_R, '%s/%s_%s_%d'
                                      %(experiment_folder, self.model_name,
                                        'total_R', self.selected_class))

            save_layerwise_relevances(relevant_neurons, '%s/%s_%d_%d_%d'
                                        %(experiment_folder, self.model_name,
                                        self.num_relevant_neurons, self.selected_class,
                                        self.subject_layer))

            save_layerwise_relevances(least_relevant_neurons, '%s/%s_%d_%d_%d_least'
                                        %(experiment_folder, self.model_name,
                                        self.num_relevant_neurons, self.selected_class,
                                        self.subject_layer))


        #FOR CIFAR MODEL TODO: Should be automatically handled.
        subject_layer -= 1
        print(relevant_neurons)


        ####################################
        #2.Quantize Relevant Neuron Outputs#
        ####################################
        if 'conv' in model.layers[self.subject_layer].name: is_conv = True
        else: is_conv = False


        train_layer_outs = get_layer_outs_new(model, self.train_inputs)
        qtized = quantize(train_layer_outs[self.subject_layer], conv, relevant_neurons, self.quantization_granularity)


        ####################
        #3.Measure coverage#
        ####################

        test_layer_outs = get_layer_outs_new(model, test_inputs)

        coverage, cov_combs = measure_combinatorial_coverage(self.model, self.model_name,
                                                                test_inputs, self.subject_layer,
                                                                relevant_neurons,
                                                                self.selected_class,
                                                                test_layer_outs, qtized,
                                                                self.quantization_granularity,
                                                                is_conv, self.covered_combinations)


def quantize(out_vectors, conv, relevant_neurons, n_clusters=3):
    if conv: n_clusters+=1
    quantized_ = []

    for i in range(out_vectors.shape[-1]):
        out_i = []
        for l in out_vectors:
            if conv: #conv layer
                out_i.append(np.mean(l[...,i]))
            else:
                out_i.append(l[i])

        #If it is a convolutional layer no need for 0 output check
        if not conv: out_i = filter(lambda elem: elem != 0, out_i)
        values = []
        if not len(out_i) < 10: #10 is threshold of number positives in all test input activations
            kmeans = cluster.KMeans(n_clusters=n_clusters)
            kmeans.fit(np.array(out_i).reshape(-1, 1))
            values = kmeans.cluster_centers_.squeeze()
        values = list(values)

        if not conv: values.append(0) #If it is convolutional layer we dont add  directly since thake average of whole filter.

        quantized_.append(values)

    quantized_ = [quantized_[rn] for rn in relevant_neurons]

    return quantized_


def determine_quantized_cover(lout, quantized):
    covered_comb = []
    for idx, l in enumerate(lout):
        if l == 0:
            covered_comb.append(0)
        else:
            closest_q = min(quantized[idx], key=lambda x:abs(x-l))
            covered_comb.append(closest_q)

    return covered_comb


def measure_combinatorial_coverage(model, model_name, test_inputs, subject_layer,
                                   relevant_neurons, sel_class,
                                   test_layer_outs, qtized, q_granularity, conv,
                                   covered_combinations=()):

    num_increase = 0

    distance_threshold = 0
    for q in qtized:
        distance_threshold += max(q)**2 #Since we consider RELU the min is 0.
    distance_threshold /= d_var

    #print("distance threshold is: " + str(distance_threshold))
    #print("Num of test inputs: " + str(len(test_inputs)))
    #distance_threshold = 3


    for test_idx in range(len(test_inputs)):
        if conv:
            lout = []
            for r in relevant_neurons:
            '''
            for covered_comb in covered_combinations:
                #distance = cdist(np.array([covered_comb]), np.array([comb_to_add]),
                #                 metric='cityblock')[0][0]
                distance = np.linalg.norm(np.array(covered_comb)-np.array(comb_to_add))
                distance = distance**2
                if distance < distance_threshold:
                    distant = False
                    break

            if distant:
                covered_combinations.append(comb_to_add)
                num_increase += 1
            '''
        else:
            covered_combinations += (comb_to_add,)

    print(relevant_neurons)
    if os.path.isfile('%s/%s_%d_%d_%d_%d_%d_max_comb.h5' %(experiment_folder,
                                                           model_name, sel_class,
                                                           q_granularity, distance_threshold,
                                                           subject_layer,
                                                           len(relevant_neurons))):

        max_comb = load_max_comb('%s/%s_%d_%d_%d_%d_%d' %(experiment_folder,
                                                          model_name, sel_class,
                                                          q_granularity, distance_threshold,
                                                          subject_layer,
                                                          len(relevant_neurons)))[0]
    else:
        all_combs = itertools.product(*qtized)
        reduced_combs = np.array([all_combs.next()]).astype('float32')


    max_comb = (q_granularity+1)**len(relevant_neurons)

    print("Number of covered combinations: " + str(len(covered_combinations)))
    covered_num = len(covered_combinations)
    coverage = float(covered_num)/max_comb
    #print("Coverage ratio: " + str(coverage))

    return coverage*100, covered_combinations


def find_relevant_neurons(kerasmodel, lrpmodel, inps, outs, subject_layer, \
            num_rel, lrpmethod=None, final_relevance_method='sum'):

    final_relevants = np.zeros([1, kerasmodel.layers[subject_layer].output_shape[-1]])

    total_R = None

    cnt = 0
    for inp in inps:
        cnt+=1
        ypred = lrpmodel.forward(np.expand_dims(inp, axis=0))

        #If not correctly classified do not take into account. UPDATE: This check is done in main code.
        #if not np.argmax(ypred) == np.argmax(outs):
        #    continue

        #prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
        mask = np.zeros_like(ypred)
        mask[:,np.argmax(ypred)] = 1
        Rinit = ypred*mask

        if not lrpmethod:
            R_inp, R_all = lrpmodel.lrp(Rinit)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == 'epsilon':
            R_inp, R_all = lrpmodel.lrp(Rinit,'epsilon',0.01)    #as Eq(58) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == 'alphabeta':
            R_inp, R_all = lrpmodel.lrp(Rinit,'alphabeta',2)     #as Eq(60) from DOI: 10.1371/journal.pone.0130140
        else:
            print('Unknown LRP method!')
            raise Exception

        R_inp = R_inp[0, 2:-2, 2:-2, 0]


        '''
        nonzero_rel_idx = np.where(R_inp>0.1)

        #DENSE:
        rel_idx = np.argsort(R_all[subject_layer][0])[::-1][:num_rel]

        #CONV:
        #for num_neuron in xrange(final_relevants.shape[-1]):
        #    final_relevants[0][num_neuron] = np.mean(R_all[subject_layer][..., num_neuron])

        #rel_idx = np.argsort(final_relevants)[0][::-1][:num_rel]

        return rel_idx, nonzero_rel_idx
        '''

        if total_R: totalR += R_all
        else: totalR = R_all

        if kerasmodel.layers[subject_layer].__class__.__name__ == 'Dense' and \
                          final_relevance_method == 'sum':
            final_relevants += R_all[subject_layer][0]
        elif kerasmodel.layers[subject_layer].__class__.__name__ == 'Dense' and \
                            final_relevance_method == 'count':
            rel_idx = np.argsort(R_all[subject_layer][0])[::-1][:num_rel]
            for fr_idx, _ in enumerate(final_relevants[0]):
                if fr_idx in rel_idx:
                    final_relevants[0][fr_idx] += 1
        elif kerasmodel.layers[subject_layer].__class__.__name__ == 'Conv2D' and \
                            final_relevance_method == 'sum':
            for num_neuron in xrange(final_relevants.shape[-1]):
                final_relevants[0][num_neuron] = np.mean(R_all[subject_layer][0][..., num_neuron])
        elif kerasmodel.layers[subject_layer].__class__.__name__ == 'Conv2D' and \
                            final_relevance_method == 'count':
            avgs = []
            for idx in xrange(R_all[subject_layer].shape[-1]):
                avgs.append(np.mean(R_all[subject_layer][0][..., idx]))

            rel_idx = np.argsort(avgs)[::-1][:num_rel]
            for fr_idx, _ in enumerate(final_relevants[0]):
                if fr_idx in rel_idx:
                    final_relevants[0][fr_idx] += 1
            #final_relevants = [fr+1 if fr in rel_idx else fr for fr in final_relevants]

    #      THE MOST RELEVANT                               THE LEAST RELEVANT
    return np.argsort(final_relevants)[0][::-1][:num_rel], np.argsort(final_relevants)[0][:num_rel], totalR




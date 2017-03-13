from __future__ import print_function
import os
import pickle
from .preprocess import CharacterDictionary
from collections import defaultdict

import tensorflow as tf
from keras.engine.topology import Layer
from keras import layers
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras import initializations, regularizers, constraints
from keras.engine import InputSpec

import threading

import codecs
import itertools
import numpy as np
"""
NER module 
"""

class TransLayer(Layer):
    """
    This layer combines output from previous layers in the shape of nsamples * nsteps * nlabels with the 
    transitional matrix (nlabels * nlabels + 2) between labels and output a tensor in the shape of nsamples * nsteps * nlabels
    
    The weight of this layer is the transitional matrix A
    A_ji is the log probability of label j following label i
    A_j0 is the log probability of label j to be the beginning label
    A_j-1 is the log probability of label j to be the endding label
    """
    def __init__(self, output_dim, init='glorot_uniform', weights=None,
                         W_regularizer=None, activity_regularizer=None,
                         W_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        
        self.initial_weights = weights
        
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)        
        super(TransLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        assert input_dim == self.output_dim
        self.input_dim = input_dim   
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]        
        self.W = self.add_weight((self.output_dim, self.output_dim + 2),
                                                  initializer=self.init,
                                                  name='{}_W'.format(self.name),
                                                  regularizer=self.W_regularizer,
                                                  constraint=self.W_constraint)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
    
    def call(self, x, mask=None):
        output = tf.expand_dims(x, -1) + tf.expand_dims(tf.expand_dims(self.W, 0), 0)
        return output
    
    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape.append(self.output_dim + 2)
        return tuple(output_shape)
    
    def get_config(self):
        config = {"output_dim" : self.output_dim, "init" : self.init.__name__, 
                         "W_regularizer" : self.W_regularizer.get_config() if self.W_regularizer else None, 
                         "activity_regularizer" : self.activity_regularizer.get_config() if self.activity_regularizer else None,
                         "W_constraint" : self.W_constraint.get_config if self.W_constraint else None,
                         "input_dim" : self.input_dim}#"weights" : self.weights,
        base_config = super(TransLayer, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))
        

    
def test_model():
    input_l = layers.Input(shape=(None, 5))
    t_l = TransLayer(output_dim=5)(input_l)
    model = Model(input=input_l, output=t_l)
    model.compile(loss="mse", optimizer="adam")
    return model
    

def seq_loss(y_true, y_pred):
    """
    this calculate the total loss of a sequence
    input:
        y_true: the ground truth in the shape of nsamples * nsteps
        y_pred: the predicted scores in the shape of nsamples * nsteps * nlabels * nlabels + 2, the output of the TransLayer
    """
    
    yshapes = tf.shape(y_pred)
    nsamples = yshapes[0]
    nsteps = yshapes[1]
    nlabels = yshapes[2]
    
    y_true = tf.slice(y_true, [0, 0, 0, 0], [nsamples, nsteps, 1, 1])
    
    idx1 = tf.reshape(tf.tile(tf.reshape(tf.range(nsamples), [-1, 1]), [1, nsteps]), [-1])
    idx2 = tf.reshape(tf.tile(tf.reshape(tf.range(nsteps), (-1, 1)), [nsamples, 1]), [-1])
    idx3 = tf.reshape(tf.cast(tf.gather_nd(y_true, tf.stack([idx1, idx2], axis=1)), tf.int32), [-1]) #y(t)

    idx22 = tf.select(tf.equal(idx2, 0), tf.zeros([nsamples * nsteps, ], dtype=tf.int32), idx2-1) #this is just to get rid of negative index
    idx4 = tf.reshape(tf.gather_nd(y_true, tf.stack([idx1, idx22], axis=1)), [-1]) #y(t-1)

    idx4 = tf.select(tf.equal(idx2, 0), tf.zeros([nsamples * nsteps], dtype=tf.int32) + nlabels, tf.cast(idx4, tf.int32)) #for the beginning of the sequence, the preceding label is always the special one, coded as the nlabels + 1 element in the transision matrix
    py_true = tf.reduce_sum(tf.gather_nd(y_pred, tf.stack([idx1, idx2, idx3, idx4], axis=1)) )
    # now for the last label in each sequence
    idx1 = tf.range(nsamples)#tf.constant(range(nsamples), dtype="int64")
    idx2 = tf.zeros([nsamples], dtype=tf.int32) + nsteps - 1#tf.constant([nsteps - 1] * nsamples, dtype="int64")
    idx3 = tf.reshape(tf.cast(tf.gather_nd(y_true, tf.stack([idx1, idx2], axis=1)), tf.int32), [-1])
    idx4 = tf.zeros([nsamples],dtype=tf.int32) + nlabels + 1#tf.constant([nlabels + 1] * nsamples, dtype="int64")
    py_true = py_true + tf.reduce_sum(tf.gather_nd(y_pred, tf.stack([idx1, idx2, idx3, idx4], axis=1)))
    
    delta = tf.slice(y_pred, [0, 0, 0, nlabels], [nsamples, 1, nlabels, 1]) # first time step
    
    i = tf.Variable(1)
    c = lambda i, delta:tf.less(i, nsteps)
    def body(i, delta):
        delta = tf.reduce_logsumexp(tf.reshape(delta, [nsamples, 1, 1, nlabels]) + tf.slice(y_pred, [0, i, 0, 0], [nsamples, 1, nlabels, nlabels]), axis=3, keep_dims=True)
        i = tf.add(i, 1)
        return i, delta
    i, delta = tf.while_loop(c, body, [i, delta])
    delta = delta + tf.slice(y_pred, [0, nsteps-1, 0, nlabels+1], [nsamples, 1, nlabels, 1]) # last time step
    delta = tf.reduce_logsumexp(delta, axis=2, keep_dims=True)
    
    py_true = -py_true + tf.reduce_sum(delta)
    return tf.cast(py_true, tf.float32) / tf.cast(nsamples, tf.float32) / tf.cast(nsteps, tf.float32)
        
def Viterbi_decode(y_pred):
    """
    this function searches for the optimal path based on the predicted scores using Viterbi algorithm
    input:
        y_pred: predicted scores in the shape of nsamples * nsteps * nlabels * nlabels + 2, the output of the TransLayer
    """
    nsamples, nsteps, nlabels, _ = y_pred.shape
    T2 = np.zeros((nsamples, nsteps, nlabels))
    t1 = y_pred[:, 0, :, -2] # first step, nsamples * nlabels
    for i in range(1, nsteps):
        t1 = t1[:, None, :] + y_pred[:, i, :, :-2] #nsamples * nlabels * nlabels
        T2[:, i, :] = np.argmax(t1, axis=2)
        t1 = np.max(t1, axis=2) #nsamples * nlabels
    t1 = t1 + y_pred[:, -1, :, -1] # last step
    lastlabel = np.argmax(t1, axis=1) #(nsamples, )
    labels = [lastlabel]
    for i in range(1, nsteps):
        lastlabel = T2[range(nsamples), nsteps - i, lastlabel.astype(np.int)]
        labels.append(lastlabel)
    
    labels = np.r_[labels[::-1]].T
    return labels

def define_model(dictsize, embedsize, lstmsize, densesize):
    input_l = layers.Input(shape=(None,))
    embed = layers.embeddings.Embedding(dictsize, embedsize, mask_zero=False)(input_l)
    lstm1 = layers.recurrent.LSTM(lstmsize, return_sequences=True)(embed)
    d1 = layers.Dense(densesize, activation="softmax")(lstm1)
    
    lstm2 = layers.recurrent.LSTM(lstmsize, return_sequences=True, go_backwards=True)(embed)
    d2 = layers.Dense(densesize, activation="softmax")(lstm2)
    
    merge = layers.merge([d1, d2], mode=lambda s: K.log(s[0]) + K.log(s[1]), output_shape=(None, densesize))#lambda s: K.log(s[0]) + K.log(s[1]))
    output = TransLayer(densesize)(merge)
    #output = layers.Activation("softmax")(merge)
    model = Model(input=input_l, output=output)
    model.compile(loss=seq_loss, optimizer="adam")
    #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", "precision", "recall"])
    return model

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generate_batch(dictionary, labeldict, infile, extractf, batch_len_inc=10, batchsize=1000, train=True):
    """
    generate batch !continuously! of samples from infile, samples of the similar length grouped in one batch
    extractf is the function to extract the text and corresponding labels from one line, assuming each line corresponds to one sentence
    the output of extractf should be a list of tuples [(token, label)...]
    """
    batches_x = defaultdict(list)
    batches_y = defaultdict(list)
    batches = {}
    i = 0
    while 1: #generate batch continuously
        print(infile)
        with codecs.open(infile, "r", "utf8") as fin:    
            for oneline_t in extractf(fin):
                i += 1
                #oneline_t = list(extractf(oneline))
                tokens = [dictionary.token2id.get(item[0], dictionary.dictsize + 1) for item in oneline_t] #unknown character mapped to dictsize + 1
                labels = [labeldict.token2id.get(item[1], labeldict.dictsize + 1) for item in oneline_t] #unknown label mapped to dictsize + 1
                thislen = len(tokens)
                if thislen == 0:
                    continue
                batches_x[thislen / batch_len_inc].append(tokens)
                batches_y[thislen / batch_len_inc].append(labels)
                if len(batches_x[thislen / batch_len_inc]) == batchsize:
                    tokens = sequence.pad_sequences(batches_x[thislen / batch_len_inc]) # sequence will be padded with 0
                    labels = sequence.pad_sequences(batches_y[thislen / batch_len_inc]) # sequence will be padded with 0
                    oshape = labels.shape
                    labels = np.repeat(labels, (labeldict.dictsize + 2) * (labeldict.dictsize + 2 + 2)).reshape(oshape[0], oshape[1], labeldict.dictsize + 2, labeldict.dictsize + 4)
                    #to_categorical(labels.reshape(labels.size, 1), nb_classes=labeldict.dictsize + 2).reshape(oshape[0], oshape[1], labeldict.dictsize + 2)
                    batches_x[thislen / batch_len_inc] = []
                    batches_y[thislen / batch_len_inc] = []
                    yield tokens, labels
        for key, item in batches_x.iteritems():
            if len(item) == 0:
                continue
            tokens = sequence.pad_sequences(item)
            labels = sequence.pad_sequences(batches_y[key])
            oshape = labels.shape
            labels = np.repeat(labels, (labeldict.dictsize + 2) * (labeldict.dictsize + 2 + 2)).reshape(oshape[0], oshape[1], labeldict.dictsize + 2, labeldict.dictsize + 4)
            #to_categorical(labels.reshape(labels.size, 1), nb_classes=labeldict.dictsize + 2).reshape(oshape[0], oshape[1], labeldict.dictsize + 2)
            yield tokens, labels
        if not train:
            break
                

supported_labels = ["PER", "LOC", "ORG", "OTH", "GPE"]
BIE_labels = ["B", "I", "E"]
BIEs = ["-".join(item) for item in itertools.product(BIE_labels, supported_labels)]

class NER(object):
    def __init__(self, dictionaryfile=os.path.join(os.path.split(os.path.abspath(__file__))[0], "v02032017_dict.pkl"), 
                               initfromfile=None, modelfile=os.path.join(os.path.split(os.path.abspath(__file__))[0], "v02032017_model.h5"), verbose=1, **kwargs):
        """
        NER tagging specialized for Chinese, supported labels are ["PER", "LOC", "ORG", "OTH", "GPE"]
        Parameters:
            dictionaryfile: str, the name of the file containing pretrained CharacterDictionary
            initfromfile: str, the name of the train data file to initialize the CharacterDictionary; 
                              one of dictionaryfile and initfromfile has to be specified, if both are specified then initfromfile will override dictionaryfile
            modelfile: str, the name of the file containing pretrained NER model
            verbose: bool, default True
            **kwargs: these kwargs will be passed to the initialization of the CharacterDictionary if initfromfile is specified
        """
        self.verbose = verbose
        if dictionaryfile is None and initfromfile is None:
            raise ValueError("either dictionaryfile or initfromfile should be specified to initialize the dictionary")
        
        self.ner_labels = CharacterDictionary([BIEs], startfrom=1, unknownC="O") # reserve 0 for empty label, for padding
        if initfromfile:
            if self.verbose:
                print("initialize character dictionary from %s"%initfromfile)
            with codecs.open(initfromfile, "r", "utf8") as fin:
                self.dictionary = CharacterDictionary(fin, preprocess=kwargs.get("preprocess", None), unknownC="-")
        else:
            if self.verbose:
                print("load character dictionary from %s"%dictionaryfile)
            self.dictionary = CharacterDictionary()
            self.dictionary.load(dictionaryfile)
        if self.verbose:
            print("dictionary size %s"%self.dictionary.dictsize)
        
        if modelfile:
            if self.verbose:
                print("load model from %s"%modelfile)
            self.model = load_model(modelfile, custom_objects={'TransLayer' : TransLayer, 'seq_loss' : seq_loss})
        else:
            self.model = None
    
    def save(self, outfolder, label):
        self.model.save(os.path.join(outfolder, '%s_model.h5'%label))
        self.dictionary.save(os.path.join(outfolder, '%s_dict.pkl'%label))
    
    
    def train(self, trainfile="1998-01-2003-utf8.txt", epoches=10, samples_per_epoch=17000, nb_worker=2, genbatchkwargs={}):
        """
        To train the NER model from a file
        Parameters:
            trainfile: str, path to the train file
            epoches: int, default 10, number of epoches to train
            samples_per_epoch: int, number of samples to process for each epoch
            nb_worker: int, number of processors to use. The above three parameters are passed to the fit_generator of a Keras model
            genbatchkwargs: dict, default empty, the parameters for generate_batch function, it should specify at least "extractf" for the function to generate tokens and labels of one sentence from the trainfile
        """
        if self.model is None:
            self.model = define_model(self.dictionary.dictsize + 2, 50, 100, self.ner_labels.dictsize + 2)
        batches = generate_batch(self.dictionary, self.ner_labels, trainfile, **genbatchkwargs) # this is a generator
        self.model.fit_generator(batches, samples_per_epoch=samples_per_epoch, nb_epoch=epoches, nb_worker=nb_worker, verbose=self.verbose)
        
    def test(self, testfile="", outfile="", genbatchkwargs={}):
        """
        To test the NER model from a file
        Parameters:
            testfile: str, path to the test file
            outfile: str, path to the output file, this file will be in the conlleval format, so that the conlleval.pl can be directly used to evaluate the performance
            genbatchkwargs: dict, default empty, the parameters for generate_batch function, it should specify at least "extractf" for the function to generate tokens and labels of one sentence from the trainfile
        """
        batches = generate_batch(self.dictionary, self.ner_labels, testfile, train=False, **genbatchkwargs) # this is a generator
        with codecs.open(outfile, "w", "utf8") as fout:
            if self.verbose:
                print( "write to %s"%outfile)
        for tokens, labels in batches:
            plabels = self.model.predict_on_batch(tokens)
            plabels = self.ner_labels.inversetransform(Viterbi_decode(plabels))
            #plabels = self.ner_labels.inversetransform(plabels.argmax(axis=-1))
            labels = labels[:, :, 0, 0]
            labels = self.ner_labels.inversetransform(labels)
            rtokens = self.dictionary.inversetransform(tokens)            
            with codecs.open(outfile, "a", "utf8") as fout:    
                for i, sent in enumerate(tokens):
                    for j, token in enumerate(sent):
                        if token == 0:
                            continue
                        fout.writelines("%s %s %s\n"%(rtokens[i][j], labels[i][j], plabels[i][j]))
                    fout.writelines("\n")
            
        
                
    def _predict(self, iterable):
        """
        iterable is an iterable of unicode text, for the seek of performance, do not pad, will process the sentences of the same length together
        """
        tokens = np.array(self.dictionary.transform(iterable))
        sentlen = np.array([len(item) for item in tokens])
        out = np.empty(tokens.shape, dtype='object')
        for uniqL in np.unique(sentlen):
            idx = sentlen == uniqL
            thistokens = np.atleast_2d(np.array(tokens[idx].tolist()))
            thislabels = self.model.predict(thistokens)
            olabels = self.ner_labels.inversetransform(Viterbi_decode(thislabels))
            #raise
            out[idx] = np.array(olabels + [[]])[:-1]
        return out

    def predict(self, iterable):
        """
        Extract named entities from the given data
        Parameters:
            iterable: iterable of unicode text, the input data. 
        Returns:
            Text tagged with format like "<PER>xxx</PER>"
        """
        labels = self._predict(iterable)
        #labels = self.ner_labels.inversetransform(labels)
        return self._print_out(iterable, labels)
        
    def _print_out(self, iterable, labels):
        out = []
        compressed_labels = set([item.split("-")[-1] for item in self.ner_labels.token2id])
        #raise
        for sent, label in zip(iterable, labels):
            
            outtext = ""
            lastlabel = "-"
            for token, tlabel in zip(sent, label):
                tlabel = tlabel.split("-")[-1]
                #print (tlabel)
                if tlabel != lastlabel and lastlabel in compressed_labels:#self.ner_labels.token2id:
                    outtext += "</%s>"%lastlabel
                if tlabel not in compressed_labels:#self.ner_labels.token2id:
                    outtext += token
                    lastlabel = tlabel
                else:
                    if tlabel != lastlabel:
                        outtext += "<%s>%s"%(tlabel, token)
                    else:
                        outtext += token
                    lastlabel = tlabel
            if lastlabel in compressed_labels:#self.ner_labels.token2id:
                outtext += "</%s>"%lastlabel
            out.append(outtext)
        return out

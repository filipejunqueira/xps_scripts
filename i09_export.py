# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:00:46 2021

@author: DA Duncan

Original version of fitNIXSW written by T-L Lee
"""
import os
import numpy as np
import h5py
import tkinter as tk
from tkinter import filedialog as fD
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.optimize import least_squares
#import lineshapes as ln
from copy import copy
import math
from scipy.signal import fftconvolve

        
class data_square: 
            
    def __init__(self):
        self.data = []
        self.hv = []
        self.energy = []
        self.angles = []
        self.nixswr = []
        self.i0Hard = []
        self.i0soft = []
        self.iDrain = []
        self.names  = []
        self.noRegions = 0
        #self.model_dir = 'D:/Dropbox/Data_b/neural_network_test/nn_post_PS/training_out'
        self.nixswDatadir = "/home/captainbroccoli/i09_data/si-"
        self.nixswQparamFile = "q_param.txt"
        self.model_dir = 'C:/Users/dadun/Dropbox/Data/neural_network_test/Ryan/NN_for_DAD/NN_for_DAD/nets'
        self.model_dirPos = 'C:/Users/dadun/Dropbox/Data/neural_network_test/Ryan/NN_for_DAD/NN_for_DAD/nets'
        #self.model_dirPos = 'C:/Users/dadun/Dropbox/Data/neural_network_test/work_computer/training_out'
        self.model_dirWidth = 'C:/Users/dadun/Dropbox/Data/neural_network_test/Ryan/NN_for_DAD/NN_for_DAD/nets'
        self.model_prefix = 'countnet_DAD'
        self.model_prefixPos = 'ClusterTrainPointPos_'
        self.model_prefixWidth = 'ClusterTrainWidth_'
        self.num_nets = 10
        self.num_netsPos = 10
        self.num_netsWidth = 10
        self.model_numbers = range(0,10)
        self.model_numbersPos = range(0,10) 
        self.model_numbersWidth = range(0,10)
        self.lowestAlt = 20
        self.nnDataPoints = 200
        self.cs = {}
        self.ub = {}
        self.lb = {}
        self.flags = {}
        self.pins = {}
        self.csAuger = {}
        self.ubAuger = {}
        self.lbAuger = {}
        self.flagsAuger = {}
        self.pinsAuger = {}        
        self.titles = ['bgr0','bgr1',  
                       'intensity1',   'energy1','grad1',  'width1', 'lwidth1',  'asym1', 'step_h1', 
                       'intensity2',  'energy2',           'width2', 'lwidth2',  'asym2', 'step_h2', 
                       'intensity3',  'energy3',           'width3', 'lwidth3',  'asym3', 'step_h3', 
                       'intensity4',   'energy4',          'width4', 'lwidth4',  'asym4', 'step_h4',   
                       'intensity5',  'energy5',           'width5', 'lwidth5',  'asym5', 'step_h5',   
                       'intensity6',  'energy6',           'width6', 'lwidth6',  'asym6', 'step_h6', ];
        self.titlesAuger =      ['intensity1a',   'energy1a', 'width1a', 'lwidth1a','step_h1a','asym1a',
                                 'intensity2a',   'energy2a', 'width2a', 'lwidth2a','step_h2a','asym2a',
                                 'intensity3a',   'energy3a', 'width3a', 'lwidth3a','step_h3a','asym3a',
                                 'intensity4a',   'energy4a', 'width4a', 'lwidth4a','step_h4a','asym4a',
                                 'intensity5a',   'energy5a', 'width5a', 'lwidth5a','step_h5a','asym5a',
                                 'intensity6a',   'energy6a', 'width6a', 'lwidth6a','step_h6a','asym6a']        
        self.z = ['',  'H','He',
                      'Li','Be',                                                         'B', 'C', 'N', 'O', 'F','Ne',
                      'Na','Mg',                                                        'Al','Si', 'P', 'S','Cl','Ar',
                       'K','Ca',    'Sc','Ti', 'V','Cr','Mn','Fe','Co','Ni','Cu','Zn',  'Ga','Ge','As','Se','Br','Kr',
                      'Rb','Sr',     'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',  'In','Sn','Sb','Te', 'I','Xe',
                      'Cs','Ba',   'La',
                                'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
                                         'Hf','Ta', 'W','Re','Os','Ir','Pt','Au','Hg',    'Tl','Pb','Bi','Po','At','Rn',
                      'Fr','Ra',    'Ac',
                                'Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No']

        #data_square.fittingParam(self)

    def fittingParam(self,Region,number_peaks,wholeRegion=True):
        bgr0            =   ['bgr0',  4500,    1e7,     0.0,     True,  0];   
        bgr1            =   ['bgr1',  0,      1000,   -1000,     True,  0]; 
        
           
        
        #
        energy1         =   ['energy1',127.76,  128,     127,    True,  0];
        grad1           =   ['grad1',0,     +0.05,   -0.05,      False,  0];
        intensity1      =   ['intensity1',6500,    1e6,     0,   True,  0];
        width1          =   ['width1',0.5,       1,     0.1,     True,  0];
        lwidth1         =   ['lwidth1', 0.5,     1.0,     0.1,    True,  0];
        asym1           =   ['asym1', 0.01,  0.2,    0.0001,     True,  0];
        step_h1         =   ['step_h1',0,     0.4,     0.0,     True,  0]; 
        #
        energy2         =   ['energy2',0.8675,    0.87, 0.85,     False,  0];
        intensity2      =   ['intensity2',6500,     1e6,   0, False,  0];
        width2          =   ['width2',0.5,        1.1,   0.1,     False,  0];
        lwidth2         =   ['lwidth2', 1.0,     1.0,     0.1,    False,  0];
        asym2           =   ['asym2', 1.0,  0.2,    0.0001,     False,  0];
        step_h2         =   ['step_h2',1.0,     0.4,     0.0,     False,  0]; 
        #
        energy3         =   ['energy3',-0.2817,  -0.27, -0.29,    False,  0];
        intensity3      =   ['intensity3',6500,       1e6,  0, False,  0];
        width3          =   ['width3',0.5,          1,  0.06,     False,  0];
        lwidth3         =   ['lwidth3', 1.0,     1.0,     0.1,    False,  0];
        asym3           =   ['asym3', 1.0,  0.2,    0.0001,     False,  0];
        step_h3         =   ['step_h3',1.0,     0.4,     0.0,     False,  0]; 
        #
        energy4         =   ['energy4',0.5847,    0.59,  0.57,    False,  0];
        intensity4      =   ['intensity4',0.5,        1e6,   0, False,  0];
        width4          =   ['width4',0.5,          1,  0.06,     False,  0];
        lwidth4         =   ['lwidth4', 1.0,     1.0,     0.1,    False,  0];
        asym4           =   ['asym4', 1.0,  0.2,   0.0001,     False,  0];
        step_h4         =   ['step_h4',1.0,     0.4,     0.0,     False,  0]; 
        #
        energy5         =   ['energy5',11.3691,    11.5, 11.1,    False,  0];
        intensity5      =   ['intensity5',700,         1e6,   0,  False,  0];
        width5          =   ['width5',0.5,           1, 0.06,     False,  0];
        lwidth5         =   ['lwidth5', 1.0,     1.0,     0.1,    False,  0];
        asym5           =   ['asym5', 1.0,  0.2,    0.0001,     False,  0];
        step_h5         =   ['step_h5',1.0,     0.4,     0.0,     False,  0]; 
        #
        energy6         =   ['energy6', 9.3464,      9.4,  9.3,   False,  0];
        intensity6      =   ['intensity6',500,         1e6,    0, False,  0];
        width6          =   ['width6',0.5,           1, 0.06,     False,  0];  
        lwidth6         =   ['lwidth6', 1.0,     1.0,     0.1,    False,  0];
        asym6           =   ['asym6', 1.0,  0.2,    0.0001,     False,  0];
        step_h6         =   ['step_h6',1.0,     0.4,     0.0,     False,  0]; 
        #
        cc=1;
        #
        holdCS= {bgr0[0]: bgr0[cc], bgr1[0]: bgr1[cc],  
               intensity1[0]: intensity1[cc], energy1[0]: energy1[cc], grad1[0]: grad1[cc], width1[0]: width1[cc], lwidth1[0]: lwidth1[cc], asym1[0]: asym1[cc], step_h1[0]: step_h1[cc], 
               intensity2[0]: intensity2[cc], energy2[0]: energy2[cc],                      width2[0]: width2[cc], lwidth2[0]: lwidth2[cc], asym2[0]: asym2[cc], step_h2[0]: step_h2[cc], 
               intensity3[0]: intensity3[cc], energy3[0]: energy3[cc],                      width3[0]: width3[cc], lwidth3[0]: lwidth3[cc], asym3[0]: asym3[cc], step_h3[0]: step_h3[cc], 
               intensity4[0]: intensity4[cc], energy4[0]: energy4[cc],                      width4[0]: width4[cc], lwidth4[0]: lwidth4[cc], asym4[0]: asym4[cc], step_h4[0]: step_h4[cc], 
               intensity5[0]: intensity5[cc], energy5[0]: energy5[cc],                      width5[0]: width5[cc], lwidth5[0]: lwidth5[cc], asym5[0]: asym5[cc], step_h5[0]: step_h5[cc], 
               intensity6[0]: intensity6[cc], energy6[0]: energy6[cc],                      width6[0]: width6[cc], lwidth6[0]: lwidth6[cc], asym6[0]: asym6[cc], step_h6[0]: step_h6[cc]}
        cc=2;
        holdUB = {bgr0[0]: bgr0[cc], bgr1[0]: bgr1[cc],  
               intensity1[0]: intensity1[cc], energy1[0]: energy1[cc], grad1[0]: grad1[cc], width1[0]: width1[cc], lwidth1[0]: lwidth1[cc], asym1[0]: asym1[cc], step_h1[0]: step_h1[cc], 
               intensity2[0]: intensity2[cc], energy2[0]: energy2[cc],                      width2[0]: width2[cc], lwidth2[0]: lwidth2[cc], asym2[0]: asym2[cc], step_h2[0]: step_h2[cc], 
               intensity3[0]: intensity3[cc], energy3[0]: energy3[cc],                      width3[0]: width3[cc], lwidth3[0]: lwidth3[cc], asym3[0]: asym3[cc], step_h3[0]: step_h3[cc], 
               intensity4[0]: intensity4[cc], energy4[0]: energy4[cc],                      width4[0]: width4[cc], lwidth4[0]: lwidth4[cc], asym4[0]: asym4[cc], step_h4[0]: step_h4[cc], 
               intensity5[0]: intensity5[cc], energy5[0]: energy5[cc],                      width5[0]: width5[cc], lwidth5[0]: lwidth5[cc], asym5[0]: asym5[cc], step_h5[0]: step_h5[cc], 
               intensity6[0]: intensity6[cc], energy6[0]: energy6[cc],                      width6[0]: width6[cc], lwidth6[0]: lwidth6[cc], asym6[0]: asym6[cc], step_h6[0]: step_h6[cc]}
        cc=3;
        holdLB = {bgr0[0]: bgr0[cc], bgr1[0]: bgr1[cc],  
               intensity1[0]: intensity1[cc], energy1[0]: energy1[cc], grad1[0]: grad1[cc], width1[0]: width1[cc], lwidth1[0]: lwidth1[cc], asym1[0]: asym1[cc], step_h1[0]: step_h1[cc], 
               intensity2[0]: intensity2[cc], energy2[0]: energy2[cc],                      width2[0]: width2[cc], lwidth2[0]: lwidth2[cc], asym2[0]: asym2[cc], step_h2[0]: step_h2[cc], 
               intensity3[0]: intensity3[cc], energy3[0]: energy3[cc],                      width3[0]: width3[cc], lwidth3[0]: lwidth3[cc], asym3[0]: asym3[cc], step_h3[0]: step_h3[cc], 
               intensity4[0]: intensity4[cc], energy4[0]: energy4[cc],                      width4[0]: width4[cc], lwidth4[0]: lwidth4[cc], asym4[0]: asym4[cc], step_h4[0]: step_h4[cc], 
               intensity5[0]: intensity5[cc], energy5[0]: energy5[cc],                      width5[0]: width5[cc], lwidth5[0]: lwidth5[cc], asym5[0]: asym5[cc], step_h5[0]: step_h5[cc], 
               intensity6[0]: intensity6[cc], energy6[0]: energy6[cc],                      width6[0]: width6[cc], lwidth6[0]: lwidth6[cc], asym6[0]: asym6[cc], step_h6[0]: step_h6[cc]}
        cc=4;
        holdFlags = {bgr0[0]: bgr0[cc], bgr1[0]: bgr1[cc],  
               intensity1[0]: intensity1[cc], energy1[0]: energy1[cc], grad1[0]: grad1[cc], width1[0]: width1[cc], lwidth1[0]: lwidth1[cc], asym1[0]: asym1[cc], step_h1[0]: step_h1[cc], 
               intensity2[0]: intensity2[cc], energy2[0]: energy2[cc],                      width2[0]: width2[cc], lwidth2[0]: lwidth2[cc], asym2[0]: asym2[cc], step_h2[0]: step_h2[cc], 
               intensity3[0]: intensity3[cc], energy3[0]: energy3[cc],                      width3[0]: width3[cc], lwidth3[0]: lwidth3[cc], asym3[0]: asym3[cc], step_h3[0]: step_h3[cc], 
               intensity4[0]: intensity4[cc], energy4[0]: energy4[cc],                      width4[0]: width4[cc], lwidth4[0]: lwidth4[cc], asym4[0]: asym4[cc], step_h4[0]: step_h4[cc], 
               intensity5[0]: intensity5[cc], energy5[0]: energy5[cc],                      width5[0]: width5[cc], lwidth5[0]: lwidth5[cc], asym5[0]: asym5[cc], step_h5[0]: step_h5[cc], 
               intensity6[0]: intensity6[cc], energy6[0]: energy6[cc],                      width6[0]: width6[cc], lwidth6[0]: lwidth6[cc], asym6[0]: asym6[cc], step_h6[0]: step_h6[cc]}
        cc=5;
        holdPins = {bgr0[0]: bgr0[cc], bgr1[0]: bgr1[cc],  
               intensity1[0]: intensity1[cc], energy1[0]: energy1[cc], grad1[0]: grad1[cc], width1[0]: width1[cc], lwidth1[0]: lwidth1[cc], asym1[0]: asym1[cc], step_h1[0]: step_h1[cc], 
               intensity2[0]: intensity2[cc], energy2[0]: energy2[cc],                      width2[0]: width2[cc], lwidth2[0]: lwidth2[cc], asym2[0]: asym2[cc], step_h2[0]: step_h2[cc], 
               intensity3[0]: intensity3[cc], energy3[0]: energy3[cc],                      width3[0]: width3[cc], lwidth3[0]: lwidth3[cc], asym3[0]: asym3[cc], step_h3[0]: step_h3[cc], 
               intensity4[0]: intensity4[cc], energy4[0]: energy4[cc],                      width4[0]: width4[cc], lwidth4[0]: lwidth4[cc], asym4[0]: asym4[cc], step_h4[0]: step_h4[cc], 
               intensity5[0]: intensity5[cc], energy5[0]: energy5[cc],                      width5[0]: width5[cc], lwidth5[0]: lwidth5[cc], asym5[0]: asym5[cc], step_h5[0]: step_h5[cc], 
               intensity6[0]: intensity6[cc], energy6[0]: energy6[cc],                      width6[0]: width6[cc], lwidth6[0]: lwidth6[cc], asym6[0]: asym6[cc], step_h6[0]: step_h6[cc]}
        #
        # Auger parameters
        energy1a         =    ['energy1a',100,  105,     95,         False,  0];
        intensity1a      = ['intensity1a',6500,    1e6,       0,     False,  0];
        lwidth1a         =    ['lwidth1a', 0.5,     1.0,     0.1,    False,  0]; 
        width1a          =     ['width1a',0.5,       1,     0.1,     False,  0];
        step_h1a         =    ['step_h1a',0.0,     0.4,     0.0,     False,  0]; 
        asym1a          =       ['asym1a', 0.1,  0.2,    0.05,       False,  0];
        #
        energy2a         =    ['energy2a',100,  105,     95,         False,  0];
        intensity2a      = ['intensity2a',6500,    1e6,       0,     False,  0];
        lwidth2a         =    ['lwidth2a', 0.5,     1.0,     0.1,    False,  0]; 
        width2a          =     ['width2a',0.5,       1,     0.1,     False,  0];
        step_h2a         =    ['step_h2a',0.0,     0.4,     0.0,     False,  0]; 
        asym2a          =       ['asym2a', 0.1,  0.2,    0.05,       False,  0];
        #
        energy3a         =    ['energy3a',100,  105,     95,         False,  0];
        intensity3a      = ['intensity3a',6500,    1e6,       0,     False,  0];
        lwidth3a         =    ['lwidth3a', 0.5,     1.0,     0.1,    False,  0]; 
        width3a          =     ['width3a',0.5,       1,     0.1,     False,  0];
        step_h3a         =    ['step_h3a',0.0,     0.4,     0.0,     False,  0]; 
        asym3a          =       ['asym3a', 0.1,  0.2,    0.05,       False,  0];
        #
        energy4a         =    ['energy4a',100,  105,     95,         False,  0];
        intensity4a      = ['intensity4a',6500,    1e6,       0,     False,  0];
        lwidth4a         =    ['lwidth4a', 0.5,     1.0,     0.1,    False,  0]; 
        width4a          =     ['width4a',0.5,       1,     0.1,     False,  0];
        step_h4a         =    ['step_h4a',0.0,     0.4,     0.0,     False,  0]; 
        asym4a          =       ['asym4a', 0.1,  0.2,    0.05,       False,  0];  
        #
        energy5a         =    ['energy5a',100,  105,     95,         False,  0];
        intensity5a      = ['intensity5a',6500,    1e6,       0,     False,  0];
        lwidth5a         =    ['lwidth5a', 0.5,     1.0,     0.1,    False,  0]; 
        width5a          =     ['width5a',0.5,       1,     0.1,     False,  0];
        step_h5a         =    ['step_h5a',0.0,     0.4,     0.0,     False,  0]; 
        asym5a          =       ['asym5a', 0.1,  0.2,    0.05,       False,  0];
        #
        energy6a         =    ['energy6a',100,  105,     95,         False,  0];
        intensity6a      = ['intensity6a',6500,    1e6,       0,     False,  0];
        lwidth6a         =    ['lwidth6a', 0.5,     1.0,     0.1,    False,  0]; 
        width6a          =     ['width6a',0.5,       1,     0.1,     False,  0];
        step_h6a         =    ['step_h6a',0.0,     0.4,     0.0,     False,  0]; 
        asym6a          =       ['asym6a', 0.1,  0.2,    0.05,       False,  0];   
        #
        cc=1;
        #                   1       2        3         4              5    %         6              7           8           9             10         11          12             13          14        15              16         17          18            19         20            21           22           23     24       
        holdCSAuger = {intensity1a[0]: intensity1a[cc], energy1a[0]: energy1a[cc], width1a[0]: width1a[cc], lwidth1a[0]: lwidth1a[cc], step_h1a[0]: step_h1a[cc],asym1a[0]: asym1a[cc], 
                        intensity2a[0]: intensity2a[cc], energy2a[0]: energy2a[cc], width2a[0]: width2a[cc], lwidth2a[0]: lwidth2a[cc], step_h2a[0]: step_h2a[cc],asym2a[0]: asym2a[cc], 
                        intensity3a[0]: intensity3a[cc], energy3a[0]: energy3a[cc], width3a[0]: width3a[cc], lwidth3a[0]: lwidth3a[cc], step_h3a[0]: step_h3a[cc],asym3a[0]: asym3a[cc], 
                        intensity4a[0]: intensity4a[cc], energy4a[0]: energy4a[cc], width4a[0]: width4a[cc], lwidth4a[0]: lwidth4a[cc], step_h4a[0]: step_h4a[cc],asym4a[0]: asym4a[cc], 
                        intensity5a[0]: intensity5a[cc], energy5a[0]: energy5a[cc], width5a[0]: width5a[cc], lwidth5a[0]: lwidth5a[cc], step_h5a[0]: step_h5a[cc],asym5a[0]: asym5a[cc], 
                        intensity6a[0]: intensity6a[cc], energy6a[0]: energy6a[cc], width6a[0]: width6a[cc], lwidth6a[0]: lwidth6a[cc], step_h6a[0]: step_h6a[cc],asym6a[0]: asym6a[cc] }
        cc=2;
        holdUBAuger = {intensity1a[0]: intensity1a[cc], energy1a[0]: energy1a[cc], width1a[0]: width1a[cc], lwidth1a[0]: lwidth1a[cc], step_h1a[0]: step_h1a[cc],asym1a[0]: asym1a[cc], 
               intensity2a[0]: intensity2a[cc], energy2a[0]: energy2a[cc], width2a[0]: width2a[cc], lwidth2a[0]: lwidth2a[cc], step_h2a[0]: step_h2a[cc],asym2a[0]: asym2a[cc], 
                   intensity3a[0]: intensity3a[cc], energy3a[0]: energy3a[cc], width3a[0]: width3a[cc], lwidth3a[0]: lwidth3a[cc], step_h3a[0]: step_h3a[cc],asym3a[0]: asym3a[cc], 
                   intensity4a[0]: intensity4a[cc], energy4a[0]: energy4a[cc], width4a[0]: width4a[cc], lwidth4a[0]: lwidth4a[cc], step_h4a[0]: step_h4a[cc],asym4a[0]: asym4a[cc], 
                   intensity5a[0]: intensity5a[cc], energy5a[0]: energy5a[cc], width5a[0]: width5a[cc], lwidth5a[0]: lwidth5a[cc], step_h5a[0]: step_h5a[cc],asym5a[0]: asym5a[cc], 
                   intensity6a[0]: intensity6a[cc], energy6a[0]: energy6a[cc], width6a[0]: width6a[cc], lwidth6a[0]: lwidth6a[cc], step_h6a[0]: step_h6a[cc],asym6a[0]: asym6a[cc] }
        cc=3;
        holdLBAuger = {intensity1a[0]: intensity1a[cc], energy1a[0]: energy1a[cc], width1a[0]: width1a[cc], lwidth1a[0]: lwidth1a[cc], step_h1a[0]: step_h1a[cc],asym1a[0]: asym1a[cc], 
                        intensity2a[0]: intensity2a[cc], energy2a[0]: energy2a[cc], width2a[0]: width2a[cc], lwidth2a[0]: lwidth2a[cc], step_h2a[0]: step_h2a[cc],asym2a[0]: asym2a[cc], 
                        intensity3a[0]: intensity3a[cc], energy3a[0]: energy3a[cc], width3a[0]: width3a[cc], lwidth3a[0]: lwidth3a[cc], step_h3a[0]: step_h3a[cc],asym3a[0]: asym3a[cc], 
                        intensity4a[0]: intensity4a[cc], energy4a[0]: energy4a[cc], width4a[0]: width4a[cc], lwidth4a[0]: lwidth4a[cc], step_h4a[0]: step_h4a[cc],asym4a[0]: asym4a[cc], 
                        intensity5a[0]: intensity5a[cc], energy5a[0]: energy5a[cc], width5a[0]: width5a[cc], lwidth5a[0]: lwidth5a[cc], step_h5a[0]: step_h5a[cc],asym5a[0]: asym5a[cc], 
                        intensity6a[0]: intensity6a[cc], energy6a[0]: energy6a[cc], width6a[0]: width6a[cc], lwidth6a[0]: lwidth6a[cc], step_h6a[0]: step_h6a[cc],asym6a[0]: asym6a[cc] }  
        cc=4;
        holdFlagsAuger = {intensity1a[0]: intensity1a[cc], energy1a[0]: energy1a[cc], width1a[0]: width1a[cc], lwidth1a[0]: lwidth1a[cc], step_h1a[0]: step_h1a[cc],asym1a[0]: asym1a[cc], 
                           intensity2a[0]: intensity2a[cc], energy2a[0]: energy2a[cc], width2a[0]: width2a[cc], lwidth2a[0]: lwidth2a[cc], step_h2a[0]: step_h2a[cc],asym2a[0]: asym2a[cc], 
                           intensity3a[0]: intensity3a[cc], energy3a[0]: energy3a[cc], width3a[0]: width3a[cc], lwidth3a[0]: lwidth3a[cc], step_h3a[0]: step_h3a[cc],asym3a[0]: asym3a[cc], 
                           intensity4a[0]: intensity4a[cc], energy4a[0]: energy4a[cc], width4a[0]: width4a[cc], lwidth4a[0]: lwidth4a[cc], step_h4a[0]: step_h4a[cc],asym4a[0]: asym4a[cc], 
                           intensity5a[0]: intensity5a[cc], energy5a[0]: energy5a[cc], width5a[0]: width5a[cc], lwidth5a[0]: lwidth5a[cc], step_h5a[0]: step_h5a[cc],asym5a[0]: asym5a[cc], 
                           intensity6a[0]: intensity6a[cc], energy6a[0]: energy6a[cc], width6a[0]: width6a[cc], lwidth6a[0]: lwidth6a[cc], step_h6a[0]: step_h6a[cc],asym6a[0]: asym6a[cc] }
        cc=5;
        holdPinsAuger = {intensity1a[0]: intensity1a[cc], energy1a[0]: energy1a[cc], width1a[0]: width1a[cc], lwidth1a[0]: lwidth1a[cc], step_h1a[0]: step_h1a[cc],asym1a[0]: asym1a[cc], 
                          intensity2a[0]: intensity2a[cc], energy2a[0]: energy2a[cc], width2a[0]: width2a[cc], lwidth2a[0]: lwidth2a[cc], step_h2a[0]: step_h2a[cc],asym2a[0]: asym2a[cc], 
                          intensity3a[0]: intensity3a[cc], energy3a[0]: energy3a[cc], width3a[0]: width3a[cc], lwidth3a[0]: lwidth3a[cc], step_h3a[0]: step_h3a[cc],asym3a[0]: asym3a[cc], 
                          intensity4a[0]: intensity4a[cc], energy4a[0]: energy4a[cc], width4a[0]: width4a[cc], lwidth4a[0]: lwidth4a[cc], step_h4a[0]: step_h4a[cc],asym4a[0]: asym4a[cc], 
                          intensity5a[0]: intensity5a[cc], energy5a[0]: energy5a[cc], width5a[0]: width5a[cc], lwidth5a[0]: lwidth5a[cc], step_h5a[0]: step_h5a[cc],asym5a[0]: asym5a[cc], 
                          intensity6a[0]: intensity6a[cc], energy6a[0]: energy6a[cc], width6a[0]: width6a[cc], lwidth6a[0]: lwidth6a[cc], step_h6a[0]: step_h6a[cc],asym6a[0]: asym6a[cc] }  
        if wholeRegion:
            self.cs.update({Region:{number_peaks:{i:copy(holdCS) for i in range(len(self.hv[Region]))}}})
            self.lb.update({Region:{number_peaks:holdLB}})
            self.ub.update({Region:{number_peaks:holdUB}})
            self.flags.update({Region:{number_peaks:holdFlags}})
            self.pins.update({Region:{number_peaks:holdPins}})  
            #
            self.csAuger.update({Region:{number_peaks:{i:copy(holdCSAuger) for i in range(len(self.hv[Region]))}}})
            self.lbAuger.update({Region:{number_peaks:holdLBAuger}})
            self.ubAuger.update({Region:{number_peaks:holdUBAuger}})
            self.flagsAuger.update({Region:{number_peaks:holdFlagsAuger}})
            self.pinsAuger.update({Region:{number_peaks:holdPinsAuger}})            
        else:
            self.cs[Region].update({number_peaks:{i:copy(holdCS) for i in range(len(self.hv[Region]))}})
            self.lb[Region].update({number_peaks:holdLB})
            self.ub[Region].update({number_peaks:holdUB})
            self.flags[Region].update({number_peaks:holdFlags})
            self.pins[Region].update({number_peaks:holdPins})  
            #
            self.csAuger[Region].update({number_peaks:{i:copy(holdCSAuger) for i in range(len(self.hv[Region]))}})
            self.lbAuger[Region].update({number_peaks:holdLBAuger})
            self.ubAuger[Region].update({number_peaks:holdUBAuger})
            self.flagsAuger[Region].update({number_peaks:holdFlagsAuger})
            self.pinsAuger[Region].update({number_peaks:holdPinsAuger})   

        for nn in range(1,number_peaks):
            self.setPin('asym'+str(nn+1),'asym1',Region=Region,number_peaks=number_peaks)     
            self.setPin('step_h'+str(nn+1),'step_h1',Region=Region,number_peaks=number_peaks) 
            self.setPin('lwidth'+str(nn+1),'lwidth1',Region=Region,number_peaks=number_peaks) 
    
    def setParam(self,param,value,Region=0,number_peaks=1,hvIndex=0):
        self.cs[Region][number_peaks][hvIndex][param] = copy(value)
    def setParamAuger(self,param,value,Region=0,number_peaks=1,hvIndex=0):
        self.csAuger[Region][number_peaks][hvIndex][param] = copy(value)  
    def setLB(self,param,value,Region=0,number_peaks=1):
        self.lb[Region][number_peaks][param] = copy(value)
    def setLBAuger(self,param,value,Region=0,number_peaks=1):
        self.lbAuger[Region][number_peaks][param] = copy(value)  
    def setUB(self,param,value,Region=0,number_peaks=1):
        self.ub[Region][number_peaks][param] = copy(value)
    def setUBAuger(self,param,value,Region=0,number_peaks=1):
        self.ubAuger[Region][number_peaks][param] = copy(value)   
    def setFit(self,param,value,Region=0,number_peaks=1):
        self.flags[Region][number_peaks][param] = copy(value)
    def setFitAuger(self,param,value,Region=0,number_peaks=1):
        self.flagsAuger[Region][number_peaks][param] = copy(value) 
    def setPin(self,param,param2,Region=0,number_peaks=1):
        if param2 == 0:
            self.pins[Region][number_peaks][param] = 0
        else:
            self.pins[Region][number_peaks][param] = self.titles.index(param2)        
    def setPinAuger(self,param,value,Region=0,number_peaks=1):
        self.pinsAuger[Region][number_peaks][param] = copy(value)
        
        
    def fitNIXSW(self,xyz,latticeParam,reflection,dataPrefix,number_peaks = 1,samp = '',width = 0.1,maxWidth = 0.4,fh0=0.9,ph0=0.5,
                 energy=None,hvOffset=0,Region=0,peakNo=0,Z=6,n=1,l=0,js=0,thetaDipolar=18,
                 thetaOffsetNI=4,thetaReflectionToSurface=0,DebyeWalleFactorSubstrate=0,DebyeWalleFactorMono=0,step = 100,
                 areagaus=1,ngaus=1010,plotter=True):
        thetaB = 90-thetaOffsetNI
        alphaB = thetaReflectionToSurface
        hs = np.array(reflection)
        rExp = np.atleast_2d(np.array(self.nixswr[Region]))
        hvExp = np.atleast_2d(np.array(self.hv[Region]))
        iExp = np.atleast_2d(np.array([self.cs[Region][number_peaks][i]["intensity" + str(peakNo+1)] for i in range(hvExp.shape[1])]))
        DWBs = DebyeWalleFactorSubstrate
        DWBm = DebyeWalleFactorMono
        if energy is None:
            indexMaxNIXSWR = np.where(rExp==np.max(rExp))[0][0]
            energy = hvExp[indexMaxNIXSWR]
        if not isinstance(energy,np.ndarray):
            energy = np.array(energy)
        #lsp0 should be a list of six items that define the unit cell of the substrate: a, b, c, alpha, beta, gamma
        lps = [latticeParam[0],latticeParam[1],latticeParam[2],latticeParam[3]/180*math.pi,latticeParam[4]/180*math.pi,latticeParam[5]/180*math.pi] 
        #xyzs should be an array of 4xN items, where each row is an atom in the unit cell, identified by its atomic number Z, its fractional coordinate (x',y',z') and its relative occupancy (o):
        #  xyzs = [Z1, x1', y1', z1', o1;
        #          Z2, x2', y2', z2', o2;
        #          ...
        #          ZN, xN', yN', zN', oN]
        if isinstance(xyz,np.ndarray):
            xyzs = xyz
        else:
            xyzs = np.array(xyz)
        fwhmgaus = width
        [betab,gammab,deltab,Eb,Erow] = self.q_param(Z,n,l,js,plotter)
        the = thetaDipolar*math.pi/180.
        phie = 0*math.pi/180.
        test_E = [i for i in range(1500,5000,10)]
        tck = interpolate.splrep(Erow,betab,s=0)
        beta = interpolate.splev(test_E,tck,der=0)    
        tck = interpolate.splrep(Erow,gammab,s=0)
        gamma = interpolate.splev(test_E,tck,der=0)  
        tck = interpolate.splrep(Erow,deltab,s=0)
        delta = interpolate.splev(test_E,tck,der=0)
        #
        idxE = (np.abs(test_E-energy+Eb)).argmin()
        betNow = beta[idxE]
        gamNow = gamma[idxE]
        delNow = delta[idxE]
        #
        Q =  (  ( delNow*np.sin(the)*np.cos(phie) )+
                ( gamNow*np.cos(phie)*np.sin(the)*np.cos(the)**2 )  )  / (  1+betNow*0.5*( 3*np.cos(the)**2-1 )  )       
        C1 = (1+Q)/(1-Q)
        C2 = 1/(1-Q)
        C3 = 0
        
        
        
        lambdaNow = 12398.54/energy
        
        datar = np.squeeze(np.array([hvExp,rExp])).transpose()
        datay = np.squeeze(np.array([hvExp,iExp])).transpose()

        nas = xyzs.shape[0]
        
        ################################### Si monochromator ##################
        hm = np.array([1,1,1])
        lpm = [5.431,5.431,5.431,90/180*math.pi,90/180*math.pi,90/180*math.pi]
        offsetX = 0.125
        offsetY = offsetX
        offsetZ = offsetX #Shift the origin to inversion center 
        xyzm = np.array([[14,0.00-offsetX,0.00-offsetY,0.00-offsetZ,1],
                [14,0.50-offsetX,0.50-offsetY,0.00-offsetZ,1],
                [14,0.50-offsetX,0.00-offsetY,0.50-offsetZ,1],
                [14,0.00-offsetX,0.50-offsetY,0.50-offsetZ,1],
                [14,0.25-offsetX,0.25-offsetY,0.25-offsetZ,1],
                [14,0.75-offsetX,0.75-offsetY,0.25-offsetZ,1],
                [14,0.75-offsetX,0.25-offsetY,0.75-offsetZ,1],
                [14,0.25-offsetX,0.75-offsetY,0.75-offsetZ,1]]) #Atomic number,x,y,z,occupancy
        
        nam = xyzm.shape[0]
        
        ucvs = lps[0]*lps[1]*lps[2]*(1-np.cos(lps[3])**2 - 
                 np.cos(lps[4])**2 -
                 np.cos(lps[5])**2 +
                 2*np.cos(lps[3])*np.cos(lps[4])*np.cos(lps[5]))**0.5; #Unit cell volume in A^3, sample.
        ucvm = lpm[0]*lpm[1]*lpm[2]; #Unit cell volume in A^3, mono Si.
        lvs=[[lps[0]                ,0                                                                       ,0],
             [lps[1]*np.cos(lps[5]) ,lps[1]*np.sin(lps[5])                                                ,0],
             [lps[2]*np.cos(lps[4]) ,lps[2]*(np.cos(lps[3])-np.cos(lps[4])*np.cos(lps[5]))/np.sin(lps[5]) ,lps[2]*( 1-np.cos(lps[3])**2-np.cos(lps[4])**2-np.cos(lps[5])**2+2*np.cos(lps[3])*np.cos(lps[4])*np.cos(lps[5]) )**0.5/np.sin(lps[5])]] #Real space lattice vectors a, b, and c in Cartesian coordinates with a parallel to X and b in the XY plane
        rlvs=[[lvs[1][1]*lvs[2][2]-lvs[1][2]*lvs[2][1],   lvs[1][2]*lvs[2][0]-lvs[1][0]*lvs[2][2],    lvs[1][0]*lvs[2][1]-lvs[1][1]*lvs[2][0]], 
              [lvs[2][1]*lvs[0][2]-lvs[2][2]*lvs[0][1],   lvs[2][2]*lvs[0][0]-lvs[2][0]*lvs[0][2],    lvs[2][0]*lvs[0][1]-lvs[2][1]*lvs[0][0]],
              [0,                                         0,                                          lps[0]*lps[1]*np.sin(lps[5])]]
        rlvs=np.array([[j/ucvs for j in i] for i in rlvs])
    
        mm = np.matmul(hs,rlvs)  ###### Line 148 dhs = 
        dhs = 1/(np.sum(mm**2)**0.5)#Bragg plane spacing for hkl reflection in A, sample.
        dhm = lpm[0]/np.sqrt(np.matmul(hm,hm.transpose())) #Bragg plane spacing for hkl reflection in A, mono Si.
        thbs=np.arcsin(dhs**(-1)*lambdaNow/2) #Bragg angle in rad, sample.
        thbm=np.arcsin(dhm**(-1)*lambdaNow/2) #Bragg angle in rad, mono Si
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Structure factors and chi values, sample
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        f0 = np.loadtxt(self.nixswDatadir +'/f0_all_free_atoms.txt')
        fps=np.zeros((nas,1))
        fpps=np.zeros((nas,1))
        fs=np.zeros((nas,1),dtype="complex_")
        f0s=np.zeros((nas,1),dtype="complex_")
        #print(xyzs)
        for i in range(nas):
            fpfppdata=np.loadtxt(self.nixswDatadir + '/' + self.z[int(xyzs[i][0])] + '.nff',skiprows=1)
            #plt.plot(fpfppdata[:,0],fpfppdata[:,1])
            tck     = interpolate.splrep(fpfppdata[:,0],fpfppdata[:,1],s=0)
            fps[i]  = interpolate.splev(energy,tck,der=0)-xyzs[i][0]
            tck     = interpolate.splrep(fpfppdata[:,0],fpfppdata[:,2],s=0)
            fpps[i] = interpolate.splev(energy,tck,der=0)
            tck     = interpolate.splrep(f0[:,0],f0[:,int(xyzs[i][0]-2)],s=0)
            fs[i]   = interpolate.splev(0.5*dhs**(-1),tck,der=0)+fps[i]+1j*fpps[i]
            f0s[i]  = xyzs[i][0]+fps[i]+1j*fpps[i];
        
        #print(xyzs[:,1:3])
        #print(hs.transpose())
        hrs=np.matmul(xyzs[:,1:4],hs.transpose())
        DWBsprefactor = np.exp( -DWBs/dhs**2/4 )
        fsOccup = np.atleast_2d(np.array([fs[i,0]*xyz[i,4] for i in range(nas)])).transpose()
        f0sOccup = np.atleast_2d(np.array([f0s[i,0]*xyz[i,4] for i in range(nas)])).transpose() 
        
        Fhs  = np.matmul( np.exp( 2*np.pi*1j*hrs.transpose()) , fsOccup ) * DWBsprefactor
        Fhbs = np.matmul( np.exp( -2*np.pi*1j*hrs.transpose()) , fsOccup ) * DWBsprefactor
        F0s  = np.sum( f0sOccup ) * DWBsprefactor;
        
        gams=2.818e-5*lambdaNow**2/np.pi/ucvs
        chihs=-gams*Fhs
        chihbs=-gams*Fhbs
        chi0s=-gams*F0s        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Structure factors and chi values, mono
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        fpm=np.zeros((nam,1))
        fppm=np.zeros((nam,1))
        fm=np.zeros((nam,1),dtype="complex_")
        f0m=np.zeros((nam,1),dtype="complex_")
        for i in range(nam):
            fpfppdata  = np.loadtxt(self.nixswDatadir + '/' + self.z[int(xyzm[i][0])] + '.nff',skiprows=1)
            tck        = interpolate.splrep(fpfppdata[:,0],fpfppdata[:,1],s=0)
            fpm[i]     = interpolate.splev(energy,tck,der=0)-xyzm[i][0]
            tck        = interpolate.splrep(fpfppdata[:,0],fpfppdata[:,2],s=0)
            fppm[i]    = interpolate.splev(energy,tck,der=0)
            tck        = interpolate.splrep(f0[:,0],f0[:,int(xyzm[i][0])-2],s=0)
            fm[i]      = interpolate.splev(0.5*dhm**(-1),tck,der=0)+fpm[i]+1j*fppm[i]
            f0m[i]     = xyzm[i][0]+fpm[i]+1j*fppm[i];
        
        
        hrm=np.matmul(xyzm[:,1:4],hm.transpose())
        
        DWBmprefactor = np.exp( -DWBm/dhm**2/4 )
        fmOccup  = np.atleast_2d(np.array([fm[i,0]*xyzm[i,4] for i in range(nam)])).transpose()
        f0mOccup = np.atleast_2d(np.array([f0m[i,0]*xyzm[i,4] for i in range(nam)])).transpose() 
        
       
        
        Fhm  = np.matmul( np.exp( 2*np.pi*1j*hrm.transpose() ) , fmOccup ) * DWBmprefactor
        Fhbm = np.matmul( np.exp( -2*np.pi*1j*hrm.transpose()) , fmOccup ) * DWBmprefactor
        F0m  = np.sum( f0mOccup ) * DWBmprefactor;
        
        gamm=2.818e-5*lambdaNow**2/np.pi/ucvm
        
        chihm=-gamm*Fhm
        chihbm=-gamm*Fhbm
        chi0m=-gamm*F0m   
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Gaussian (normalized to integrated area)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        


        rangegaus = (np.max(hvExp)-np.min(hvExp))*2
        degaus = rangegaus/(ngaus-1)
        egaus= np.array([-round((ngaus-1)/2)*degaus+i*degaus for i in range(ngaus)])
        
        #%%%%%%%%%%%%%%%%%%%%%
        #%Sample rocking curve
        #%%%%%%%%%%%%%%%%%%%%%
        #%df is the angle between photon incidence and the surface towards the
        #%detector
        #%di is the angle between photon incidence and the surface away from the
        #%detector
        bs=-np.sin((thetaB-alphaB)/180*np.pi)/np.sin((thetaB+alphaB)/180*np.pi)
        Ps=1.0
        
        
        
        ewidths=(energy*np.divide(  np.abs( np.real(chihs)*Ps ), np.sin(thbs)**2  ) / abs(bs)**0.5)[0]
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% These parameters can be played with
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        des1=-10*ewidths
        des2=10*ewidths       


        nsteps=int(np.round((des2-des1)/degaus))
        a1= np.array([i for i in range(nsteps)])
        des=des1+(a1)*degaus

        etas=(2*bs*des*np.sin(thbs)**2/energy-chi0s*(1-bs)/2)/Ps/np.sqrt(np.abs(bs)*chihs*chihbs)
        
        xs = np.array([j-np.sqrt(j**2-1) if np.real(j) >= 0 else j+np.sqrt(j**2-1) for j in etas],dtype="complex_")*np.sqrt(np.abs(bs)*chihs*chihbs)/chihbs
        rs=np.abs(xs)**2
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Convolution 1 (sample rocking curve  gaussian)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        #%%%%%%%%%%%%%%%%%%%%
        #% Mono rocking curve
        #%%%%%%%%%%%%%%%%%%%%

        bm=-1
        Pm=1.0
        ewidthm=energy*np.abs(np.real(chihm)*Pm)/np.sin(thbm)**2/np.sqrt(np.abs(bm))
        dem1=-4*ewidthm
        dem2=10*ewidthm
        nstepm=int(np.round((dem2-dem1)/degaus))
        
        a1= np.array([i for i in range(nstepm)])
        
        dem=dem2-(a1)*degaus # Inverted for convolution
        
        etam=(2*bm*dem*np.sin(thbm)**2/energy-chi0m*(1-bm)/2)/Pm/np.sqrt(np.abs(bm)*chihm*chihbm)
        xm=np.array([j-np.sqrt(j**2-1) if np.real(j) >= 0 else j+np.sqrt(j**2-1) for j in etam])*np.sqrt(np.abs(bm)*chihm*chihbm)/chihbm

        rm=np.abs(xm)**2

        rmn=(rm*rm)/np.sum(rm*rm)
        
        cfwhmmIdx = [i for i in range(rm.shape[0]) if rm[i]>max(rm)/2]

        cfwhmm = 0.5*(dem[max(cfwhmmIdx)]+dem[min(cfwhmmIdx)])
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% Convolution 2 (mono rocking curve * (sample rocking curve * gaussian))
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rsgm=fftconvolve(rmn,rs,'full')
        
        #%%%%%%%%%%%%%%%%%%%
        #% Fit rocking curve
        #%%%%%%%%%%%%%%%%%%%

        X0 = datar[:,0]
        Y0  = datar[:,1]

        #% Estimate initial fitting parameters for rocking curve

        #eoffset=X0[Y.index(np.max(Y))]-0.3

        eoffset=X0[np.where(Y0==np.amax(Y0))]-0.3
        rscale=(np.max(datar[:,1])-0.5*(datar[-1,1]+datar[0,1]))/0.9*3
        rbgoffset=0.5*(datar[-1,1]+datar[0,1])/rscale
        #rbgslope=(datar[-1,1]-datar[0,1])/(datar[-1,0]-datar[0,0])/rscale
        #steps = 100
        multiples_p = [10, 1/1e3, 1/1e4, 1e3, 1e2] #rescales inputs so they are on a similar order of magnitude
        #p0 = [width*multiples_p[0],eoffset[0]*multiples_p[1],rscale*multiples_p[2],0*multiples_p[3],rbgoffset*multiples_p[4]]
        p0 = [0.25038*multiples_p[0],2630.1859*multiples_p[1],30771.0539*multiples_p[2],-0.001*multiples_p[3],0*multiples_p[4]]
        p0 = np.array(p0)
        
        lb = [0*multiples_p[0], (eoffset-1)*multiples_p[1], 0*multiples_p[2], -1.0, 0*multiples_p[4]];
        ub = [0.9*multiples_p[0], (eoffset+1)*multiples_p[1], rscale*10*multiples_p[2], 1.0, rbgoffset*10*multiples_p[4]]; 


        lsqf1Result = least_squares(self.f1,p0,bounds=(lb,ub),
                            args=(X0,Y0,des,dem,rsgm,egaus,rangegaus,degaus,cfwhmm,nsteps,nstepm,multiples_p,areagaus,ngaus,False))
        p = lsqf1Result.x

        v0,rsga,rsgx,desgm = self.f1(p,X0,Y0,des,dem,rsgm,egaus,rangegaus,degaus,cfwhmm,nsteps,nstepm,multiples_p=multiples_p,areagaus=areagaus,ngaus=ngaus,plotter=True)
        v = sum(v0**2)
        p=[p[0]/multiples_p[0], p[1]/multiples_p[1], p[2]/multiples_p[2], p[3]/multiples_p[3], p[4]/multiples_p[4]]
        datarf=datar
        datarf[:,0]=datar[:,0]-p[1]
        datarf[:,1]=datar[:,1]/p[2]-p[3]*datarf[:,0]-p[4]
        dedth=energy*(np.pi/180)*np.cos(thbs)/np.sin(thbs)

        gaus=np.exp(-4*math.log(2)*(egaus/p[0])**2)
        gaus=gaus/np.sum(gaus)
        gaus=gaus*areagaus


        if plotter:
            rangeDataRf = np.max(datarf)-np.min(datarf) 
            print('                      dE/dth = ' + f'{dedth:.3}' + ' eV/deg')
            print('                Least-square = ' + f'{v:.3}')
            print('fitted Gaussian width = p(0) = ' + f'{p[0]:.3f}' + 'eV')
            print('        Energy offset = p(1) = ' + f'{p[1]:.2f}' + 'eV')
            print('   Incident intensity = p(2) = ' + f'{p[2]:.3}')
            print('   R background slope = p(3) = ' + f'{p[3]:.3}')
            print('  R background offset = p(4) = ' + f'{p[4]:.3}')
            print('                        Rmax = ' + f'{max(rsgm):.3f}')
            print('                Energy range = ' + f'{rangeDataRf:.3f}' + ' eV')
            print('               Angular range = ' + f'{rangeDataRf/dedth:.3f}' + 'deg')
            print('\n')

        #%%%%%%%%%%%%%%%%%%%%%
        #% Fit XSW yield curve
        #%%%%%%%%%%%%%%%%%%%%%

        datayf=datay[:,:]
        datayf[:,0] = datay[:,0]-p[1]
        noffbragg=5
        datayfNorm = np.sum([datay[i,1] for i in range(datay.shape[0]) if (i<noffbragg or i>datay.shape[0]-noffbragg)])
        #datayfNorm = 


                
        datayf[:,1]=datay[:,1]*2*noffbragg/datayfNorm
        X=np.array(datayf[:,0])
        Y=np.array(datayf[:,1])



        
        res = np.ones((2*step+1,step+1))
        for nn in range(2*step+1):
            for ll in range(step+1):
                fh = nn/step
                ph = ll/step
                
                res[nn,ll]=np.sum(np.abs(self.f2([1,fh,ph],X,Y,xs,rs,rmn,desgm,gaus=gaus,C1=C1,C2=C2,C3=C3,plotter=False)))


        minMatrix = np.min(res)
        fh0,ph0 = np.where(res==minMatrix)
        fh0 = fh0[0]
        ph0 = ph0[0]
        if plotter:
            plt.show()
            plt.clf()
            plt.imshow(res,extent=[0,1,2,0])
            plt.xlabel('pH')
            plt.ylabel('fH')
            plt.suptitle(self.names[Region] + ' peak ' + str(peakNo) + " of " + str(number_peaks))               
            plt.show()
            plt.clf()            
            
            #colormap('jet')
            #xlabel('coherent position')
            #ylabel('coherent fraction')
            #colorbar
            #savefig([dataprefix '_2D.fig'])
            #saveas(gcf,[dataprefix '_2D'],'pdf')
            #saveas(gcf,[dataprefix '_2D'],'svg')

        #q_in = [1,0.51659,0.33548]
        q_in = [1,fh0/step,ph0/step]
        q_in = np.array(q_in)
        
        
        lb = np.array([0,0,-1])
        ub = np.array([4,2,2])
        
        
        lsqf2Result = least_squares(self.f2,q_in,bounds=(lb,ub),
                     method = 'trf',#ftol=1e-8,xtol=1e-8,max_nfev=1000,
                     args=(X,Y,xs,rs,rmn,desgm,gaus,C1,C2,C3,False))
        q = lsqf2Result.x
        v,ysgm,ysgmx = self.f2(q,X,Y,xs,rs,rmn,desgm,gaus=gaus,C1=C1,C2=C2,C3=C3,plotter=True)
        
        v2 = sum(v)**2#*dataerr[:,1]**2)**2)
        #jacob2 = np.sqrt(np.sum((jacobian[:,1]*dataerr[:,1]**2)**2))
        #jacob3 = np.sqrt(np.sum((jacobian[:,2]*dataerr[:,2]**2)**2))
        #err_fh = np.sqrt(v2/(len(Y)-3)/jacob2*2)
        #err_ph = np.sqrt(v2/(len(Y)-3)/jacob3*2)

        #err_q = [err_fh,err_ph];
        while q[2]>=1.0:
            q[2]=q[2]-1.0
        while q[2]<0.0:
            q[2]=q[2]+1

        datayf[:,1]=datayf[:,1]*q[0];
        ys=rs*C1+2*C2*q[1]*np.sqrt(rs)*np.cos(C3+np.arctan2(np.imag(xs),np.real(xs))-2*np.pi*q[2]);
        if fwhmgaus>0:
            ysg=fftconvolve(gaus,ys,'full')
        else:
            ysg=ys
            
        ysgm=fftconvolve(rmn,ysg,'full')+1
        yob=datayfNorm/q[0]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% Save the data, fit and figure
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if plotter:
            # Screen output of fitting results for yield
            
            print('               Least-square = ' + f'{v2:.3}')
            print('                 Yob = q(1) = ' + f'{yob:.3}')
            print('                  fH = q(2) = ' + f'{q[1]:.3f}')# + ' ± ' + str(err_fh))
            print('                  PH = q(3) = ' + f'{q[2]:.3f}')# + ' ± ' + str(err_ph))
            print('    Non-dipolar corrections used:')
            print('           C1 = (1+Q)/(1-Q) = ' + f'{C1:.4f}')
            print('           C2 =   1/(1-Q)   = ' + f'{C2:.4f}')
            print('           C3 = phase shift = ' + f'{C3:.4f}')

            #plot XSW profiles (relative to bragg reflection)
            
            plt.plot(datarf[:,0],datarf[:,1],'ok',linewidth=3)#,sqrt(datar(:,2))/p(3), 'oc','Linewidth', 3)
            plt.plot(datayf[:,0],datayf[:,1],'ok',linewidth=3) #,dataerr(:,2)/yob, 'og','Linewidth', 3)
            #plot fits (relative to bragg reflection)
            plt.plot(datarf[:,0],rsgx, 'k', linewidth=3)
            plt.plot(datarf[:,0],ysgmx, 'r', linewidth=3)
            plt.xlabel('$hv - E_{Bragg}$ (eV)')
            plt.ylabel('normalised intensity')
            plt.suptitle(self.names[Region] + ' peak ' + str(peakNo+1) + " of " + str(number_peaks) + ' XSW fit') 
            

            
            #label plots
            #set(gca,'fontsize',20)
            #xlabel('E - E_{Bragg} (eV)','FontSize',20,'FontWeight','bold','Color','k')
            #ylabel('Relative Absorption','FontSize',20,'FontWeight','bold','Color','k')
            #figTxt =('                 Filename: ' + dataPrefix + '\n\n' + 'General Information: \n' + 
            #        '   Energy = ' + str(energy) + ' eV,      mono = Si(' + str(hm[0]) + ' ' + str(hm[1]) + ' ' + str(hm[2]) + '),\n' + 
            #        '   sample = ' + samp + '(' + str(hs[0]) + ' ' + str(hs[1]) + ' ' + str(hs[2]) + '),     dE/dth = ' + str(dedth) + ' (ev/deg)\n' +
            #        'Fitting the Bulk Reflection Profile: ' '\n' +
            #        '   Least-square = ' + str(v) + ',       bs = ' + str(bs) + ',       bm = ' + str(bm) + '\n' +
            #        '   Gaussian width = ' + str(fwhmgaus) + ' (eV),   Gaussian area = ' + str(areagaus) + '\n' +
            #        '   Rmax = ' + str(max(rsgm)) + ',                 I0 = ' + str(p(3)) + '(cts)' '\n' +
            #        '   Energy range = ' + str(max(datarf[:,0])-min(datarf[:,0])) + ' (eV), Angular range = ' + str((max(datarf[:,0])-min(datarf[:,0]))/dedth) + ' (deg)\n' +
            #        'Fitting the XSW Absorption Curve: \n' +
            #        '   Least-square = ' + str(v2) + '\n' +
            #        '   Yob = ' + str(yob) + '\n' +
            #        '   fH = ' + str(q[1]) + '\n' + #' ± ' + str(round(err_fh,2,'significant')) + '\n' +
            #        '   PH = ' + str(q[2]) + '\n' + #' ± ' + str(round(err_ph,2,'significant')) + '\n' +
            #        'Non-dipolar corrections used:\n' +
            #        '   C1 = (1+Q)/(1-Q) = ' + str(C1) + '\n' +
            #        '   C2 = 1/(1-Q) = ' + str(C2) + '\n' +
            #        '   C3 = phase shift = ' + str(C3) + '\n' +
            #        '   Q =  ' + str(Q))
       
   
    def f1(self,p,X0,Y,des,dem,rsgm,egaus,rangegaus,degaus,cfwhmm,nsteps,nstepm,multiples_p=[10, 1/1e3, 1/1e4, 1e3, 1e2],areagaus=1,ngaus=1010,plotter=True):
        p=[p[0]/multiples_p[0], p[1]/multiples_p[1], p[2]/multiples_p[2], p[3]/multiples_p[3], p[4]/multiples_p[4]]
        fwhmgaus = p[0]
        if fwhmgaus > 0:
            gaus=np.exp(-4*math.log(2)*(egaus/fwhmgaus)**2)
            gaus=gaus/np.sum(gaus)
            gaus=gaus*areagaus
            #plt.plot(egaus,gaus)
            #plt.suptitle('gaussian')
            #plt.show()
            #plt.clf()
            a1=[i for i in range(nsteps+ngaus-1)]
            desg=des[0]-(rangegaus+egaus[0])+np.array(a1)*degaus
            rsga=fftconvolve(gaus,rsgm,'full')
            #rsga=fftconvolve(gaus,gaus,'full');
        else:
            desg=des;
            rsga=rsgm;
        #

        a1=[i for i in range(nstepm+desg.shape[0]-1)]
        desgm=desg[0]-(max(dem)-cfwhmm)+np.array(a1)*degaus
        
        X=np.squeeze(np.array([i-p[1] for i in X0]))

        #
        if max(X)>max(desgm):
            X=X-(max(X)-max(desgm))
        elif min(X)<min(desgm):
            X=X-(min(X)-min(desgm))
        
        tck        = interpolate.splrep(desgm,rsga,s=0)
        rsgx       = interpolate.splev(X,tck,der=0)  
        #print(str(rsgx))
        e=(Y/p[2]-p[3]*X-p[4]-rsgx)*Y
        
        if plotter:
            #plt.plot(X,Y/p[2])
            #plt.plot(X,rsgx)
            #plt.xlabel('$hv - E_{Bragg}$ (eV)')
            #plt.ylabel('normalised intensity')
            #plt.suptitle('fit of the reflection')
            #plt.show()
            #plt.clf()            
            return e,rsga,rsgx,desgm
        else:
            return e
    
    def f2(self,q,X,Y,xs,rs,rmn,desgm,gaus=None,C1=1,C2=1,C3=0,plotter=True): #Function defining the difference between the theoretical and measured yield curves (Chi^2 can be potentially introduced here)

        ys=rs*C1+2*q[1]*C2*np.sqrt(rs)*np.cos(C3+np.arctan2(np.imag(xs),np.real(xs))-2*np.pi*q[2]);
        if gaus is None:
            ysg = ys
        else:
            ysg=fftconvolve(gaus,ys,'full') # Convolution of yield with a Gaussian function
        #
        ysgm=fftconvolve(rmn,ysg)+1; # Convolution of the Gaussian convoluted yield with mono rocking curve   
        #
        tck        = interpolate.splrep(desgm,ysgm,s=0)
        ysgmx       = interpolate.splev(X,tck,der=0)          
        e=(q[0]*Y-ysgmx)#/(dataerr[:,1]**2);
        #print("e shape = " + str(e.shape))
        #plt.plot(X,ysgmx)
        #plt.plot(X,q[0]*Y)
        if plotter:
            return e,ysgm,ysgmx
        else:
            return e
   
        
    def q_param(self,Z,n,l,js,plotter=True):
        if isinstance(Z,str):
            Z = self.z.index(Z)
        if not isinstance(l,str):
            l = self.lNumber(l)
        if l == 's':
            js = ''
        else:
            js = self.jsNumber(js)
        
        if plotter:
            print("\n\n            Measured orbital = " + self.z[Z] + ' ' +str(n)+l+js)
        
        file = open(self.nixswDatadir + '/' + self.nixswQparamFile)
        
        name = str(Z) + ' ' + str(n) + l + js + '\n'
    
        
        fileText = file.readlines()   
        idxName = fileText.index(name)
        Eb = int(fileText[idxName+1])
        Erow = list(map(float,fileText[idxName+2].split()))
        
        beta = list(map(float,fileText[idxName+4].split()))
        gamma = list(map(float,fileText[idxName+5].split()))
        delta = list(map(float,fileText[idxName+6].split()))
        return beta,gamma,delta,Eb,Erow
        
        
    
    def lNumber(self,l):
        return {0:'s',1:'p',2:'d',3:'d'}[l]
    
    
    def jsNumber(self,js):
        return {0.5:'1/2',
                1.5:'3/2',
                2.5:'5/2',
                3.5:'7/2'}[js]
          
    
    def plotFitResult(self,Region=0,paramTitle='intensity1',number_peaks=0):
        hvExp = self.hv[Region]
        iExp = [self.cs[Region][number_peaks][i][paramTitle] for i in range(len(hvExp))]
        if len(hvExp)>1:
            plt.plot(hvExp,iExp,linewidth=2)
            plt.xlabel("$hv - E_{Bragg}$ (eV)")
            plt.ylabel("normalised intensity")
            plt.suptitle(paramTitle+" for a " + str(number_peaks) + "peak fit")
            plt.show()
            plt.clf()
        else:
            print(paramTitle + ' = ' + str(iExp[0]))
        
        
    
    def nnFitPeaks(self,Region=0,plotter=True,angleFit=False):
        if Region not in self.cs:
            self.fittingParam(Region,1)
        [dataCount,dataNorm,yNew] = self.dataCounter(Region=Region)
        predList = self.countPeaks(dataCount,Region=Region,plotter=plotter)
        predMax = max(predList)   
        predAlt = [predList.index(i)+1 for i in predList if (i > predMax/self.lowestAlt)]   
        for number_peaks in predAlt:
            if number_peaks not in self.cs[Region]:
                self.fittingParam(Region,number_peaks,wholeRegion=False)               
            [predPosAvg,stdPos] = self.posPeaks(dataCount,dataNorm,yNew,Region=Region,number_peaks=number_peaks,plotter=plotter)
            #
            self.setPredictedPos(predPosAvg,stdPos,Region=Region,number_peaks=number_peaks,angleFit=angleFit)
            #
            self.peakFit(Region=Region,number_peaks=number_peaks)

    def setPredictedPos(self,predPosAvg,stdPos,Region=0,number_peaks=1,angleFit = False):
        if not angleFit:
            self.setParam('bgr0',np.min(np.sum(self.data[Region][0],axis=0)),Region=Region,number_peaks=number_peaks)
        for nn in range(number_peaks):
            if nn == 0:
                self.setParam('energy'+str(nn+1),predPosAvg[nn],Region=Region,number_peaks=number_peaks)
                self.setLB('energy'+str(nn+1),predPosAvg[nn]-stdPos[nn],Region=Region,number_peaks=number_peaks)
                self.setUB('energy'+str(nn+1),predPosAvg[nn]+stdPos[nn],Region=Region,number_peaks=number_peaks)    
            else:
                Ediff = predPosAvg[nn]-predPosAvg[0]     
                self.setParam('energy'+str(nn+1),Ediff,Region=Region,number_peaks=number_peaks)
                self.setLB('energy'+str(nn+1),Ediff-stdPos[nn],Region=Region,number_peaks=number_peaks)
                self.setUB('energy'+str(nn+1),Ediff+stdPos[nn],Region=Region,number_peaks=number_peaks)                                    
            self.setFit('energy'+str(nn+1),True,Region=Region,number_peaks=number_peaks)
            self.setFit('intensity'+str(nn+1),True,Region=Region,number_peaks=number_peaks)
            absVal = abs(np.squeeze(self.energy[Region])-predPosAvg[nn]).tolist()
            indexVal = absVal.index(min(absVal))
            if not angleFit:
                dataHold = np.sum(self.data[Region][0],axis=0)
                Ival = dataHold[indexVal]-np.min(dataHold)
            if Ival < 0:
                Ival = 0
            self.setParam('intensity'+str(nn+1),Ival,Region=Region,number_peaks=number_peaks)
            self.setFit('width'+str(nn+1),True,Region=Region,number_peaks=number_peaks)
        for nn in range(number_peaks,6):
            self.setFit('energy'+str(nn+1),False,Region=Region,number_peaks=number_peaks)
            self.setFit('intensity'+str(nn+1),False,Region=Region,number_peaks=number_peaks)
            self.setFit('width'+str(nn+1),False,Region=Region,number_peaks=number_peaks)      
            
    def setPredictedWidth(self,predPosWidth,stdWidth,Region=0,number_peaks=1,angleFit = False):
        for nn in range(number_peaks):
            self.setParam('width'+str(nn+1),predPosWidth[2*nn],Region=Region,number_peaks=number_peaks)
            self.setLB('width'+str(nn+1),predPosWidth[2*nn]-stdWidth[2*nn],Region=Region,number_peaks=number_peaks)
            self.setUB('width'+str(nn+1),predPosWidth[2*nn]+stdWidth[2*nn],Region=Region,number_peaks=number_peaks)    
            self.setParam('lwidth'+str(nn+1),predPosWidth[2*nn+1],Region=Region,number_peaks=number_peaks)
            self.setLB('lwidth'+str(nn+1),predPosWidth[2*nn+1]-stdWidth[2*nn+1],Region=Region,number_peaks=number_peaks)
            self.setUB('lwidth'+str(nn+1),predPosWidth[2*nn+1]+stdWidth[2*nn+1],Region=Region,number_peaks=number_peaks)                                                     
            self.setFit('width'+str(nn+1),True,Region=Region,number_peaks=number_peaks)
            self.setFit('lwidth'+str(nn+1),True,Region=Region,number_peaks=number_peaks)
            self.setPin('lwidth'+str(nn+1),0,Region=Region,number_peaks=number_peaks)
        for nn in range(number_peaks,6):
            self.setFit('energy'+str(nn+1),False,Region=Region,number_peaks=number_peaks)
            self.setFit('intensity'+str(nn+1),False,Region=Region,number_peaks=number_peaks)
            self.setFit('width'+str(nn+1),False,Region=Region,number_peaks=number_peaks)         
            self.setFit('lwidth'+str(nn+1),False,Region=Region,number_peaks=number_peaks) 

    def dataCounter(self,Region=0,angleFit=False,energyFit=False):
        dataShape = self.data[Region].shape
        if not angleFit and not energyFit:
            if len(dataShape)==3:
                sumData = np.squeeze(np.sum(self.data[Region],axis=(0,1)))
            elif len(dataShape)==2:
                sumData = np.squeeze(np.sum(self.data[Region],axis=0))
            else:
                print('Data is not in expected 3D or 2D format')
                return
        if energyFit and not angleFit:
            if len(dataShape)==3:
                sumData = np.squeeze(np.sum(self.data[Region],axis=(1)))  
            else:
                sumData = np.squeeze(self.data[Region])
        xOld = [x*self.nnDataPoints/len(np.squeeze(self.energy[Region])) for x in range(0,len(np.squeeze(self.energy[Region])))]       
        xNew = [x for x in range(0,self.nnDataPoints)]
        if not energyFit and not angleFit:
            dataSub = sumData-np.ndarray.min(sumData)
            dataNorm = dataSub/np.ndarray.max(dataSub)
            tck = interpolate.splrep(xOld,dataNorm,s=0)
            yNew = np.atleast_2d(interpolate.splev(xNew,tck,der=0)).transpose()
            dataCount = np.tile(yNew,(1,1,1))
        elif energyFit and not angleFit:
            dataSub = np.array([sumData[i]-np.ndarray.min(sumData[i]) for i in range(len(self.hv[Region]))])
            dataNorm = np.array([dataSub[i]/np.ndarray.max(dataSub[i]) for i in range(len(self.hv[Region]))])
            yNew = np.array([np.transpose(np.atleast_2d(interpolate.splev(xNew,interpolate.splrep(xOld,dataNorm[i],s=0)))) for i in range(len(self.hv[Region]))])
            dataCount = yNew
        return dataCount,dataNorm,yNew

    def countPeaks(self,dataCount,Region=0,energyFit=False,plotter=True):
        if energyFit:
            pred = np.zeros((len(self.hv[Region]),6))   
        if not energyFit:
            pred = np.zeros((1,6))
        print("Predicting number of peaks: ")
        for nn in range(self.num_nets):
            print('...network = '+str(nn))
            name = self.model_prefix+str(self.model_numbers[nn])
            model = load_model(self.model_dir+'/'+name+'_model.h5')
            pred+= model.predict(dataCount)        
        pred/=(nn+1)
        if energyFit:
            pred = np.atleast_2d(np.average(pred,axis=0))
        predList = [pred.tolist()[0][x]*100 for x in range(6)]
        if plotter:
            plt.clf()
            X = [1,2,3,4,5,6]
            
            plt.bar(X,predList)
            plt.xlabel('number of peaks')
            plt.ylabel('predicted likelihood (%)')
            plt.show()
            plt.clf()
        return predList
    
    def posPeaks(self,dataCount,dataNorm,yNew,Region=0,number_peaks=1,energyFit=False,plotter=True):
        
        minEnergy = np.ndarray.min(self.energy[Region])
        maxEnergy = (np.ndarray.max(self.energy[Region])-minEnergy)/(self.nnDataPoints-1)  
        if not energyFit:
            predPos = np.zeros((self.num_netsPos,1,number_peaks))
        elif energyFit:
            predPos = np.zeros((self.num_netsPos,len(self.hv[Region]),number_peaks))
        print("Predicting positions: ")
        for nn in range(self.num_netsPos):
            print('...network = '+str(nn))
            name = self.model_prefixPos+str(number_peaks)+'_'+str(self.model_numbersPos[nn])
            model = load_model(self.model_dirPos+'/'+name+'_model.h5')
            predPos[nn]= model.predict(dataCount)   
        if not energyFit:   
            if number_peaks == 1:
                predPosAvg = [(self.nnDataPoints-np.average(np.squeeze(predPos))*(self.nnDataPoints-1))*maxEnergy+minEnergy]
                stdPos = [np.std(np.squeeze(predPos))*(self.nnDataPoints-1)*2*maxEnergy]
            else:
                predPosAvg = [(self.nnDataPoints-np.average(np.squeeze(predPos),axis=0)[x]*(self.nnDataPoints-1))*maxEnergy+minEnergy for x in range(number_peaks)]
                stdPos = [np.std(np.squeeze(predPos),axis=0)[x]*(self.nnDataPoints-1)*2*maxEnergy for x in range(number_peaks)]
                
        elif energyFit: 
            if number_peaks == 1:
                predPosAvg = [(self.nnDataPoints-np.squeeze(np.average(predPos,axis=(0,1)))*(self.nnDataPoints-1))*maxEnergy+minEnergy]
                stdPos = [np.std(np.squeeze(predPos))*(self.nnDataPoints-1)*2*maxEnergy]
            else:
                predPosAvg = [(self.nnDataPoints-np.squeeze(np.average(predPos,axis=(0,1)))[x]*(self.nnDataPoints-1))*maxEnergy+minEnergy for x in range(number_peaks)]
                stdPos = [np.squeeze(np.std(predPos,axis=(0,1)))[x]*(self.nnDataPoints-1)*2*maxEnergy for x in range(number_peaks)]                
            dataNorm=np.average(dataNorm,axis=0)
            yNew=np.average(yNew,axis=0)
        if plotter:
            xOld = [x*self.nnDataPoints/len(dataNorm) for x in range(0,len(dataNorm))]       
            xNew = [x for x in range(0,self.nnDataPoints)]            
            colours = ['r','b','c','m','g','y']
            plt.clf()
            plt.plot([(self.nnDataPoints-xOld[x])*maxEnergy+minEnergy for x in range(len(xOld))],dataNorm,'k--')
            plt.plot([(self.nnDataPoints-xNew[x])*maxEnergy+minEnergy for x in range(self.nnDataPoints)],yNew,'k-')    
            for nn in range(number_peaks):
                plt.plot([predPosAvg[nn],predPosAvg[nn]],[0,1],colours[nn]+'-')
                plt.plot([predPosAvg[nn]+stdPos[nn],predPosAvg[nn]+stdPos[nn]],[0,1],colours[nn]+'--')
                plt.plot([predPosAvg[nn]-stdPos[nn],predPosAvg[nn]-stdPos[nn]],[0,1],colours[nn]+'--')     
            plt.xlabel('binding energy (eV)')
            plt.ylabel('intensity (arb. units)')
            plt.suptitle(self.names[Region] + ' ' + str(number_peaks) + ' peak fit')
            plt.yticks([],[])
            plt.gca().invert_xaxis()        
            plt.show() 
        return predPosAvg,stdPos
    
    def widthPeaks(self,dataCount,dataNorm,yNew,predPosAvg,Region=0,number_peaks=1,energyFit=False,plotter=True):   
        minEnergy = np.ndarray.min(self.energy[Region])
        maxEnergy = (np.ndarray.max(self.energy[Region])-minEnergy)/(self.nnDataPoints-1) 
        #predPos = np.array([[[(self.nnDataPoints-(predPosAvg-minEnergy)/maxEnergy)/(self.nnDataPoints-1)]]])
        predPos = np.array([(self.nnDataPoints-(predPosAvg-minEnergy)/maxEnergy)/(self.nnDataPoints-1)])
        if not energyFit:
            predWidth = np.zeros((self.num_netsWidth,1,2*number_peaks))
        elif energyFit:
            predWidth = np.zeros((self.num_netsWidth,len(self.hv[Region]),2*number_peaks))
        print("Predicting widths: ")
        for nn in range(self.num_netsWidth):
            print('...network = '+str(nn))
            name = self.model_prefixWidth+str(number_peaks)+'_'+str(self.model_numbersWidth[nn])
            model = load_model(self.model_dirWidth+'/'+name+'_model.h5')
            predWidth[nn]= model.predict([dataCount,predPos]) 
        e = np.squeeze(self.energy[Region])
        stepE = np.average([e[i]-e[i-1] for i in range(1,len(e))])
        rangeE = max(e)-min(e)            
        if not energyFit:   
            #if number_peaks == 1:
            predWidthAvg = 2*np.average(np.squeeze(predWidth),axis=(0))*0.25*rangeE+0.1
            stdWidth = 2*np.std(np.squeeze(predWidth),axis=(0))*0.25*rangeE+0.1
            #else:
            #     
                #predWidthAvg = [np.average(np.squeeze(predWidth),axis=(0))[x+y]*0.25*rangeE+0.1 for x in range(0,number_peaks) for y in range(2)]
                #stdWidth = [np.std(np.squeeze(predWidth),axis=(0))[x+y]*0.25*rangeE+0.1 for x in range(0,number_peaks) for y in range(2)]
        elif energyFit: 
            #if number_peaks == 1:
            predWidthAvg = 2*np.squeeze(np.average(predWidth,axis=(0)))*0.25*rangeE+0.1
            stdWidth = 2*np.std(np.squeeze(predWidth),axis=(0))*0.25*rangeE+0.1
            #else:
                #predWidthAvg = [np.average(np.squeeze(predWidth),axis=(0))[2*x+y]*0.25*rangeE+0.1 for x in range(0,number_peaks) for y in range(2)]
                #stdWidth = [np.std(np.squeeze(predWidth),axis=(0))[2*x+y]*0.25*rangeE+0.1 for x in range(0,number_peaks) for y in range(2)]         
            dataNorm=np.average(dataNorm,axis=0)
            yNew=np.average(yNew,axis=0)
        if plotter:
            xOld = [x*self.nnDataPoints/len(dataNorm) for x in range(0,len(dataNorm))]       
            xNew = [x for x in range(0,self.nnDataPoints)]     
            newE = [max(e)+rangeE+i*stepE for i in range(len(e)*3-1)]   
            rangeStep = int(abs(rangeE/stepE))
            colours = ['r','b','c','m','g','y']
            plt.clf()
            plt.plot([(self.nnDataPoints-xOld[x])*maxEnergy+minEnergy for x in range(len(xOld))],dataNorm,'k--')
            plt.plot([(self.nnDataPoints-xNew[x])*maxEnergy+minEnergy for x in range(self.nnDataPoints)],yNew,'k-')   
           #print(predWidth)
           #print(predWidthAvg)
            
            for nn in range(number_peaks):
                #peakHold = ln.DSVoigtStep(newE,[1,predPosAvg[nn],predWidthAvg[nn][0],predWidthAvg[nn][1],0,0])
                #peakHoldU = ln.DSVoigtStep(newE,[1,predPosAvg[nn],predWidthAvg[nn][0]+stdWidth[nn][0],predWidthAvg[nn][1]+stdWidth[nn][1],0,0])
                #peakHoldL = ln.DSVoigtStep(newE,[1,predPosAvg[nn],predWidthAvg[nn][0]-stdWidth[nn][0],predWidthAvg[nn][1]-stdWidth[nn][1],0,0])
                peakHold = ln.DSVoigtStep(newE,[1,predPosAvg[nn],predWidthAvg[2*nn],predWidthAvg[2*nn+1],0,0])
                peakHoldU = ln.DSVoigtStep(newE,[1,predPosAvg[nn],predWidthAvg[2*nn]+stdWidth[2*nn],predWidthAvg[2*nn+1]+stdWidth[2*nn+1],0,0])
                peakHoldL = ln.DSVoigtStep(newE,[1,predPosAvg[nn],predWidthAvg[2*nn]-stdWidth[2*nn],predWidthAvg[2*nn+1]-stdWidth[2*nn+1],0,0])                
                peak = [peakHold[i] for i in range(rangeStep,2*rangeStep+1)]
                peakU = [peakHoldU[i] for i in range(rangeStep,2*rangeStep+1)]
                peakL = [peakHoldL[i] for i in range(rangeStep,2*rangeStep+1)]
                plt.plot([(self.nnDataPoints-xOld[x])*maxEnergy+minEnergy for x in range(len(xOld))],peak,colours[nn]+'-')
                plt.plot([(self.nnDataPoints-xOld[x])*maxEnergy+minEnergy for x in range(len(xOld))],peakU,colours[nn]+'--')
                plt.plot([(self.nnDataPoints-xOld[x])*maxEnergy+minEnergy for x in range(len(xOld))],peakL,colours[nn]+'--')     
            plt.xlabel('binding energy (eV)')
            plt.ylabel('intensity (arb. units)')
            plt.suptitle(self.names[Region] + ' ' + str(number_peaks) + ' peak fit')
            plt.yticks([],[])
            plt.gca().invert_xaxis()        
            plt.show() 
        return predWidthAvg,stdWidth    
            
    def peakFit(self,Region=0,number_peaks=1,plotter=True):
        flags = list(self.flags[Region][number_peaks].values())
        titles = [self.titles[i] for i in range(len(self.titles)) if flags[i]]
        ub = [list(self.ub[Region][number_peaks].values())[i] for i in range(len(self.ub[Region][number_peaks])) if flags[i]]
        lb = [list(self.lb[Region][number_peaks].values())[i] for i in range(len(self.lb[Region][number_peaks])) if flags[i]]
        c  = [list(self.cs[Region][number_peaks][0].values())[i] for i in range(len(self.cs[Region][number_peaks][0])) if flags[i]]
        
        errorBounds = [ub[i]<lb[i] for i in range(len(ub))]
        
        if any(errorBounds):
            for nn in errorBounds:
                if nn:
                    print(titles[nn] + ' has a lower bound higher than its upper bound')
                    print('ub = ' + str(ub[nn]))
                    print('lb = ' + str(lb[nn]))  
            return
        
        
        flagsAuger = list(self.flagsAuger[Region][number_peaks].values())
        titlesAuger = [self.titlesAuger[i] for i in range(len(self.titlesAuger)) if flagsAuger[i]]
        ubAuger = [list(self.ubAuger[Region][number_peaks].values())[i] for i in range(len(self.ubAuger[Region][number_peaks])) if flagsAuger[i]]
        lbAuger = [list(self.lbAuger[Region][number_peaks].values())[i] for i in range(len(self.lbAuger[Region][number_peaks])) if flagsAuger[i]]
        cAuger  = [list(self.csAuger[Region][number_peaks][0].values())[i] for i in range(len(self.csAuger[Region][number_peaks][0])) if flagsAuger[i]]   
        errorBounds = [ubAuger[i]<lbAuger[i] for i in range(len(ubAuger))]
        if any(errorBounds):
            for nn in errorBounds:
                if nn:
                    print(titlesAuger[nn] + ' has a lower bound higher than its upper bound')
                    print('ub = ' + str(ub[nn]))
                    print('lb = ' + str(lb[nn]))  
            return
        e = np.squeeze(self.energy[Region])
        stepE = np.average([e[i]-e[i-1] for i in range(1,len(e))])
        rangeE = max(e)-min(e)
        newE = [max(e)+rangeE+i*stepE for i in range(len(e)*3-1)]
        rangeStep = int(abs(rangeE/stepE))
        print("Fitting XP spectra: ")
        
        for nn in range(len(self.hv[Region])):
            print("... fitting hv = " + str(nn) + "/" + str(len(self.hv[Region])))
            c = [list(self.cs[Region][number_peaks][nn].values())[i] for i in range(len(self.cs[Region][number_peaks][nn])) if flags[i]]
            cAuger  = [list(self.csAuger[Region][number_peaks][nn].values())[i] for i in range(len(self.csAuger[Region][number_peaks][nn])) if flagsAuger[i]] 
            cNow = c
            cNow.extend(cAuger)                  
            if nn == 0 and plotter:                    
                self.fitNow(cNow,flags,flagsAuger,titles,titlesAuger,newE,rangeStep,nn,plotter,Region=Region,number_peaks=number_peaks,hvIndex=nn)
            self.fitOut = least_squares(self.fitNow,cNow,bounds=(lb,ub),
                                        method = 'trf',#ftol=1e-8,xtol=1e-8,max_nfev=1000,
                                        args=(flags,flagsAuger,titles,titlesAuger,newE,rangeStep,nn,False,Region,number_peaks,nn))
            c_out = self.fitOut.x
            for i in titles:
                self.setParam(i,c_out[titles.index(i)],Region=Region,number_peaks=number_peaks,hvIndex=nn)   
                if nn+1 < len(self.hv[Region]):
                    self.setParam(i,c_out[titles.index(i)],Region=Region,number_peaks=number_peaks,hvIndex=nn+1)   
            if plotter:
                self.fitNow(c_out,flags,flagsAuger,titles,titlesAuger,newE,rangeStep,nn,plotter,Region=Region,number_peaks=number_peaks,hvIndex=nn)
            for i in titlesAuger:
                self.setParamAuger(i,c_out[i+len(titles)],Region=Region,number_peaks=number_peaks,hvIndex=nn)
            if nn < len(self.hv[Region])-1 and nn+1 not in self.cs[Region][number_peaks]:
                self.cs[Region][number_peaks].update(self.cs[Region][number_peaks][nn])
            
    def fitNow(self,c,flags,flagsAuger,titles,titlesAuger,newE,rangeStep,ii,plotter,Region=0,number_peaks=1,hvIndex=0,angleFit=False):
        cHere = [list(self.cs[Region][number_peaks][hvIndex].values())[i] 
                 if not list(self.flags[Region][number_peaks].values())[i] 
                 else c[titles.index(list(self.flags[Region][number_peaks].keys())[i])] 
                 for i in range(len(self.flags[Region][number_peaks]))]
        cHereAuger = [list(self.csAuger[Region][number_peaks][hvIndex].values())[i] 
                      if not list(self.flagsAuger[Region][number_peaks].values())[i] 
                      else c[len(flags)+titlesAuger.index(list(self.flagsAuger[Region][number_peaks].keys())[i])] 
                      for i in range(len(self.flagsAuger[Region][number_peaks]))]
        if plotter:
            plt.clf()
        lineshape = self.fitLineshape(cHere,cHereAuger,newE,rangeStep,ii,plotter,Region=Region,number_peaks=number_peaks)
        if plotter:
            #self.testme = lineshape
            plt.plot(np.squeeze(self.energy[Region]),lineshape,'k')
            if not angleFit:
                plt.plot(np.squeeze(self.energy[Region]),np.squeeze(np.sum(self.data[Region][ii],axis=0)),'ko',markersize=0.5)
            plt.xlabel('binding energy (eV)')
            plt.ylabel('intensity (arb. units)')
            plt.suptitle(self.names[Region] + ' ' + str(number_peaks) + ' peak fit')
            #plt.yticks([],[])
            plt.gca().invert_xaxis()                 
            plt.show()
        if not angleFit:
            f = np.squeeze(np.sum(self.data[Region][ii],axis=0))-lineshape
        return f
            
    def pins_check(self,cHere,da,Region=0,number_peaks=1):
        
        if self.pins[Region][number_peaks][self.titles[da]] == 0:
            return cHere[da]
        else:
            return cHere[da]*cHere[self.pins[Region][number_peaks][self.titles[da]]]
        
    def fitLineshape(self,cHere,cHereAuger,newE,rangeStep,ii,plotter,Region=0,number_peaks=1):
        
        
        colours = ['r','b','c','m','g','y']
        zz = 0
        da = 0
        bgr0 = self.pins_check(cHere,da,Region=Region,number_peaks=number_peaks)
        da+=1
        bgr1 = self.pins_check(cHere,da,Region=Region,number_peaks=number_peaks)
         
        for kk in range(number_peaks):
            da+=1
            intensity1 = self.pins_check(cHere,da,Region=Region,number_peaks=number_peaks) 
            if kk == 0:
                da+=1
                energy1 = self.pins_check(cHere,da,Region=Region,number_peaks=number_peaks) 
                da+=1
                grad1 = self.pins_check(cHere,da,Region=Region,number_peaks=number_peaks) 
                da+=1
                e1 = energy1+(self.hv[Region][ii]-self.hv[Region][0])*grad1
                bck = [bgr0+(i-e1)*bgr1 for i in np.squeeze(self.energy[Region])]
                lineshape = bck
            else:
                da+=1
                energy2 = self.pins_check(cHere,da,Region=Region,number_peaks=number_peaks)
                da+=1
                e1 = energy1+energy2+(self.hv[Region][ii]-self.hv[Region][0])*grad1
            width1 = self.pins_check(cHere,da,Region=Region,number_peaks=number_peaks)
            da+=1
            lwidth1 = self.pins_check(cHere,da,Region=Region,number_peaks=number_peaks)  
            da+=1
            asym1 = self.pins_check(cHere,da,Region=Region,number_peaks=number_peaks)
            da+=1
            step_h = self.pins_check(cHere,da,Region=Region,number_peaks=number_peaks)
            #print("width = " + str(width1))
            #print("lwidth = " + str(lwidth1))
            #print("asym1 = " + str(asym1))
            peakHold = ln.DSVoigtStep(newE,[intensity1,e1,width1,lwidth1,asym1,step_h])
            peak = [peakHold[i] for i in range(rangeStep,2*rangeStep+1)]
            lineshape=[lineshape[i]+peak[i] for i in range(len(peak))]
            #print("lineshape1 = " +str(lineshape))
            if plotter:
                plt.plot(np.squeeze(self.energy[Region]),[peak[i]+bck[i] for i in range(len(bck))],colours[zz])
                zz+=1
                if zz == len(colours):
                    zz = 0
        return lineshape            
    def sumRegions(self,regionList,name=None):
        data = copy(self.data[regionList[0]])
        hv = copy(self.hv[regionList[0]])
        energy = copy(self.energy[regionList[0]])
        if len(hv) > 1:
            angle_test = self.angles[regionList[0]].all()
            nixswr_test = self.nixswr[regionList[0]].all() 
        else:
            angle_test = self.angles[regionList[0]]
            nixswr_test = self.nixswr[regionList[0]]
        if angle_test is not None:
            angles = copy(self.angles[regionList[0]])
        else:
            angles = None
        if nixswr_test is not None:
            nixswr = copy(self.nixswr[regionList[0]])
        else:
            nixswr = None            
        for nn in range(1,len(regionList)):
            if not hv == self.hv[regionList[nn]]:
                print("WARNING: region " + str(regionList[nn]) + " was measured at a different photon energy than region " + str(regionList[0]))
            if not data.shape == self.data[regionList[nn]].shape:
                print("Region " + str(regionList[nn]) +' has a shape of ' + str(self.data[regionList[nn]].shape) +
                      ". The previous region(s) were of shape " + str(data.shape))
            if len(hv) > 1:
                angle_test = self.angles[regionList[nn]].all()
                nixswr_test = self.nixswr[regionList[nn]].all() 
                angle_control = angles.all()
                nixswr_control = nixswr.all()
            else:
                angle_test = self.angles[regionList[nn]]
                nixswr_test = self.nixswr[regionList[nn]]  
                angle_control = angles
                nixswr_control = nixswr
                
            if nixswr_control is None and  nixswr_test is not None:
                print("Region " + str(regionList[nn]) +' has a nixswr, whereas previous regions did not')
                return
            elif nixswr_control is not None and  nixswr_test is None:
                print("Region " + str(regionList[nn]) +' has no nixswr, whereas previous regions did')
                return                
            elif nixswr_control is not None:
                nixswr+=self.nixswr[regionList[nn]]
            if angle_control is None and  angle_test is not None:
                print("Region " + str(regionList[nn]) +' has a range of angles, whereas previous regions did not')
                return 
            elif angle_control is not None and  angle_test is None:
                print("Region " + str(regionList[nn]) +' has a no angles, whereas previous regions did')
                return 
            elif not angle_control == None:
                angles+=self.angles[regionList[nn]]
            data+= self.data[regionList[nn]]
            energy+= self.energy[regionList[nn]]
        if  angle_control is not None:
            angles/=nn+1
        energy/=nn+1
        if name is None:
            name = "Sum of Region " + str(regionList[0]) + " to " + str(regionList[-1])   
        self.appendData(data,hv,name,energy,angles=angles,nixswr=nixswr)    
        
    def plotSpectrum(self,Region=0,hvIndex=-1,angleFit=False):
        plt.clf()    
        dataShape = self.data[Region].shape
        if len(dataShape)==3 and hvIndex==-1:
            if not angleFit:
                plotData = np.squeeze(np.sum(self.data[Region],axis=(0,1)))
        elif len(dataShape)==3 and hvIndex>-1:
            if not angleFit:
                plotData = np.squeeze(np.sum(self.data[Region][hvIndex],axis=(0,1)))
        elif len(dataShape)==2:
            if not angleFit:
                plotData = np.squeeze(np.sum(self.data[Region],axis=0))
        else:
            print('Data is not in expected 3D or 2D format')
            return
        plt.plot(np.squeeze(self.energy[Region]),plotData)
        plt.xlabel('binding energy (eV)')
        plt.ylabel('intensity (arb. units)')
        plt.suptitle(self.names[Region])
        plt.yticks([],[])
        plt.gca().invert_xaxis()
        
    def appendData(self,data,hv,name,energy,angles=None,nixswr=None):
        self.noRegions+=1
        self.names.append(name)
        self.hv.append(hv)
        self.energy.append(energy)
        self.data.append(data)
        if len(hv) > 1:
            angle_test = angles.all()
            nixswr_test = nixswr.all() 
        else:
            angle_test = angles
            nixswr_test = nixswr    
        if angle_test is not None:
            self.angles.append(angles)
        else: 
            self.angles.append(None)
        if nixswr_test is not None:
            self.nixswr.append(nixswr)
        else:
            self.nixswr.append(None)   
            
    def i09_export_data(self,filename=None,pathname=None):
        if pathname is not None:
            os.chdir(pathname)
        if filename is None:
            root = tk.Tk()
            root.withdraw()
            wFilename = fD.askopenfilename(title='Please select the XPS file to load in',
                                           filetypes = ('nexus file','*.nxs'))
            wFilename = os.path.split(wFilename)
            pathname=wFilename[0]+'/'
            filename=wFilename[1]
        os.chdir(pathname)    
        fid = h5py.File(filename,'r')
        fidItems = list(fid['entry1'].keys())
        for nn in fidItems:
            if nn == 'hm3amp20':
                self.i0Hard = fid['/entry1/hm3amp20/hm3amp20'][:]
            elif nn == 'sm5amp8':
                self.i0Soft = fid['/entry1/sm5amp8/sm5amp8'][:]
            elif nn == 'smpmamp39':
                self.iDrain = fid['/entry1/smpmamp39/smpmamp39'][:]  
            elif nn == 'scaler2':
                self.i0Hard = fid['/entry1/scaler2/hm3amp20'][:]
                self.i0Soft = fid['/entry1/scaler2/sm5amp8'][:]
                self.iDrain = fid['/entry1/scaler2/smpmamp39'][:] 
            elif nn == 'instrument':
                nothingHere = 1
            elif nn == 'user01':
                nothingHere = 1
            elif nn == 'before_scan':
                nothingHere = 1
            elif nn == 'ew4000':
                nothingHere = 1
            elif nn == 'end_time':
                nothingHere = 1
            elif nn == 'entry_identifier':
                nothingHere = 1
            elif nn == 'experiment_identifier':
                nothingHere = 1
            elif nn == 'scan_identifier':
                nothingHere = 1                
            elif nn == 'program_name':
                nothingHere = 1
            elif nn == 'scan_command':
                nothingHere = 1
            elif nn == 'scan_dimensions':
                nothingHere = 1
            elif nn == 'scan_dimensions':
                nothingHere = 1
            elif nn == 'start_time': 
                nothingHere = 1
            elif nn == 'title':
                nothingHere = 1
            else: 
                self.noRegions+=1
                regionItems = list(fid['/entry1/'+nn])
                self.names.append(nn)
                self.hv.append([fid['/entry1/'+nn+'/excitation_energy'][i] for i in range(len(fid['/entry1/'+nn+'/excitation_energy']))])
                self.energy.append(np.atleast_2d(fid['/entry1/'+nn+'/energies'][0]))
                data_hold = np.array([fid['/entry1/'+nn+'/image_data'][i] for i in range(len(self.hv[-1]))])
                #if len(self.hv[-1])==1 and 'angles' in regionItems:
                #    data_hold = np.array([data_hold])
                self.data.append(data_hold)
                if 'angles' in regionItems:
                    self.angles.append(fid['/entry1/'+nn+'/angles'][:])
                else: 
                    self.angles.append(None)
                if 'nixswr' in regionItems:
                    self.nixswr.append(fid['/entry1/'+nn+'/nixswr'][:])
                else:
                    self.nixswr.append(None)

                
            

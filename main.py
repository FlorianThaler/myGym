import numpy as np
import random
import tensorflow as tf

import os

from config import SEED_VAL

from lunarLanding.training_lunarLander import trainingFunc_lunarLander_deepQ, trainingFunc_lunarLander_deepPolGrads_reinforce

from lunarLanding.hyperParamTuning_lunarLander import hyperParamTuningFunc_deepQ

from utils.dataAnalysis import processTrainingResults, processHyperParamTuningResults
from lunarLanding.evaluation_lunarLander import modelEvaluationFunc_lunarLander

def trainingRunnerFunc(algorithm):

    model = None
    policy = None
    cumRwdList = []
    lossList = []

    if algorithm == 'deepQ':
        model, policy, cumRwdList, lossList = trainingFunc_lunarLander_deepQ()
    elif algorithm == 'deepPolGrads':
        model, policy, cumRwdList, lossList = trainingFunc_lunarLander_deepPolGrads_reinforce()
    else:
        print('# ### unknown algorithm')

    return model, policy, cumRwdList, lossList

def hyperParamTuningRunnerFunc(algorithm):

    study = None

    if algorithm == 'deepQ':
        study = hyperParamTuningFunc_deepQ()
    elif algorithm == 'deepPolGrads':
        print('# ### hyper parameter tuning not implemented yet for this algorithm')
    else:
        print('# ### unknown algorithm')

    return study

def seedingFunc():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(SEED_VAL)
    random.seed(SEED_VAL)
    tf.random.set_seed(SEED_VAL)

def main():

    seedingFunc()

    example = 'lunarLander'

    mode = 'training'
    # mode = 'tuning' 

    algorithm = 'deepQ'

    if example == 'lunarLander':
        if mode == 'training':
            # ### train lunar landing agent
            model, policy, cumRwdList, lossList = trainingRunnerFunc(algorithm)

            # ### process training results
            processTrainingResults(cumRwdList, lossList)

            # ### evaluate training results
            # modelEvaluationFunc_lunarLander(policy)
        elif mode == 'tuning':
            # ### tune hyperparameters by means of optuna package
            study = hyperParamTuningRunnerFunc(algorithm)

            # ### process tuning results
            processHyperParamTuningResults(study)

    elif example == 'highway':
        print('# ### not implemented yet')
    else:
        print('# ### unknown example')

if __name__ == '__main__':
    main()
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import numpy as np

import gym

from utils.deepQ import train_deepQ, createModel_deepQ, act_deepQ
from utils.deepPolicyGradients import createModel_deepPolGrads_reinforce, train_deepPolGrads_reinforce, act_deepPolGrads

from config import SEED_VAL

def createEnv_lunarLander():
    """
        this function creates the training environment; additionally the environment's random number
        generated is seeded.

    Returns:
        Env: instance of class Env representing the lunar lander environment
    """
    env = gym.make('LunarLander-v2')
    stateSpaceDim = env.observation_space.shape[0]
    numActions = env.action_space.n

    # ### seed environment
    env.seed(SEED_VAL)

    return env, stateSpaceDim, numActions

def inputGenerationFunc_lunarLander(stateList, stateSpaceDim):
    """
        given a list of environment states this function creates by rowwise
        stacking a 2d numpy array. this kind of data structure is used as input
        to the neural network

    Args:
        stateList (list): list of environment states
        stateSpaceDim (int): dimension of the state space

    Returns:
        numpy array: 2d array containing the state data
    """
    numStates = len(stateList)
    retVal = np.zeros((numStates, stateSpaceDim))

    for i in range(0, numStates):
        retVal[i, :] = stateList[i].reshape(1, -1)

    return retVal

def trainingFunc_lunarLander_deepQ(lr = 1e-3):
    """
        this function incorporates the deep Q training of the neural network 
        which is used to control the lunar lander module

    Args:
        lr (float): learning rate of the training procedure. Defaults to 1e-3.

    Returns:
        Model, function, list, list: the function returns to the caller an instance of the keras class Model, 
            representing the trained neural network, a function representing the policy obtained by the 
            learning procedure, the list of obtained cumulative rewards and the list of training losses
    """
    # ### create environment
    env, inputDim, outputDim = createEnv_lunarLander()

    # ### create neural network model using keras
    modelParamDict = {}
    modelParamDict['inputDim'] = inputDim
    modelParamDict['outputDim'] = outputDim
    layerInfoList = []
    layerInfoList.append((128, 'tanh'))
    layerInfoList.append((128, 'tanh'))
    modelParamDict['layerInfoList'] = layerInfoList

    model = createModel_deepQ(modelParamDict)                       # NOTE: this is the primary model which is used to 
                                                                    #   choose actions. it is moreover the model which 
                                                                    #   is optimised - hence it has to be compiled.

    targetModel = createModel_deepQ(modelParamDict)                 # NOTE: this is the so called target model which is
                                                                    #   used to compute the target values in the optimisation
                                                                    #   process

    # ### train model
    trainingParamDict = {}
    trainingParamDict['learningRate'] = lr
    trainingParamDict['eps_init'] = 1.0
    trainingParamDict['eps_min'] = 0.05
    trainingParamDict['memSize'] = 2 ** 16
    trainingParamDict['optimFreq'] = 1
    trainingParamDict['updateFreq'] = 2 ** 7
    trainingParamDict['numDraws_expl'] = 2 ** 10
    trainingParamDict['resetFreq_expl'] = 2 ** 6
    trainingParamDict['numDraws_trng'] = 2 ** 17
    trainingParamDict['maxNumDrawsPerEpisode'] = 2 ** 8

    cumRwdList, lossList = train_deepQ(env, inputGenerationFunc_lunarLander,\
                                        inputDim, model, targetModel, trainingParamDict)

    policy = lambda obs: act_deepQ(obs, model, inputDim, inputGenerationFunc_lunarLander)

    return model, policy, cumRwdList, lossList

def trainingFunc_lunarLander_deepPolGrads_reinforce(lr = 1e-4):
    """
        this function incorporates the deep policy gradient training for 
        the lunar landing controller. 

    Args:
        lr (float): learning rate of the training procedure. Defaults to 1e-3.

    Returns:
        Model, function, list, list: the function returns to the caller an instance of the keras class Model, 
            representing the trained neural network, a function representing the policy obtained by the 
            learning procedure, the list of obtained cumulative rewards and the list of training losses
    """#
    # ### create environment
    env, inputDim, outputDim = createEnv_lunarLander()

    # ### create neural network model using keras
    modelParamDict = {}
    modelParamDict['inputDim'] = inputDim
    modelParamDict['outputDim'] = outputDim
    layerInfoList = []
    layerInfoList.append((128, 'tanh'))
    layerInfoList.append((128, 'tanh'))
    modelParamDict['layerInfoList'] = layerInfoList

    model = createModel_deepPolGrads_reinforce(modelParamDict)

    # ### train model
    trainingParamDict = {}
    trainingParamDict['learningRate'] = lr
    trainingParamDict['numDraws_trng'] = 2 ** 17
    trainingParamDict['maxNumDrawsPerEpisode'] = 2 ** 8

    cumRwdList, lossList = train_deepPolGrads_reinforce(env, inputGenerationFunc_lunarLander,\
                                                            inputDim, outputDim, model, trainingParamDict)

    policy = lambda obs: act_deepPolGrads(obs, model, inputDim, outputDim, inputGenerationFunc_lunarLander)

    return model, policy, cumRwdList, lossList
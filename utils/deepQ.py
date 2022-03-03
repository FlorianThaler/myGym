import numpy as np

import random

from collections import deque

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform, Constant

from config import SEED_VAL

def createModel_deepQ(modelParamDict):
    """ this function provides a keras model of a deep q agent

    Args:
        modelParamDict (dictionary): dictionary containing informations on the structure
            of the neural network. its keys are inputDim, outputDim, layerInfoList

    Returns:
        keras.Model: neural network model representing the deep q agent
    """

    # define input layer
    x = layers.Input(shape = (modelParamDict['inputDim'], ))

    # define hidden layer
    layerInfoList = modelParamDict['layerInfoList']
    y = layers.Dense(layerInfoList[0][0], activation = layerInfoList[0][1], use_bias = True, kernel_initializer = glorot_uniform(seed = SEED_VAL), bias_initializer = Constant(0.1))(x)
    for i in range(1, len(layerInfoList)):
        y = layers.Dense(layerInfoList[i][0], activation = layerInfoList[i][1], use_bias = True, kernel_initializer = glorot_uniform(seed = SEED_VAL), bias_initializer = Constant(0.1))(y)

    # define output layer
    y = layers.Dense(modelParamDict['outputDim'], activation = 'linear', use_bias = True, kernel_initializer = glorot_uniform(seed = SEED_VAL), bias_initializer = Constant(0.1))(y)

    # create model
    model = keras.Model(inputs = x, outputs = y)

    return model

def updateTargetModel(model, target_model, mode = 'hard'):

    if mode == 'hard':
        pass
    elif mode == 'soft':
        pass
    else:
        pass

    return 

def compileModel_deepQ(model, learningRate = 1e-3):
    model.compile(loss = 'mse', optimizer = Adam(learning_rate = learningRate))
    return model

def act_deepQ(obs, model, stateSpaceDim, inputGenerationFunc):
    qVals = model.predict(inputGenerationFunc([obs], stateSpaceDim))
    a = np.argmax(qVals)

    return a

def add2ReplayMemory(replayMemory, obs, a, r, newObs, done):
    replayMemory.append((obs, a, r, newObs, done))

def train_deepQ(env, inputGenerationFunc, stateSpaceDim, model, target_model, paramDict):
    """ function which incorporates the deep q training approach; per default
        a target model is used as well as a replay memory

    Args:
        env (_type_): _description_
        model (_type_): _description_

    Raises:

    Returns:


    """

    # ### initialisations

    replayMemory = deque(maxlen = paramDict['memSize']) 

    cumRwdList = []
    avrgLossList = []

    compileModel_deepQ(model, paramDict['learningRate'])

    # ### exploration phase

    print('# ### exploration phase')

    obs = env.reset()
    for i in range(0, paramDict['numDraws_expl']):

        a = np.random.choice(env.action_space.n)
        newObs, r, done, _ = env.step(a)

        add2ReplayMemory(replayMemory, obs, a, r, newObs, done)
        obs = newObs.copy()

        if ((i > 1) and (i % paramDict['resetFreq_expl'] == 0)) or done:
            obs = env.reset()

    # ### training phase

    print('# ### training phase')

    # determine decay factor of exploration rate in such a way that after approximately 
    # one third of the number of draws the exploration rate is less or equal paramDict['eps_min].
    decayFactor = np.exp((3.0 / paramDict['numDraws_trng']) * np.log(paramDict['eps_min'] / paramDict['eps_init'])) 
    eps = paramDict['eps_init']

    i = 0                                      # total draw counter
    episodeId = -1
    stopTraining = False
    while not stopTraining:                     # should/could be replaced with for loop

        obs = env.reset()
        cumRwd = 0.0
        episodeId += 1
        lossList = []

        for j in range(0, paramDict['maxNumDrawsPerEpisode']):

            # ### simulation phase
            a = -1

            # action selection
            if np.random.rand() < eps:
                # choose random action
                a = np.random.choice(env.action_space.n)
            else:
                # choose action using neural network model
                a = act_deepQ(obs, model, stateSpaceDim, inputGenerationFunc)

            # apply action and make observation
            newObs, r, done, _ = env.step(a)
            cumRwd += r

            # increase draw counter
            i += 1

            # add data to replay memory
            add2ReplayMemory(replayMemory, obs, a, r, newObs, done)

            # update variables and decrease exploration rate
            obs = newObs.copy()
            eps = np.maximum(eps * decayFactor, paramDict['eps_min'])

            # ### optimisation phase
            if i % paramDict['optimFreq'] == 0:
                history = optimise_deepQ(inputGenerationFunc, stateSpaceDim, replayMemory, model, target_model)

                lossList.append(history.history['loss'])

            if i % paramDict['updateFreq'] == 0:
                # align model and target model
                target_model.set_weights(model.get_weights())

            if done:
                break

        cumRwdList.append(cumRwd)
        avrgLossList.append(np.mean(lossList))

        print('# --- episode {}: cum rwd = {:.2f}'.format(episodeId, cumRwd))

        if i > paramDict['numDraws_trng']:
            stopTraining = True

    return cumRwdList, avrgLossList

def optimise_deepQ(inputGenerationFunc, stateSpaceDim, replayMemory, model,\
                        target_model, batchSize = 2 ** 6, gamma = 0.99):

    sampleIdx = random.sample(range(0, len(replayMemory)), batchSize)

    inputSample_0 = inputGenerationFunc([replayMemory[i][0] for i in sampleIdx], stateSpaceDim)
    inputSample_1 = inputGenerationFunc([replayMemory[i][3] for i in sampleIdx], stateSpaceDim)
    rwdSample = np.array([replayMemory[i][2] for i in sampleIdx])
    actionSample = np.array([replayMemory[i][1] for i in sampleIdx], dtype = int)
    doneSample = np.array([replayMemory[i][4] for i in sampleIdx], dtype = int)

    optimTarget = model.predict(inputSample_0)
    qVals_1 = target_model.predict(inputSample_1)

    optimTarget[np.arange(0, batchSize, dtype = int), actionSample] \
        = rwdSample + gamma * np.multiply(np.logical_not(doneSample), np.amax(qVals_1, axis = 1))

    history = model.fit(inputSample_0, optimTarget, batch_size = batchSize, epochs = 1, verbose = 0)

    return history
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform, Constant

from config import SEED_VAL

def createModel_deepPolGrads_reinforce(modelParamDict):

    # define input layer
    x = layers.Input(shape = (modelParamDict['inputDim'], ))

    # define hidden layer
    layerInfoList = modelParamDict['layerInfoList']
    y = layers.Dense(layerInfoList[0][0], activation = layerInfoList[0][1], use_bias = True, kernel_initializer = glorot_uniform(seed = SEED_VAL), bias_initializer = Constant(0.001))(x)
    for i in range(1, len(layerInfoList)):
        y = layers.Dense(layerInfoList[i][0], activation = layerInfoList[i][1], use_bias = True, kernel_initializer = glorot_uniform(seed = SEED_VAL), bias_initializer = Constant(0.001))(y)

    # define output layer
    y = layers.Dense(modelParamDict['outputDim'], activation = 'softmax', use_bias = True, kernel_initializer = glorot_uniform(seed = SEED_VAL), bias_initializer = Constant(0.001))(y)

    # create model
    model = keras.Model(inputs = x, outputs = y)

    return model

def compileModel_deepPolGrads_reinforce(model, learningRate = 1e-3):
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = learningRate))
    return model

def act_deepPolGrads(obs, actorModel, stateSpaceDim, controlSpaceDim, inputGenerationFunc):
    p = actorModel.predict(inputGenerationFunc([obs], stateSpaceDim))[0]
    a = np.random.choice(controlSpaceDim, p = p)

    return a

def optimise_deepPolGrads_reinforce(model, obsList, actionList, rewardList, \
                                        controlSpaceDim, gamma = 0.99):
    batchSize = len(obsList)

    # determine discounted rewards
    discntdRwdArr = np.zeros(batchSize)
    for i in range(0, batchSize):
        s = 0.0
        dscnt = 1.0
        for k in range(i, batchSize):
            s += rewardList[k] * dscnt
            dscnt *= gamma
        
        discntdRwdArr[i] = s

    # normalisation
    mu = np.mean(discntdRwdArr)
    tmp = np.std(discntdRwdArr)
    if tmp > 0.0:
        sig = np.std(discntdRwdArr)
    else:
        sig = 1.0    

    optimTarget = np.zeros((batchSize, controlSpaceDim))
    optimTarget[np.arange(0, batchSize, dtype = int), actionList] = 1

    history = model.fit(np.vstack(obsList), optimTarget, sample_weight = (discntdRwdArr - mu) / sig, epochs = 1, verbose = 0)

    return history

def train_deepPolGrads_reinforce(env, inputGenerationFunc, stateSpaceDim, controlSpaceDim, \
                            actorModel, paramDict):
    # ### initialisations

    cumRwdList = []
    lossList = []

    compileModel_deepPolGrads_reinforce(actorModel, paramDict['learningRate'])

    # ### training phase

    print('# ### training phase')

    i = 0
    episodeId = -1
    stopTraining = False

    while not stopTraining:

        obs = env.reset()
        cumRwd = 0.0
        episodeId += 1

        # ### simulation phase
        obsList = []
        actionList = []
        rewardList = []
        for j in range(0, paramDict['maxNumDrawsPerEpisode']):
            # choose action
            a = act_deepPolGrads(obs, actorModel, stateSpaceDim, controlSpaceDim, inputGenerationFunc)
        
            # apply action and make new observation
            newObs, r, done, _ = env.step(a)

            obsList.append(obs)
            actionList.append(a)
            rewardList.append(r)

            # increase draw counter
            i += 1

            obs = newObs.copy()

            if done:
                break

        # ### optimisation phase
        history = optimise_deepPolGrads_reinforce(actorModel, obsList, actionList, rewardList, \
                                                     controlSpaceDim)

        cumRwd = np.sum(rewardList)
        cumRwdList.append(cumRwd)
        lossList.append(history.history['loss'])

        print('# --- episode {}: cum rwd = {:.2f}'.format(episodeId, cumRwd))

        if i > paramDict['numDraws_trng']:
            stopTraining = True

    return cumRwdList, lossList

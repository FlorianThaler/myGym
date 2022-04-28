import numpy as np

from lunarLanding.training_lunarLander import createEnv_lunarLander

def simulateFullEpisode_eval(env, policy, maxNumDraws = 2 ** 9):
    retValDict = {}

    obsList = []
    actionList = []
    rewardList = []
    numDrawsUntilDone = -1

    obs = env.reset()

    for i in range(0, maxNumDraws):
        a = policy(obs)
        newObs, r, done, _ = env.step(a)

        obsList.append(obs)
        actionList.append(a)
        rewardList.append(r)

        obs = newObs.copy()
        if done:
            numDrawsUntilDone = i + 1
            break

    retValDict['obsList'] = obsList
    retValDict['actionList'] = actionList
    retValDict['rewardList'] = rewardList
    retValDict['numDrawsUntilDone'] = numDrawsUntilDone

    return retValDict

def modelEvaluationFunc_lunarLander(policy, numEvalEpisodes = 2 ** 4):
    # create environment
    env, _, _ = createEnv_lunarLander()

    cumRwd = 0.0
    totNumDraws = 0
    for i in range(0, numEvalEpisodes):
        episodeDict = simulateFullEpisode_eval(env, policy)
        cumRwd += np.sum(episodeDict['rewardList'])
        totNumDraws += episodeDict['numDrawsUntilDone']

    return cumRwd / totNumDraws
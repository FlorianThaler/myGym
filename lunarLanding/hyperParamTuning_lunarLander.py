import optuna

from lunarLanding.training_lunarLander import trainingFunc_lunarLander_deepQ
from lunarLanding.evaluation_lunarLander import modelEvaluationFunc_lunarLander

def objectiveFunc_deepQ_learningRate(trial):
    lwrBnd = 1e-4
    upprBnd = 1e-3
    learningRate = trial.suggest_uniform('learningRate', low = lwrBnd, high = upprBnd)

    model, policy, cumRwdList, lossList = trainingFunc_lunarLander_deepQ(lr = learningRate)

    avrgRwdPerDraw = modelEvaluationFunc_lunarLander(policy)

    return avrgRwdPerDraw


def hyperParamTuningFunc_deepQ():

    # ### find optimal learning rate
    study = optuna.create_study(direction = 'maximize')

    study.optimize(objectiveFunc_deepQ_learningRate, n_trials = 5)

    return study
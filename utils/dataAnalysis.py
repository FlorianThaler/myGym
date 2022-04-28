import numpy as np

from matplotlib import pyplot as plt

from optuna.visualization import plot_optimization_history

def processTrainingResults(cumRwdList, lossList):
    """
        this function depicts the training results in terms of the cumulative
        reward and the optimisation losses.

    Args:
        cumRwdList (list): list of cumulative (episodial) rewards
        lossList (list): list of optimisation losses
    """
    # ### compute moving average of cumulative rewards
    n = 2 ** 5
    mvngAvrgList = [np.asarray(cumRwdList[i - n : i]).sum() / n for i in range(n, len(cumRwdList))]

    fig1 = plt.figure(figsize = (15, 15))
    ax = fig1.add_subplot(1, 1, 1)
    ax.plot(np.arange(0, len(cumRwdList)), cumRwdList, label = 'cumRwd')
    ax.plot(np.arange(n, len(cumRwdList)), mvngAvrgList, color = 'r', label = 'mnvgAvrg')
    ax.set_title('evolution of (episodial) cumulative rewards')
    ax.legend()

    fig2 = plt.figure(figsize = (15, 15))
    ax = fig2.add_subplot(1, 1, 1)
    ax.plot(np.arange(0, len(lossList)), lossList)
    ax.set_title('evolution of optimisation loss')

    plt.show()

def processHyperParamTuningResults(study):
    """
        this function depicts the results of the optuna hyperparameter tuning process.

    Args:
        study (Study): instance of optuna class Study
    """
    for key in study.best_params.keys():
        print('# ### maximiser ({}) = {}'.format(key, study.best_params[key]))

    print('# ### maximum of objective function = {:.2f}'.format(study.best_value))

    figure = plot_optimization_history(study)
    figure.show()
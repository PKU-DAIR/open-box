# License: MIT

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from openbox import Observation, get_config_space, get_objective_function

import openml

from openbox.experimental.online.blendsearch import BlendSearchAdvisor
from openbox.optimizer.generic_smbo import SMBO

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

dataset = openml.datasets.get_dataset('autoUniv-au7-500')

Xy, _, classes, names = dataset.get_data(dataset_format='array')

X, y = Xy[:, :-1], Xy[:, -1]
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

space = get_config_space('lightgbm')
function = get_objective_function('lightgbm', x_train, x_val, y_train, y_val)

x0 = space.sample_configuration()

# Run
if __name__ == "__main__":

    MAX_RUNS = 100

    advisors = [
    BlendSearchAdvisor(
        config_space=space,
        task_id='OpenBox'
    )]

    axes = None
    histories = {}

    dim = len(function(x0)['objs'])

    opt = SMBO(
        function,
        space,
        max_runs=MAX_RUNS,
        time_limit_per_trial=10,
    )

    history = opt.run()

    histories['BO'] = history

    for advisor in advisors[2:]:
        print("Now running" + str(advisor.__class__))

        for i in trange(MAX_RUNS):

            config = advisor.get_suggestion()

            ret = function(config)

            observation = Observation(config=config, objs=ret['objs'])
            advisor.update_observation(observation)

            if trange == range:
                print('===== ITER %d/%d.' % (i + 1, MAX_RUNS))

        history = advisor.get_history()
        histories[advisor.__class__.__name__] = history

    for k, v in histories.items():

        if dim == 1:
            axes = v.plot_convergence(ax=axes, name=k)
        elif dim == 2:
            inc = v.get_incumbents()
            inc.sort(key=lambda x: x[1][0])
            plt.plot([x[1][0] for x in inc], [x[1][1] for x in inc])

        print(k)
        print(v)

    plt.title(dataset.name)

    if dim <= 2:
        plt.legend()
        plt.show()

    # if not os.path.exists("tmp"):
    #     os.mkdir("tmp")

    # with open("tmp/" + time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime()) + ".txt", "w") as f:
    #     f.write(str({a: histories[a].get_incumbents() for a in histories}))

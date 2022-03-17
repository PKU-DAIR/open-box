# License: MIT

import itertools
import numpy as np
from scipy.optimize import minimize


def Loss_func(true_y, pred_y, func_id):
    if func_id == 0:
        # Return the L2 loss.
        return 1./(true_y.shape[0])*np.linalg.norm(true_y-pred_y, 2)

    # Compute the rank loss for varied loss function.
    true_y = np.array(true_y)[:, 0]
    pred_y = np.array(pred_y)[:, 0]
    comb = itertools.combinations(range(true_y.shape[0]), 2)
    pairs = list()
    # Compute the pairs.
    for _, (i, j) in enumerate(comb):
        if true_y[i] > true_y[j]:
            pairs.append((i, j))
        elif true_y[i] < true_y[j]:
            pairs.append((j, i))
    loss = 0.
    pair_num = len(pairs)
    if pair_num == 0:
        return 0.
    for (i, j) in pairs:
        if func_id == 1:
            loss += max(pred_y[j] - pred_y[i], 0.)
        elif func_id == 2:
            loss += np.exp(pred_y[j] - pred_y[i])
        elif func_id == 3:
            loss += np.log(1 + np.exp(pred_y[j] - pred_y[i]))
        elif func_id == 4:
            z = 10 * (pred_y[j] - pred_y[i])
            loss += np.log(1 + np.exp(z))
        else:
            raise ValueError('Invalid loss type!')
    return loss/pair_num


def Loss_der(true_y, A, x, func_id):
    y_pred = A * np.mat(x).T
    if func_id == 0:
        # Return the derivative for L2 loss.
        return -2./(A.shape[0])*np.array(A.T*(true_y-y_pred))[:, 0]

    true_y = np.array(true_y)[:, 0]
    pred_y = np.array(y_pred)[:, 0]

    comb = itertools.combinations(range(true_y.shape[0]), 2)
    pairs = list()
    # Compute the pairs.
    for _, (i, j) in enumerate(comb):
        if true_y[i] > true_y[j]:
            pairs.append((i, j))
        elif true_y[i] < true_y[j]:
            pairs.append((j, i))
    # Calculate the derivatives.
    grad = np.zeros(A.shape[1])
    pair_num = len(pairs)
    if pair_num == 0:
        return grad
    for (i, j) in pairs:
        if func_id == 1:
            if pred_y[j] > pred_y[i]:
                grad += (A[j] - A[i]).A1
        elif func_id == 2:
            grad += np.exp(pred_y[j] - pred_y[i]) * (A[j] - A[i]).A1
        elif func_id == 3:
            e_z = np.exp(pred_y[j] - pred_y[i])
            grad += e_z / (1 + e_z) * (A[j] - A[i]).A1
        elif func_id == 4:
            c = 10.
            e_z = np.exp(c * (pred_y[j] - pred_y[i]))
            grad += e_z / (1 + e_z) * (c * (A[j] - A[i])).A1
        else:
            raise ValueError('Invalid func id!')
    return grad/pair_num


def scipy_solve(A, b, loss_type, debug=False):
    n, m = A.shape

    # Add constraints.
    ineq_cons = {'type': 'ineq',
                 'fun': lambda x: np.array(x),
                 'jac': lambda x: np.eye(len(x))}
    eq_cons = {'type': 'eq',
                 'fun': lambda x: np.array([sum(x) - 1]),
                 'jac': lambda x: np.array([1.]*len(x))}

    x0 = np.array([1. / m] * m)

    def f(x):
        w = np.mat(x).T
        return Loss_func(b, A*w, loss_type)

    def f_der(x):
        return Loss_der(b, A, x, loss_type)

    res = minimize(f, x0, method='SLSQP', jac=f_der,
                   constraints=[eq_cons, ineq_cons],
                   options={'ftol': 1e-8, 'disp': False})

    status = False if np.isnan(res.x).any() else True
    if not res.success and status:
        res.x[res.x < 0.] = 0.
        res.x[res.x > 1.] = 1.
        if sum(res.x) > 1.5:
            status = False

    if debug:
        loss = f(res.x)
        print('Ranking loss', loss)
    return res.x, status

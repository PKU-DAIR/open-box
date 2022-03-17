# License: MIT

import datetime
import hashlib
import json

from bson import ObjectId
from django.http import HttpResponse, JsonResponse

from openbox.utils.config_space import json as config_json
from openbox.utils.config_space import Configuration
from openbox.core.base import Observation

from openbox.artifact.data_manipulation.db_object import User, Task, Runhistory
from user_board.utils.common import get_password

# Global mapping from task_id to config advisor
advisor_dict = {}


def task_register(request):
    """
    Receive a task's (task_id, config_space) and dump it into a pickle file.

    Parameters
    ----------
    request: a django.core.handlers.wsgi.WSGIRequest object with following keys:
        'id' : a unique task id
        'config_space_json' : json string of a config_space created by ConfigSpace.read_and_write.json.write()

    Returns
    -------
    (readable information) string in HttpResponse form

    """
    if request.method == 'POST':
        if request.POST:
            # Parse request
            # task_id = request.POST.get('task_id')
            email = request.POST.get('email')
            password = request.POST.get('password')
            user = User().find_one({'email': email})
            if user is None:
                return JsonResponse({'code': 0, 'msg': '[bo_advice/views.py] User does not exist'})
            else:
                if user['password'] != get_password(password):
                    return JsonResponse({'code': 0, 'msg': '[bo_advice/views.py] Incorrect Password'})

            # task_id = request.POST.get('task_id')
            config_space_json = request.POST.get('config_space_json')
            config_space = config_json.read(config_space_json)

            num_constraints = int(request.POST.get('num_constraints', 0))
            num_objs = int(request.POST.get('num_objs', 1))
            options = json.loads(request.POST.get('options', '{}'))

            time_limit_per_trial = int(request.POST.get('time_limit_per_trial', 300))
            active_worker_num = int(request.POST.get('time_limit_per_trial', 1))
            parallel_type = request.POST.get('parallel_type', 'async')
            task_name = request.POST.get('task_name', 'task')
            if task_name == 'task':
                n = Task().find_all().count()
                task_name += '_' + str(n)
            # Create advisor
            advisor_type = request.POST.get('advisor_type', 'default')
            max_runs = request.POST.get('max_runs', 200)

            task_id = str(Task().insert_one({'task_name': task_name,
                                             'owner': str(user['_id']),
                                             'create_time': datetime.datetime.now(),
                                             'config_space': json.loads(config_space_json),
                                             'status': 'running',
                                             'advisor_type': advisor_type,
                                             'max_run': max_runs,
                                             'surrogate_type': options['surrogate_type'],
                                             'time_limit_per_trial': time_limit_per_trial,
                                             'active_worker_num': active_worker_num,
                                             'parallel_type': parallel_type
                                             }))

            if advisor_type == 'default':
                from openbox.core.generic_advisor import Advisor
                config_advisor = Advisor(config_space, num_objs=num_objs, num_constraints=num_constraints,
                                         task_id=task_id, **options)
            elif advisor_type == 'tpe':
                from openbox.core.tpe_advisor import TPE_Advisor
                config_advisor = TPE_Advisor(config_space)
            else:
                raise ValueError('Invalid advisor type!')

            # Save advisor in a global dict
            advisor_dict[task_id] = config_advisor

            return JsonResponse({'code': 1, 'msg': 'SUCCESS', 'task_id': task_id})
        else:
            return JsonResponse({'code': 0, 'msg': 'Empty post data'})
    else:
        return JsonResponse({'code': 0, 'msg': 'Should be a POST request'})


def get_suggestion(request):
    """
    Print the suggestion according to the specified task id.

    Parameters
    ----------
    request: a dict

    Returns
    -------
    SUCCESS : a new encoded (configuration)
    FAILED : a readable information string in HttpResponse form
    """

    if request.method == 'POST':
        if request.POST:
            task_id = request.POST.get('task_id')
            task = Task().find_one({'_id': ObjectId(task_id)})
            if task['status'] == 'stopped':
                return JsonResponse({'code': 0, 'msg': 'The task has stopped' })
            config_advisor = advisor_dict[task_id]

            suggestion = config_advisor.get_suggestion()
            print('-' * 21)
            print('Get suggestion')
            print(suggestion, '-' * 21, sep='')
            res = json.JSONEncoder().encode(suggestion.get_dictionary())

            return JsonResponse({'code': 1, 'res': res})
        else:
            return JsonResponse({'code': 0, 'msg': 'Empty post data'})
    else:
        return JsonResponse({'code': 0, 'msg': 'Should be a POST request'})


def update_observation(request):
    """
    Update observation in config advisor.

    Parameters
    ----------
    request : a dict

    Returns
    -------
    a readable information string in HttpResponse form
    """
    if request.method == 'POST':
        if request.POST:
            task_id = request.POST.get('task_id')
            config_advisor = advisor_dict[task_id]
            config_dict = json.loads(request.POST.get('config'))
            config = Configuration(config_advisor.config_space, config_dict)
            trial_state = int(request.POST.get('trial_state'))
            constraints = json.loads(request.POST.get('constraints'))
            objs = json.loads(request.POST.get('objs'))
            trial_info = json.loads(request.POST.get('trial_info'))
            item = {
                'task_id': task_id,
                'config': config_dict,
                'result': list(objs),
                'status': trial_state,
                'trial_info': trial_info['trial_info'],
                'worker_id': trial_info['worker_id'],
                'cost': trial_info['cost']}
            runhistory_id = Runhistory().insert_one(item)
            observation = Observation(
                config=config, objs=objs, constraints=constraints,
                trial_state=trial_state, elapsed_time=trial_info['cost'],
            )
            config_advisor.update_observation(observation)

            config_advisor.save_history()

            print('-' * 21)
            print('Update observation')
            print(observation)
            print('-' * 21)
            return JsonResponse({'code': 1, 'msg': 'SUCCESS'})
        else:
            return JsonResponse({'code': 0, 'msg': 'Empty post data'})
    else:
        return JsonResponse({'code': 0, 'msg': 'Should be a POST request'})


def get_result(request):
    """
    Get BO result and history.

    Parameters
    ----------
    request: a dict

    Returns
    -------
    BO incumbents, BO history
    """

    if request.method == 'POST':
        if request.POST:
            task_id = request.POST.get('task_id')
            config_advisor = advisor_dict[task_id]

            incumbents = config_advisor.history_container.incumbents
            incumbents = [(k.get_dictionary(), v) for k, v in incumbents]
            history = config_advisor.history_container.data
            history = [(k.get_dictionary(), v) for k, v in history.items()]
            print('-' * 21)
            print('BO result')
            print(incumbents, '-' * 21, sep='')
            res = json.JSONEncoder().encode({'result': json.dumps(incumbents),
                                             'history': json.dumps(history)})

            return HttpResponse(res)
        else:
            return HttpResponse('[bo_advice/views.py] error6')
    else:
        return HttpResponse('[bo_advice/views.py] error7')

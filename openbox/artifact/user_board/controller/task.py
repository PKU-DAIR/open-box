# License: MIT

import sys
import numpy as np

from bson import ObjectId
from django.http import JsonResponse

from openbox.artifact.data_manipulation.db_object import Task, Runhistory


def task_action(request, task_id: str, action: int):
    if request.method == 'GET':
        task = Task().find_one({'_id': ObjectId(task_id)})
        if task is None:
            return JsonResponse({'code': 1, 'msg': 'The task not exist '})

        if action == 1:
            Task().collection.update_one({'_id': ObjectId(task_id)},{'$set':{'status': 'running'}})
            return JsonResponse({'code': 1, 'msg': 'Resume Successfully'})
        elif action == 0:
            Task().collection.update_one({'_id': ObjectId(task_id)},{'$set':{'status': 'stopped'}})
            return JsonResponse({'code': 1, 'msg': 'Stop Successfully'})
        else:

            return JsonResponse({'code': 0, 'msg': 'Param Error'})


def show_task(request, owner: str):
    if request.method == 'POST':
        draw = int(request.POST.get('draw'))
        length = int(request.POST.get('length'))
        start = int(request.POST.get('start'))
        search_value = request.POST.get('search[value]')

        ret = {}

        task = Task()
        filter_ = {'owner': owner}
        ret['recordsFiltered'] = ret['recordsTotal'] = task.collection.find(filter_).count()
        if search_value != '':
            filter_['task_name'] = {'$regex': search_value}
            ret['recordsFiltered'] = task.collection.find(filter_).count()
        tasks = task.collection.find(filter_,
                                     {'task_name': 1, 'config_space': 1, 'create_time': 1, 'status': 1,
                                      'max_run': 1}).sort([('create_time', -1)]).limit(length).skip(start)
        task_ids = []
        task_list = []
        for t in tasks:
            task_ids.append(t['_id'])
            task_list.append(t)
        pipeline = [{'$group': {'_id': {'task_id': "$task_id"},
                                'count': {'$sum': 1}
                                }
                     }]
        run_history_times = {}
        for r in Runhistory().collection.aggregate(pipeline):
            run_history_times[r['_id']['task_id']] = r['count']

        data = []
        for t in task_list:
            config_str = str(t['config_space'])
            if len(config_str) > 45:
                config_str = config_str[1:45]
            if str(t['_id']) not in run_history_times.keys():
                max_run = '0/' + str(t['max_run'])
            else:
                max_run = str(run_history_times[str(t['_id'])]) + '/' + str(t['max_run'])
            data.append(
                [t['task_name'], config_str,
                 t['create_time'].strftime("%Y-%m-%d %H"), t['status'],
                 max_run, str(t['_id']), t['config_space']])

        ret['draw'] = draw
        ret['data'] = data
        return JsonResponse(ret)


def history(request, task_id):
    if request.method == 'GET':
        run_history = Runhistory()
        run_history_list = [x for x in run_history.find_many({'task_id': task_id})]
        table_list = []
        rh_config = {}
        option = {'data': [], 'visualMap': {}}
        perf_list = []

        for rh in run_history_list:
            result = round(rh['result'][0], 4)
            config_str = str(rh['config'])
            if len(config_str) > 35:
                config_str = config_str[1:35]
            else:
                config_str = config_str[1:-1]
            table_list.append(
                [str(rh['_id']), result, config_str, rh['status'], rh['trial_info'], rh['worker_id'], rh['cost']])
            rh_config[str(rh['_id'])] = rh['config']
            data = []
            for parameter in rh['config'].keys():
                data.append(rh['config'][parameter])
            data.append(result)
            option['data'].append(data)
            perf_list.append(result)

        if len(run_history_list) > 0:
            option['schema'] = list(run_history_list[0]['config'].keys()) + ['perf']
            option['visualMap']['min'] = np.percentile(perf_list, 0)
            option['visualMap']['max'] = np.percentile(perf_list, 90)
            option['visualMap']['dimension'] = len(option['schema']) - 1
        else:
            option['visualMap']['min'] = 0
            option['visualMap']['max'] = 100
            option['visualMap']['dimension'] = 0

        line_data = {'min': [], 'over': [], 'scat': []}
        min_value = sys.maxsize
        for idx, perf in enumerate(perf_list):
            if perf <= min_value:
                min_value = perf
                line_data['min'].append([idx, perf])
                line_data['scat'].append([idx, perf])
            else:
                line_data['over'].append([idx, perf])
        line_data['min'].append([len(option['data']), min_value])
        return JsonResponse(
            {'code': 1, 'option': option, 'table_list': table_list, 'rh_config': rh_config, 'line_data': line_data})

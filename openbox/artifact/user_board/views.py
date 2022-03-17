# License: MIT

from bson import ObjectId
from openbox.artifact.data_manipulation.db_object import User, Task, Runhistory

from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.shortcuts import render

from user_board.utils.common import create_token, authenticate_token, get_password


def index(request):
    if request.method == 'GET':
        return render(request, 'login.html')


def logout(request):
    if request.method == 'GET':
        return render(request, 'logout.html')


def register(request):
    if request.method == 'GET':
        return render(request, 'register.html')


def activate(request, token):
    if request.method == "GET":
        acc, payload = authenticate_token(token)
        if acc == 0:
            return HttpResponse(payload)
        else:
            user = User().find_one({'_id': ObjectId(payload['user_id'])})
            if user is None:
                return HttpResponse("User does not exist!")
            if user['is_active'] == 1:
                return HttpResponse("Email is activated!")
            User().collection.update_one({'_id': ObjectId(payload['user_id'])}, {"$set": {'is_active': 1}})
        return render(request, 'login.html', {'is_register': 1})


def reset_password(request, param):
    if request.method == 'GET':
        if param == "send_mail":
            return render(request, 'reset_password.html', {"change_password": 0})
        else:
            return render(request, 'reset_password.html', {"change_password": 1, 'token': param})


def show_task(request, user_id: str):
    if request.method == 'GET':
        context = {}
        context['task_field'] = ['Task Name', 'Configuration', 'Create Time', 'Status', 'Max_run']
        context['user_id'] = user_id
        return render(request, 'task_list.html', context)


def task_detail(request, task_id: str):
    if request.method == 'GET':
        context = {}
        context['task_field'] = ['Advisor Type', 'Surrogate Type', 'Time Limit Per Trial', 'Active Worker Num',
                                 'Parallel Type']
        task = Task().find_one({'_id': ObjectId(task_id)})
        context['task'] = [task['advisor_type'], task['surrogate_type'], task['time_limit_per_trial'],
                           task['active_worker_num'], task['parallel_type'], ]
        context['rh_field'] = ['Result', 'Configuration', 'Status', 'Trial Info', 'Worker Id', 'Cost']
        context['task_id'] = task_id
        return render(request, 'history_list.html', context)

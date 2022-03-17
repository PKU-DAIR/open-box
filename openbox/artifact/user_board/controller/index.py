# License: MIT

import datetime
import sys

from bson import ObjectId
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMessage
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.template.loader import render_to_string

from openbox.artifact.artifact.settings import EMAIL_ACTIVE_ENABLE
from openbox.artifact.data_manipulation.db_object import User, Runhistory
from user_board.utils.common import get_password, create_token, authenticate_token


def login(request):
    if request.method == 'POST':
        if request.POST:
            email = request.POST.get('email')
            password = request.POST.get('password')
            user = User().find_one({'email': email})
            if user is None:
                return JsonResponse({'code': 0, 'msg': 'User not exists!'})
            if user['password'] != get_password(password):
                return JsonResponse({'code': 0, 'msg': 'Incorrect email or password.'})
            elif user['is_active'] == 0 and EMAIL_ACTIVE_ENABLE is True:
                return JsonResponse({'code': 0, 'msg': 'Email not activated!'})
            else:
                request.session["user_email"] = email
                request.session['user_id'] = str(user['_id'])
                return JsonResponse({'code': 1, 'msg': '', 'user_id': str(user['_id'])})
        else:
            return JsonResponse({'code': 0, 'msg': 'Empty Post Data'})


def register(request):
    if request.method == 'POST':
        user_name = request.POST.get('user_name')
        password = request.POST.get('password')
        email = request.POST.get('email')
        user = User().find_one({'email': email})
        if user is not None:
            if user['is_active'] == 1:
                return JsonResponse({'code': 0, 'msg': 'Email already exists!'})
            if (datetime.datetime.now() - user['create_time']).seconds < 60:
                return JsonResponse({'code': 0, 'msg': 'Please check your email to activate your account!'})
        if user is not None and user['is_active'] == 0:
            user_id = str(user['_id'])
        else:
            user_id = str(User().insert_one({'username': user_name,
                                             'email': email,
                                             'password': get_password(password),
                                             'salt': 'example_salt',
                                             'is_active': 0,
                                             'create_time': datetime.datetime.now()}))
        if EMAIL_ACTIVE_ENABLE is False:
            return JsonResponse({'code': 1, 'msg': 'success'})
        account_activation_token = create_token({'user_id': user_id, 'user_is_active': 0}, 60)
        mail_subject = 'Activate your account.'
        current_site = get_current_site(request)
        message = render_to_string('acc_active_email.html', {
            'user_name': user_name,
            'domain': current_site.domain,
            'token': account_activation_token,
            'reset_password': 0
        })
        email = EmailMessage(
            mail_subject, message, to=[email]
        )
        email.send()

        return JsonResponse({'code': 1, 'msg': 'Activate your account by the email!'})


def logout(request):
    request.session.flush()
    return HttpResponseRedirect('/user_board/index/')


def reset_password(request):
    if request.method == 'POST':
        is_send_mail = request.POST.get('is_send_mail')
        if is_send_mail == '1':
            email = request.POST.get('email')
            user = User().find_one({'email': email})

            if user is None:
                return JsonResponse({'code': 0, 'msg': 'Email not exists!'})

            account_activation_token = create_token({'user_id': str(user['_id'])}, 60)
            mail_subject = 'Please reset your password.'
            current_site = get_current_site(request)
            message = render_to_string('acc_active_email.html', {
                'user_name': user['username'],
                'domain': current_site.domain,
                'token': account_activation_token,
                'reset_password': 1
            })
            email = EmailMessage(
                mail_subject, message, to=[email]
            )
            email.send()
            return JsonResponse({'code': 1, 'msg': 'The email send successfully!'})

        else:
            password = request.POST.get('password')
            token = request.POST.get('token')
            if not all([password, token]):
                return JsonResponse({'code': 0, 'msg': 'password or token is None!'})
            acc, payload = authenticate_token(token)
            if acc == 0:
                return HttpResponse(payload)
            else:
                user = User().find_one({'_id': ObjectId(payload['user_id']), 'is_active': 1})
                if user is None:
                    return JsonResponse({'code': 0, 'msg': 'User not exist or Email not activated!'})
                User().collection.update_one({'_id': ObjectId(payload['user_id'])},
                                           {"$set": {'password': get_password(password)}})
            return JsonResponse({'code': 1, "msg": "Reset password Successfully, Please Login!"})

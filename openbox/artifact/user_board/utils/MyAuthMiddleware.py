# License: MIT

import re

from django.http import HttpResponseRedirect
from django.utils.deprecation import MiddlewareMixin

access_path = ['/user_board/index/',
               '/user_board/register/',
               '/user_board/api/login/',
               '/user_board/api/register/',
               '/user_board/reset_password/send_mail/',
               '/user_board/api/reset_password/',
               ]
access_path_re = [
    '/user_board/activate/.*',
    '/user_board/reset_password/.*',
    '/bo_advice/.*',
]


class MyAuthMiddleware(MiddlewareMixin):

    def process_request(self, request):
        is_access_path_re = False
        for p in access_path_re:
            if re.match(p, request.path) is not None:
                is_access_path_re = True
                break
        if request.path not in access_path and is_access_path_re is False:
            if request.session.get('user_id', None):
                pass
            else:
                return HttpResponseRedirect('/user_board/index/')

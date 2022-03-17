# License: MIT

from django.urls import path

from . import views
from user_board.controller import index, task

app_name = 'user_board'
urlpatterns = [
    # /user_board/
    path('', views.index, name='index'),
    path('index/', views.index, name='index'),
    path('logout/', views.logout, name='logout'),
    path('register/', views.register, name='register'),
    path('show_task/<str:user_id>/', views.show_task, name='show_task'),
    path('task_detail/<str:task_id>/', views.task_detail, name='detail'),
    path('activate/<str:token>/', views.activate, name='activate'),
    path('reset_password/<str:param>/', views.reset_password, name='reset_password'),

    # api
    path('api/login/', index.login, name='index'),
    path('api/logout/', index.logout, name='index'),
    path('api/register/', index.register, name='register'),
    path('api/task_action/<str:task_id>/<int:action>/', task.task_action, name='task_action'),
    path('api/history/<str:task_id>/', task.history, name='history'),
    path('api/show_task/<str:owner>/', task.show_task, name='show_task'),
    path('api/reset_password/', index.reset_password, name='reset_password'),

]

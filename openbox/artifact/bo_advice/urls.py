# License: MIT

from django.urls import path

from . import views

app_name = 'bo_advice'
urlpatterns = [
    # ex: /bo_advice/task_register/
    path('task_register/', views.task_register, name='task_register'),
    # ex: /bo_advice/get_suggestion/
    path('get_suggestion/', views.get_suggestion, name='get_suggestion'),
    # ex: /bo_advice/update_observation/
    path('update_observation/', views.update_observation, name='update_observation'),
    # ex: /bo_advice/get_result/
    path('get_result/', views.get_result, name='get_result'),
]

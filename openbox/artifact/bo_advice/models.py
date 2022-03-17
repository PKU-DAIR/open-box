# License: MIT

import datetime
from django.db import models
from django.utils import timezone

class OnlineOptimizer(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    pub_date = models.DateTimeField('date published')

    def __str__(self):
        return self.question_text

    def was_active_today(self):
        return self.pub_date <= timezone.now() - datetime.timedelta(days=1)


class FinishedRun(models.Model):
    host_optimizer = models.ForeignKey(OnlineOptimizer, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    config = models.CharField(max_length=200)
    perf = models.FloatField()

    def __str__(self):
        return self.name + ' : ' + self.config

# License: MIT

from .db_manipulator import DBBase


class User(DBBase):
    def __init__(self, conf_directory: str = './conf'):
        super(User, self).__init__("user", conf_directory)


class Task(DBBase):
    def __init__(self, conf_directory: str = './conf'):
        super(Task, self).__init__("task", conf_directory)


class Runhistory(DBBase):
    def __init__(self, conf_directory: str = './conf'):
        super(Runhistory, self).__init__("runhistory", conf_directory)

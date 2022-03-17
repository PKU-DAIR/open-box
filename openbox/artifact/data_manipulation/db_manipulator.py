# License: MIT

import os
import pymongo
import configparser


class DBManipulator(object):
    def __init__(self, conf_directory: str):
        config_path = os.path.join(conf_directory, 'service.conf')
        config = configparser.ConfigParser()
        config.read(config_path)
        name_server = dict(config.items('database'))
        host = name_server['database_address']
        port = name_server['database_port']
        username = name_server['user']
        password = name_server['password']
        db_url = 'mongodb://' + username + ':' + password + '@%s:%s/' % (host, port)

        # Connect to MongoDB
        self.db_client = pymongo.MongoClient(db_url)
        self.db = self.db_client[username]


class BaseHandle(object):
    @staticmethod
    def insert_one(collection, data):
        res = collection.insert_one(data)
        return res.inserted_id

    @staticmethod
    def insert_many(collection, data_list):
        res = collection.insert_many(data_list)
        return res.inserted_ids

    @staticmethod
    def find_one(collection, data, data_field={}):
        if len(data_field):
            res = collection.find_one(data, data_field)
        else:
            res = collection.find_one(data)
        return res

    @staticmethod
    def find_many(collection, data, data_field={}):
        if len(data_field):
            res = collection.find(data, data_field)
        else:
            res = collection.find(data)
        return res

    @staticmethod
    def update_one(collection, data_condition, data_set):
        res = collection.update_one(data_condition, data_set)
        return res

    @staticmethod
    def update_many(collection, data_condition, data_set):
        """ 修改多条数据 """
        res = collection.update_many(data_condition, data_set)
        return res

    @staticmethod
    def replace_one(collection, data_condition, data_set):
        res = collection.replace_one(data_condition, data_set)
        return res

    @staticmethod
    def delete_many(collection, data):
        res = collection.delete_many(data)
        return res

    @staticmethod
    def delete_one(collection, data):
        res = collection.delete_one(data)
        return res


class DBBase(object):
    def __init__(self, collection, conf_directory: str = './conf'):
        self.manipulator = DBManipulator(conf_directory)
        self.collection = self.manipulator.db[collection]

    def insert_one(self, data):
        res = BaseHandle.insert_one(self.collection, data)
        return res

    def insert_many(self, data_list):
        res = BaseHandle.insert_many(self.collection, data_list)
        return res

    def find_one(self, data, data_field={}):
        res = BaseHandle.find_one(self.collection, data, data_field)
        return res

    def find_many(self, data, data_field={}):
        res = BaseHandle.find_many(self.collection, data, data_field)
        return res

    def find_all(self, data={}, data_field={}):
        """select * from table"""
        res = BaseHandle.find_many(self.collection, data, data_field)
        return res

    def find_in(self, field, item_list, data_field={}):
        """SELECT * FROM inventory WHERE status in ("A", "D")"""
        data = dict()
        data[field] = {"$in": item_list}
        res = BaseHandle.find_many(self.collection, data, data_field)
        return res

    def find_or(self, data_list, data_field={}):
        """db.inventory.find(
           {"$or": [{"status": "A"}, {"qty": {"$lt": 30}}]})

        SELECT * FROM inventory WHERE status = "A" OR qty < 30
        """
        data = dict()
        data["$or"] = data_list
        res = BaseHandle.find_many(self.collection, data, data_field)
        return res

    def find_between(self, field, value1, value2, data_field={}):
        data = dict()
        data[field] = {"$gt": value1, "$lt": value2}
        res = BaseHandle.find_many(self.collection, data, data_field)
        return res

    def find_more(self, field, value, data_field={}):
        data = dict()
        data[field] = {"$gt": value}
        res = BaseHandle.find_many(self.collection, data, data_field)
        return res

    def find_less(self, field, value, data_field={}):
        data = dict()
        data[field] = {"$lt": value}
        res = BaseHandle.find_many(self.collection, data, data_field)
        return res

    def find_like(self, field, value, data_field={}):
        """ where key like "%audio% """
        data = dict()
        data[field] = {'$regex': '.*' + value + '.*'}
        print(data)
        res = BaseHandle.find_many(self.collection, data, data_field)
        return res

    def query_limit(self, query, num):
        res = query.limit(num)
        return res

    def query_count(self, query):
        res = query.count()
        return res

    def query_skip(self, query, num):
        res = query.skip(num)
        return res

    def query_sort(self, query, data):
        res = query.sort(data)
        return res

    def delete_one(self, data):
        res = BaseHandle.delete_one(self.collection, data)
        return res

    def delete_many(self, data):
        res = BaseHandle.delete_many(self.collection, data)
        return res

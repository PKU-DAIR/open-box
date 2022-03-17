# License: MIT

import datetime
import hashlib

import jwt

from openbox.artifact.artifact import settings


def create_token(payload, timeout):
    salt = settings.SECRET_KEY

    headers = {
        "typ": "jwy",
        "alg": "HS256"
    }
    payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(minutes=timeout)
    token = jwt.encode(payload=payload, key=salt, algorithm="HS256", headers=headers)

    return token


def authenticate_token(token):
    salt = settings.SECRET_KEY
    try:
        payload = jwt.decode(token, salt, True)
    except jwt.exceptions.ExpiredSignatureError:
        return 0, 'Expired Error'
    except jwt.DecodeError:
        return 0, 'Decode Error'
    except jwt.InvalidTokenError:
        return 0, 'Invalid Token'
    return 1, payload


def get_password(password):
    md5 = hashlib.md5(settings.SECRET_KEY.encode("utf8"))
    md5.update(password.encode('utf-8'))
    return md5.hexdigest()

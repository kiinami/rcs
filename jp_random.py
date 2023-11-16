"""
random_inference.py
2023-11-02 kiinami
"""

from utils import get_data, write_data
from random import randint


def random():
    data, users = get_data()

    mn = data.min()[1]
    mx = data.max()[1]

    write_data([(user[0], [str(randint(mn, mx)) for _ in range(10)]) for user in users])


if __name__ == '__main__':
    random()

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from openbox import version


def main():
    print(f'openbox v{version}')


if __name__ == "__main__":
    import sys
    sys.exit(main())

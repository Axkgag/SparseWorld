#!/usr/bin/env bash
set -e

ALIYUN_PYPI="http://mirrors.aliyun.com/pypi/simple"
TRUSTED_HOST="mirrors.aliyun.com"

pip config set global.trusted-host "${TRUSTED_HOST}"

pip install -v -e . -i "${ALIYUN_PYPI}"

pip install -r requirements.txt -i "${ALIYUN_PYPI}"

cd mmdet3d/models/sparsedetectors/csrc

python setup.py build_ext --inplace -i "${ALIYUN_PYPI}"

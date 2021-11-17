# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Numpy
#     scikit-learn에서 Numpy 배열은 기본 데이터 구조.
#     scikit-learn은 Numpy 배열 형태의 데이터를 입력 받음, 그래서 우리가 사용할 데이터는 모두 Numpy 배열로 변환 되어야한다.

import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print ('x:\n',x)

# ## SciPy
#     SciPy는 과학 계산용 함수를 모아놓은 파이썬 패키지
#     고성능 선형대수, 함수 최적화, 신호 처리, 특수한 수학 함수의 통계 분포 등을 포함한 기능들을 제공
#     scikit-learn에서 데이터를 표현하는 또 하나의 방법인 희소 행렬 기능을 제공함.
#     sparse matrix(희소 행렬)은 0을 많이 포함한 2차원 배열을 저장할 때 사용

from scipy import sparse 

# +
# 대각선 원소는 1이고 나머지는 0인 2차원 Numpy 배열을 만듭니다.

eye = np.eye(4)
print("Numpy 배열:\n", eye)

# +
# Numpy 배열을 CSR 포맷의 Scipy 희박 행렬로 변환합니다.
# 0이 아닌 원소만 저장됩니다.

sparse_matrix = sparse.csr_matrix(eye)
print('SciPy의 CSR 행렬:\n', sparse_matrix)
# -



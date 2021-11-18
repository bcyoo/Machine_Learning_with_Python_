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
#     display_name: Python 3 (ipykernel)
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

## COO 포맷을 이용하여 희소 행렬 생성
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print('COO 표현:\n', eye_coo)

# ## matplotlib

plt.style.use(['dark_background'])

# +
# %matplotlib inline
import matplotlib.pyplot as plt

## -10에서 10까지 100개의 간격으로 나눠어진 배열 생성
x = np.linspace(-10, 10, 100)

## sin함수를 사용하여 y배열을 생성
y = np.sin(x)

## plot 함수는 한 배열의 값을 다른 배열에 대응해서 선 그래프를 그림

plt.plot(x, y, marker ='x')
# -

# ## pandas
#

# +
import pandas as pd

## 회원 정보가 들어간 data set

data = {'name': ['John', 'Anna', 'Peter', 'Linda'],
        'Location' : ['New York', 'Paris', 'Berlin', 'London'],
        'Age' : [24, 13, 53, 33]    
       }

data_pandas = pd.DataFrame(data)

## display는 주피터 노트북에서 DataFrame을 미려하게 출력.
display(data_pandas)
# +
## Age 열의 값이 30인 이상인 모든 행을 선택

display(data_pandas[data_pandas.Age > 30])
# -


pip install mglearn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# ## 붓꽃의 품종 분류
#     붓꽃의 꽃잎(petal)과 꽃받침(sepal)의 폭고 길이를 센터미터 단위로 측정
#     setosa, versicolor, virginica 종으로 분류한 붓꽃의 측정 데이터로 채집한 붓꽃의 품종을 구분하려함.
#     목표는 어떤 품종이닞 구분해놓은 측정 데이터를 이용해 새로 채집한 붓꽃의 품종을 예측하는 머신러닝 모델을 만드는 것
#     
#     품종읠 정확하게 분류한 데이터를 가지고 있으믈 지도학습에 속한다.
#     이 경우 몇가지 선택사항 중 하나를 선택하는 문제이므로 Classification(분류)에 해당함.
#     출력될 수 있는 값들을 Class라고 함
#     Data Set에 있는 붓꽃 데이터는 모두 세 Class 중 하나에 속함
#     이 예는 세개의 Class를 분류하는 문제
#     
#     데이터 포인트 하나에 대한 기대 출력은 꽃의 품종이 됩니다.
#     이런 특정 데이터 포인트에 대한 출력 즉 품종을 Label이라고 함

# +
## 데이터 적재

from sklearn.datasets import load_iris
iris_dataset = load_iris()

# +
## iris 는 Dictionat 형태로 key 값으로 구성됨

print('iris_dataset의 키:\n', iris_dataset.keys())

# +
## DESCR 에는 데이터셋에대한 간랽한 설명이 있음

print(iris_dataset['DESCR'][:193] + '\n...')

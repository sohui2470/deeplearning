# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# pandas 실습
import pandas as pd

# pd.__version__
# -

# 시리즈 series: 1차원 배열, 값 & 인덱스 부여
sr = pd.Series([17000, 18000, 1000, 5000],
               index = ["피자", "치킨", "콜라", "맥주"])
print(sr)
print(sr.values)
print(sr.index)

# +
# 데이터프레임 dataframe
# 2차원 리스트를 매개변수로 전달: index 행, columns열, value값
a = [[1,2,3],[4,5,6],[7,8,9]]
b = ['one', 'two', 'three']
c = ['A', 'B', 'C']

df = pd.DataFrame(a, index=b,columns= c)
print(df)

print(df.index)
print(df.columns)
print(df.values)

# +
# 리스트로 생성하기
data = [
    ['1000', 'Steve', 90.72],
    ['1001', 'James', 78.09],
    ['1002', 'Doyeon', 98.43],
    ['1003', 'Jane', 64.19]
]
df = pd.DataFrame(data)
print(df)

# 열 지정
dfcol = pd.DataFrame(data, columns=['학번', '이름', '점수'])
print("\n", dfcol)

dfcol.tail(3)
# -

# 딕셔너리로 생성하기
data = {'학번': ['1000', '1001', '1002', '1003'],
       '이름': ['Steve', 'James', 'Doyeon', 'Jane'],
       '점수': [90.72, 78.09, 98.43, 64.19]}
dfdict = pd.DataFrame(data)
(dfdict)

print(dfdict.head(1))
print("\n", dfcol.tail(3))
print("\n", dfdict['이름'])
# 인덱스 출력
print(dfcol.index)

# +
# Numpy 실습
import numpy as np

a = np.array([1,2,3,4,5]) # 리스트를 가지고 1차원 배열 생성
print(type(a))
print(a)

b = np.array([[10, 20, 30], [60, 70, 80]])
print(b)

# 행렬의 차원, 크기
print(a.ndim)
print(a.shape)
print(b.ndim)
print(b.shape)

# +
# ndarray 초기화
a = np.zeros((2,3)) # zeros: 해당 배열에 모두 0 삽입
print(a)

b = np.ones((2,3)) # ones: 모두 1 삽입
print(b)

c = np.eye(5) # 대각선 1, 나머지 0인 2차원 배열(n*n)
print(c)

d = np.full((3,3),7) # 모든 값이 특정 상수인 배열
print(d)

e = np.random.random((4,4)) # 랜덤
print(e)

# +
# np.arange() 범위지정해 배열 생성

a = np.arange(8) # 0부터 n-1까지
print(a)

b = np.arange(1, 16, 5) # 1~16 5간격으로
print(b)
# -

a = np.array(np.arange(30).reshape((5,6))) #배열 생성 후 다차원으로 변형
print(a)

a = np.array([[1, 2, 3,], [4, 5, 6]])
print(a)

# +
# 슬라이스: 각 차원별 범위 지정해야 함
b = a[0:2, 0:2]
print(b)

c = a[:, :1] # 1열 출력
print(c)

d = a[1,:] # 2행 출력
print(d)
# -

# 정수 인덱싱: 웝배열 중에서 부분 배열 구하기
a = np.array([[1,2], [4,5], [7,8]])
b = a[[2,1], [1,0]]
print(b)

# +
# numpy 연산
x = np.array([1,2,3])
y = np.array([4,5,6])

b = x + y
b1 = np.add(x,y)
print(b)
print(b1)

c = x - y
c1 = np.subtract(x,y)
print(c)
print(c1)

d = x * y
d1 = np.multiply(x,y)
print(d)
print(d1)

e = x / y
e1 = np.divide(x,y)
print(e)
print(e1)

f = x % y
f1 = np.remainder(x,y)
print(f)
print(f1)

# +
# numpy 행렬*벡터, 행렬*행렬: np.dot()
a = np.array([[1,2,3],[4,5,6]]) #2*3
b = np.array([[7,8],[9,10],[11,12]]) #3*2
             
c = np.dot(a,b)
print(c)

# +
# Matplotlib 실습

# %matplotlib inline # 주피터노트북에 그림을 표시하도록 지정
import matplotlib.pyplot as plt
# -

plt.title('practice')
plt.plot([1,2,3,4],[2,4,8,6])
plt.xlabel('hours')
plt.ylabel('score')
plt.show() # 주피터노트북에선 자동 시각화되나, 타 개발환경 고려해 show() 삽입

plt.title('students')
plt.plot([1,2,3,4],[2,4,6,6])
plt.plot([1,2,3,4],[1,5,3,7])
plt.xlabel('date')
plt.ylabel('use')
plt.legend(['Amy','Ben']) # 범례 삽입
plt.show()




Machine learning and Deep learning for Data Science 1

# ***Assignment 1***

* submitter : 유지상
* student ID : 2022-29673



- 과제 논의자 : 임상수, 최광호, 박성우, 이다예, 이세라

---

**<u>목차</u>**

[TOC]



## Problem 1

문제에 들어가기 앞서, 해당 Carseat dataset은 categorical variable이 존재한다. 이는 pandas dataframe 상, object로 인식되는데, 0, 1, 2 등의 factor 변수로 변환되어야 추후 모델에 잘 적합할 수 있으므로, 이를 변환하는 과정을 먼저 거쳐야 한다.

- pandas 라이브러리에는 **pd.factorize(Seires)[0]** 이라는 함수가 존재한다. 범주형 변수를 말 그대로 factorize하는 함수다.

  > 그러나 내가 원하는 변수명에 특정 값을 부여하는 과정은 따로 구현할 수 없다는 단점이 있다.
  >
  > 다시 말해, **함수가 자동으로 int값을 각 category에 부여한다.**

  

- 임의로 특정 category에 유저가 원하는 값을 할당하고자 한다면, **DataFrame의 Map 메소드**를 활용할 수 있다.

  > **Series = Series.map(dictionary)**
  >
  > 즉, map method는 새롭게 매핑한 Series를 return한다. 따라서 새로이 return된 Series로 기존 Series를 업데이트 해야 한다.

  > **cf) Series.unique()** : 동 컬럼 중 distinct value를 찾아 출력해준다.



```python
# How to use 'factorize' method in pandas?

ds1['Urban'] = pd.factorize(ds1['Urban'])[0]
ds1['US'] = pd.factorize(ds1['US'])[0]

'''
But we cannot guarantee that 'Yes' would be marked as 1 or 'No' would be marked as 0. It's just arbitrary.
'''
```

```python
# How to use 'map' method in pandas?

print(ds1['ShelveLoc'].unique())
ds1['ShelveLoc'] = ds1['ShelveLoc'].map({'Bad':0, 'Medium':1, 'Good':2})

ds1['Urban'] = ds1['Urban'].map({'Yes':1, 'No':0})
ds1['US'] = ds1['US'].map({'Yes':1, 'No':0})
```

### (a) 

- multiple linear regression 모델은 다음과 같이 적합하였으며, 동 모델의 적합 결과 R^2의 값은 다음과 같이 도출되었다.

```python
# Fitting a multi-linear regression model to the dataset

model = sm.OLS.from_formula('Sales ~ Price + Urban + US', data = ds1)

model_fitted = model.fit()

model_fitted.summary()
```

$$
R^2 = 0.239
$$

> **proof**
>
> <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220403111059262.png" alt="image-20220403111059262" style="zoom:80%;" />



### (b)

- 각 계수에 대해서는 아래의 table을 바탕으로 다음과 같이 해석할 수 있다.

> **proof**
>
> <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220403111221814.png" alt="image-20220403111221814" style="zoom:80%;" />

1) 양적 변수(Quantitative variable)인 Price는 그 계수가 -0.0545의 음수 값이고, |t|가 10.389로 흔히 t에 대해 설정하는 threshold인 2를 훨씬 상회한다. 따라서 동 계수에 대한 p-value는 반올림하여 0에 근접하며, 이는 곧 동 계수가 0이라는 영가설을 상기 모델 적합의 결과, (95% 신뢰수준 하에서, 혹은 그 이상) 기각할 수 있음을 뜻한다. 

   즉, Price는 반응변수(Responsive variable)인 Sales와 음의 관계에 있다고 말할 수 있다. 다시 말해, Price가 증가할수록 Carseat의 Sales는 하락하는 경향을 유의적으로 보인다.

2) 질적 변수(Qualitative variable)인 Urban의 경우, 그 계수가 -0.0219로 음의 값이나, |t| 값이 현저히 낮고, p-value가 1에 근접한다. 즉, 위 모델의 적합 결과, Urban과 Sales 간의 관계는 없다는 영가설 (Null hypothesis)를 신뢰수준 95% (혹은 심지어 90%)에서 기각할 수 없다. 결국, Urban이라는 질적 변수가 yes 이거나 no 라는 여부는 Sales에 영향을 미치지 않는다.

3) 질적 변수인 US의 경우, 그 계수가 1.2006으로 양의 값이고, |t| 값은 4.635로 2를 훨씬 상회한다. 역시 p-value 또한 반올림해 0에 근접하므로, 동 계수가 0이라는 영가설을 기각할 수 있고, US라는 질적 변수가 yes 일 경우, no인 경우에 비해 평균적으로 1.2006 높음을 알 수 있다. (다른 변수는 동일함을 가정할 때)

4) 물론, intercept 또한 그 t값이 2를 훨씬 상회하고, p-value가 극도로 낮아 유의적이라 할 수 있다. (intercept가 0이라는 영가설을 기각할 수 있다) 즉, 다른 모든 변수가 0일 때, 평균적으로 상수 13.0435의 Sales 값을 갖는다고 할 수 있다.



### (c)

- 상기 multi-linear regression model을 방정식의 형태로 기재하면 다음과 같다. (적합 이후의 결과를 기재하겠다.)

$$
Sales = 13.0435 - 0.0219*Urban(binary) + 1.2006*US(binary) - 0.0545*Price
$$

> s.t. Sales and Price are continuous positive (real) values



### (d)

- 앞서 (b)에서 언급한 바와 같이, Urban의 경우, 영가설을 기각할 수 없다. 이와 달리, 그 외의 변수인 US, Price의 경우에는 95% 신뢰수준 하에서 영가설을 기각할 수 있다.
- US의 p-value는 0.000, Price의 p-value도 0.000으로 0.05 이하이기 때문이다.



### (e)

- 상기 모델에 근거하여 다음과 같이 재 적합하였다.
- 즉, 앞서 계수에 대한 영가설을 기각할 수 없었던 Urban 변수는 배제하였고, 나머지 두 설명변수만을 바탕으로 다음과 같이 적합하였다.

```python
# Fitting a 'smaller' multi-linear regression model to the dataset above.

model2 = sm.OLS.from_formula('Sales ~ Price + US', data = ds1)

mode12_fitted = model2.fit()

model2_fitted.summary()
```



### (f)

- (a)에서 적합하였던 모델의 결과와 (e)에서 적합한 모델의 결과는 다음과 같이 요약해볼 수 있다.

> **for (a)**
>
> <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220403112239769.png" alt="image-20220403112239769" style="zoom:80%;" />
>
> <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220403112249846.png" alt="image-20220403112249846" style="zoom:80%;" />

> **for (e)**
>
> <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220403112404916.png" alt="image-20220403112404916" style="zoom:80%;" />
>
> <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220403112412594.png" alt="image-20220403112412594" style="zoom:80%;" />



1) 상기 두 결과의 R^2를 보면, 변수가 하나 줄었음에도 불구하고 그 값은 동일한 것으로 나타난다. (해당 데이터 셋에 대한 유사한 설명력을 보인다는 뜻이다.) 물론 정확하게 동일하다고 볼 수는 없으나, 반올림한 결과, 큰 차이가 나타나지 않았다는 점은 변수 제거 과정에서 배제한 Urban 변수의 Sales에 대한 설명력이 매우 떨어졌음을 반증한다.

2)  다만, Linear regression의 경우, 다른 변수를 모델에 새로이 추가할 때, MSE는 결코 하락하지 않는다. 다시 말해, 어떠한 변수이든지 계속해서 추가 삽입할 때마다, 모델의 설명력은 지속적으로 상승할 수 있다는 것이다. 하지만, 이는 다른 샘플 데이터에 대한 설명력을 약화시키는 결과를 가져오는데, 이는 training dataset에 대한 과적합의 결과다. 따라서 설명변수의 개수에 대한 일종의 페널티를 부여함으로써 이를 어느 정도 극복할 수 있는데, 그 중 하나가 상기 표에 제시되어 있는 adjusted R^2 값이다.

3) 실제로 변수가 3개였던 기존 모델에서는 R^2의 값이 0.234였던 것에 반해, 변수가 2개인 현 모델에서는 그 값이 0.235로 0.001 상승하였다. 이는 변수가 1개 줄어듦에 따른 페널티가 감소했기 때문이라 해석할 수 있으며, 동 측정치에 따르면 변수가 2개인 현 모델이 이전 모델에 보다 낫다(better)라고 말할 수 있다.

4) AIC, BIC 역시 변수의 개수에 대한 penalty를 부여해 조정된 해석을 제공하나, R^2와는 달리 그 자체로 조정된 MSE값을 나타내기 때문에 작아야 더 좋은 모델이라 해석할 수 있다. 이 역시 변수가 줄어든 경우, 외려 더 나은 지표를 보이므로, 이 측정치에 따라도 변수가 2개인 현 모델이 이전 모델에 비해 보다 낫다(better)라고 말할 수 있다. 

   > 하단의 \*Problme 1 추가 분석\* 참고

### (g)

- (e) 모델에서 각 계수에 대한 95%의 신뢰구간은 다음과 같다.

```python
# Source code
# "conf_int" means confidence interval; (1 - confidence)

model2_fitted.conf_int(0.05)
```

> **proof**
>
> <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220403114543856.png" alt="image-20220403114543856" style="zoom:80%;" />





------

## <u>*** Problem 1 : 추가 분석 ***</u>

다음은 문제에서 요구하지는 않았으나, 데이터 셋을 보다 면밀히 파악하기 위해 추가로 수행한 분석이다.

#### Point 1 : Carseat 데이터 셋의 column(feature)은 서로 어떠한 관계를 맺고 있는가?

- Dataset의 전반적인 특성을 알아보기 위해서는, 해당 데이터 셋을 행렬로 간주하여, **선형대수의 행렬 성질을 적용**해보는 과정이 필요하다.
- 대표적으로 column 간의 상관성이 지나치게 높게 나타나는 경우, redundancy가 존재한다고 하는데, 이는 계량경제학에서 흔히 말하는 **다중공선성(multi-collinearity)**을 말한다.
- 다중공선성이 존재함은 곧, 열벡터 간, 선형 종속에 가까운 관계를 맺고 있음을 말하므로, 사실상 **데이터 셋에 대한 설명력은 떨어지나 불필요한 feature가 많아진다**는 것이다.
- 그렇다면 Carseat 데이터 셋은 열 벡터 간에, 즉, feature 간에 어떠한 관계를 갖고 있을까? seaborn의 pairplot을 활용하여 다음과 같은 상관도를 그려볼 수 있겠다.

```python
pairs = sns.PairGrid(ds1)

pairs.map_upper(sns.regplot)
pairs.map_lower(sns.kedplot)
pairs.map_diag(sns.distplot, rug = True)

plt.show()
```

![pairs](C:\Users\user\Desktop\pairs.png)

위 그림에서 다음을 확인해볼 수 있다.

1. **Sales와 Price는 음의 상관성**을 갖고 있다. 따라서 회귀모델을 적합할 때, Price를 설명변수로 추가하는 것이 바람직할 것이다.

2. **Urban값 에 따라 Sales에 유의미한 차이가 관측되지 않는다**.  실제로 kde plot (Urban, Sales)에서 평균값은 거의 근접하게 나타나며, reg plot(Urban, Sales) 역시 거의 수평선을 이루고 있다.

   > 물론, 두 집단의 Sales 평균이 유의미하게 다른지 확인하려면 one-way ANOVA를 사용할 수 있겠지만, 간단한 시각화 과정에서는 위 정도 분석까지만 다루었다.  

3. **US값 에 따라 Sales에 유의미한 차이가 관측된다**. kde plot( US, Sales)에서 평균값은 조금 다르게 나타나며, reg plot에서도 수평선이 아닌, 경사를 그린다. 즉, US에 거주하는 경우, 비교적 Sales는 높게 나타난다.

4. **열벡터 간에도 다중공선성은 관측되지 않는다.**

   1. 다만, Urban의 Sales에 대한 설명력은 낮게 나타날 것으로 기대되기 때문에, 동 변수를 무리해서 모델에 삽입할 경우, **설명력이 낮은 변수의 과적합(overfitting) 문제**로 인해, **다른 dataset에서는 외려 그 설명력이 낮게 도출될 우려**가 있다. 

   3. 또한 adjusted R^2이나 과적합에 대한 penalty를 부여하는 다른 score function을 구해볼 수도 있겠다. 

      > Point 2에서 다시 다룸



#### Point 2 : Urban 설명변수를 모델에 추가함에 따른 pros & cons?

<u>*일반적으로 새로운 설명변수를 모델에 추가할 때, 어떠한 장단점이 있는가?*</u> 

- <u>**pros**</u>

  1. 해당 설명변수가 dataset에 대해 높은 설명력을 갖고 있다면, population의 original model에 근접해지는 셈이 된다.
  2. 즉, 데이터 셋에 대한 전반적인 설명력이 높아지고, 그 결과  현재 주어진 sampled data뿐만 아니라, 다른 여타의 dataset에 대해서도 높은 설명력을 보일 것이다.

- <u>**cons**</u>

  1) 해당 dataset에 대한 과적합(overfitting) 문제가 발생할 우려가 있다. (결과적으로 bias-variance trade off에서 variance가 증가한다)

  2) 다만, 항상 어떠한 변수를 새로이 추가한다고 할 때, 이것이 반드시 과적합으로 이어지는 것이라 보기에는 곤란하다.

  3) pros에 언급한 바와 같이, 동 변수가 population을 잘 설명할 수 있는 변수라면, bias-variance trade off에서 bias가 대폭 감소함에 따라 전반적인 모델의 설명력(예측력)은 더욱 높아지기 때문이다. 

  4) 결국, 새로이 추가하는 변수가 반응변수를 얼마나 잘 설명해낼 수 있을지가 bias-variance 관점에서도 가장 중요한 지표가 되는 셈이다. 

     > 이를 측정하기 위해 CV method나 다양한 scores with penalty를 사용하는 것이다.

​	앞서 adjusted R^2의 값은 Urban 변수를 제거하자 외려 소폭 상승했다. 이는 다시 말하면, Urban 변수가 Sales에 대해 높은 설명력을 보유하지 않았기 때문에, 새로운 변수를 추가한 결과, 주어진 dataset에 대한 과적합 문제만 붉어졌다는 의미라고 해석해볼 수 있겠다.



#### Point 3 : Adjusted R^2 이외에, 다른 score functions를 활용해보면 어떻게 되는가? (AIC, BIC)

- AIC, BIC는 모델 적합의 summary 결과에 같이 제시되어 있다.

  

- <u>**첫번째 모델 (Urban 변수가 빠짐)**의 경우,</u>

  1) **AIC(mallow's Cp)** : 1863
  2) **BIC** : 1879

  

  > 참고로 Gaussian errors를 가정한 linear regression model에서 mallow's Cp와 AIC는 같은 값을 갖는다.

  

- <u>**두번째 모델(Urban 변수가 삽입)**의 경우,</u>

  1) **AIC(mallow's Cp)** : 1861
  2) **BIC** : 1873

  

  ​	두번째 모델의 경우, AIC, BIC가 모두 저하되는 것을 확인할 수 있다.  이때 AIC, BIC는 결국 변수의 개수에 penalty를 부여하여 구한, (조정된 adjusted) MSE다. 즉, adjusted MSE of models fitted to the training set인 셈이다. 따라서 설명력을 나타내는 지표인 adjusted R^2와 달리, 그 값이 낮을수록 동 모델을 better하다고 판단할 수 있는데,  이에 따르면 두 번째 모델을 better하다고 판단할 수 있다.



---

## Problem 2

문제에 들어가기에 앞서, random seed를 설정해야 한다. random choice 기능을 주로 사용할 것이므로 random.seed(2022) 로 사전에 설정하도록 하겠다.

```python
import random
random.seed(2022)
```

### (a)

정해진 패키지를 사용하지 않고, 직접 손으로 step 하나 하나 구현할 것이다.

- **step 0**

  1) 로지스틱 회귀분석을 통해 적합을 할 때, 우선적으로 범주형 변수를 더미화(factorize)해야 한다.

     ```python
     ds2 = pd.read_csv('Default.csv')
     
     ds2['default'] = ds2['default'].map({'Yes':1, 'No':0})
     ds2['student'] = ds2['student'].map({'Yes':1, 'No':0})
     ```

  1) 또한 범주형 변수를 제외한 각 변수의 값 역시 정규화 과정을 거쳐야 올바른 Logistic Regression 분석이 가능하다. 다만, 수많은 정규화 방법 중에서, -1 ~ 1의 값을 갖는 일반적인 normalization 방법을 사용하였다.

     ```python
     for column in ['income', 'balance']:
         ds2[column] = 2*((ds2[column] - ds2[column].min()) / (ds2[column].max() - ds2[column].min())) - 1
     ```

     

- **step 1**

  > sample set를 training set과 validation set으로 구분지어라.

  이때 주의해야 할 점은, Default 데이터 세트가 편향된 자료라는 점이다. 다시 말해, default가 0인 경우가 압도적으로 많아, 층화추출(stratified extraction)을 하지 않으면, 왜곡된 결과가 도출될 수 있다. 따라서 아래의 샘플링 과정은 층화추출 방식을 따를 것이다. (test set은 20%, train set은 80%를 할당할 것이다)

  

  **인덱스 할당하기**

  ```python
  row_num = ds2.shape[0]
  
  index = range(0, row_num)
  
  index_default = list(ds2.loc[ds2['default']==1].index)
  index_nodefault = list(ds2.loc[ds2['default']==0].index)
  
  random.shuffle(index_default)
  random.shuffle(index_nodefault)
  
  index_train = index_default[:int(len(index_default)*0.8)] + index_nodefault[:int(len(index_nodefault)*0.8)]
  index_test = index_default[int(len(index_default)*0.8):] + index_nodefault[int(len(index_nodefault)*0.8):]
  ```

  **인덱스로 데이터 셋을 구분하기**

  ```
  train_set = ds2.iloc[index_train,:]
  test_set = ds2.iloc[index_test,:]
  ```



- **step 2**

  > train_set만을 활용하여 Logistic Regression 적합하기

  LR을 적합하기 위한 패키지는 다양하다. 그 중에서 sklearn.linear_model에 존재하는 LogisticRegression을 활용하여 모델을 적합하였다.

  ```python
  from sklearn.linear_model import LogisticRegression
  logreg = LogisticRegression()
  logreg.fit(train_set[['income', 'balance']], train_set['default'])
  ```

  

- **step 3**

  > 각 행 데이터에 대해서 적합된 Logistic Regression model을 활용하여, default 확률(개념적) 값을 도출하라. 그리고 그 값을 사용하여, 0.5의 threshold를 기준으로 이 보다 높으면 default 1로, 이보다 작거나 같으면 default 0으로 예측하라.

  ```python
  predicted_default = logreg.predict_proba(test_set[['income', 'balance']])
  
  # To determine whether the row data belongs to 1(default) or 0(no default)
  predicted_default_b = []
  for i in range(len(predicted_default)):
      if predicted_default[i][1] > 0.5:
          predicted_default_b.append(1)
      else:
          predicted_default_b.append(0)
  ```

  

- **step 4**

  > validation set error를 계산하라. 단, 이는 전체 관측치로부터 잘못 분류된 케이스의 개수를 세는 것이다.

  ```python
  predicted_default_b = pd.Series(predicted_default_b)
  
  false_counter = 0
  
  for i in range(test_set.shape[0]):
      if test_set['default'].iloc[i] != predicted_default_b.iloc[i]:
          false_counter += 1
  
  print(f'Validation set error is {false_counter/test_set.shape[0]}')
  ```

도출된 최종 값은 다음과 같다. (Validation set error)
$$
0.027486256871564217
$$


### (b)

적합된 모델의 각 계수에 대하여 해석해야 한다. 그러나 sklearn의 경우, 적합된 모델의 coefficient와 그 결과에 대한 summary table을 제시하지 않아, statsmodels.api의 다음 패키지를 사용하여 summary 결과만 가져왔다.

```python
import statsmodels.api as sm

logreg_for_summary = sm.Logit(train_set['default'], sm.add_constant(train_set[['income', 'balance']])).fit()

print(logreg_for_summary.summary())
```

[^**코드작성**]: 여기서 주의할 점은, sm.Logit의 경우 단순히 constant 상수 값은 적합하지 않고, beta(i) (단, i != 0) 값만 적합한다는 것이다. 따라서 상수를 추가하려면 X dataset에 대하여, sm.add_constant(X dataset)을 추가로 해주어야 한다.



> **proof**
>
> ![image-20220406105852862](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220406105852862.png)

1) Intercept는 -3.2089로 음수다. 동 계수에 대한 p-value가 0에 근접할 정도로 극도로 작으므로 intercept가 0이라는 영가설을 기각할 수 있다.
2) Income에 대한 계수는 0.8867의 양의 값을 갖고,  p-value가 0에 근접할 정도로 극도로 작아, 동 값이 0이라는 영가설을 기각할 수 있다. 이 역시, income이 증가할수록 default가 1일 확률에 대한 logit 값이 증가한다는 것을 뜻한다. 즉, income이 증가할수록 default가 yes일 확률이 증가한다.
3) Balance에 대한 계수도 7.2833의 양의 값을 가지며, 동 p-value가 0에 근접할 정도로 극도로 작다. 이 역시 동 값이 0이라는 영가설을 기각할 수 있다. 이것이 갖는 함의는, balance가 증가할수록 default가 1일 확률에 대한 logit 값이 증가한다는 것을 뜻한다. 다시 말해, balance가 증가할수록 default가 yes일 확률이 증가한다는 것이다.
4) 그러나 이와는 별개로, 상기 데이터 셋은 default인 자료가 상대적으로 턱 없이 부족하다. 즉, 편향되어 있다. 따라서 반드시 validation error rate만 조사할 것이 아니라, confusion matrix를 그려 추가로 다양한 지표를 살펴야 올바른 판단을 내릴 수 있다. 다시 말해서, 앞서 도출한 validation set error가 아닌, precision, recall, f1-score도 같이 구해보아야 한다.



### (c)

CV method를 직접 구현할 것이다. 5-fold CV method가 될 것이며, 총 5번의 검증을 거쳐 이를 평균한 값을 구하는 것이 목표다.

- 다만, 앞서 Validation set error만을 구하는 것은 편향되어 있는 dataset의 특성을 잘 반영할 수 없다고 언급했다.
- 따라서 매번 반복 실행을 통해 도출되는 예측값을 바탕으로 Validation set error 뿐만 아니라, precision, recall, f1-score를 모두 구해볼 것이다.

```python
index_default = list(ds2.loc[ds2['default']==1].index)
index_nodefault = list(ds2.loc[ds2['default']==0].index)

random.shuffle(index_default)
random.shuffle(index_nodefault)

interval_default = int(len(index_default)*0.2)
interval_nodefault = int(len(index_nodefault)*0.2)

Validation_set_error = 0
precision = 0
recall = 0
f1_score = 0

for i in range(5):
    
    # Constructing each train & test set pair
    if i != 4:
        index_test = index_default[interval_default*i:interval_default*(i+1)] + index_nodefault[interval_nodefault*i:interval_nodefault*(i+1)]
    else:
        index_test = index_default[interval_default*i:] + index_nodefault[interval_nodefault*i:]
    index_train = list(set(range(10000)) - set(index_test))
    
    # Dividing the whole dataset into two subsets; train & test
    train_set = ds2.iloc[index_train,:]
    test_set = ds2.iloc[index_test,:]
    
    # Fitting each Logistic Regression model with sklearn for one epoch
    logreg = LogisticRegression()
    logreg.fit(train_set[['income', 'balance']], train_set['default'])
    
    # Making a prediction about whether each row data would be default or not by probability
    predicted_default = logreg.predict_proba(test_set[['income', 'balance']])

    # To determine whether the row data belongs to 1(default) or 0(no default)
    predicted_default_b = []
    for a in range(len(predicted_default)):
        if predicted_default[a][1] > 0.5:
            predicted_default_b.append(1)
        else:
            predicted_default_b.append(0)
            
    # Calculating each statistic
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    predicted_default_b = pd.Series(predicted_default_b)
    for b in range(test_set.shape[0]):
        if test_set['default'].iloc[b] == 1:
            if predicted_default_b.iloc[b] == 1:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if predicted_default_b.iloc[b] == 1:
                false_positive += 1
            else:
                true_negative += 1        
    
    # For validation set error
    Validation_set_error += 1-((true_positive + true_negative)/test_set.shape[0])
    
    # For Precision
    precision += true_positive/(true_positive + false_positive)
    
    # For Recall
    recall += true_positive/(true_positive + false_negative)
    
    # For f1-score
    f1_score += 2*((true_positive/(true_positive + false_positive)) * (true_positive/(true_positive + false_negative))) / ((true_positive/(true_positive + false_positive)) + (true_positive/(true_positive + false_negative)))
    
    print(f'fold {i+1}')
    print(f'\tVSE = {1-((true_positive + true_negative)/test_set.shape[0])}\tPRE = {true_positive/(true_positive + false_positive)}\tREC = {true_positive/(true_positive + false_negative)}\tf1-s = {2*(precision * recall) / (precision + recall)}')

print(f'Average Validation set error : {Validation_set_error / 5}\nAverage precision : {precision/5}\nAverage recall : {recall/5}\nAverage f1-score : {f1_score/5}')
```

도출 결과는 다음과 같다.

> **fold 1** 
>
> ==VSE === 0.02701350675337666	==PRE === 0.7307692307692307	==REC === 0.2878787878787879	==f1-s== = 0.41304347826086957 
>
> 
>
> **fold 2** 
>
> ==VSE === 0.027513756878439266	==PRE === 0.7391304347826086	==REC === 0.25757575757575757	==f1-s== = 0.7956551255940257 
>
> 
>
> **fold 3** 
>
> ==VSE === 0.026013006503251668	==PRE === 0.7692307692307693	==REC === 0.30303030303030304	==f1-s === 1.2306379347130358 
>
> 
>
> **fold 4** 
>
> ==VSE === 0.026513256628314164	==PRE === 0.8095238095238095	==REC === 0.25757575757575757	==f1-s== = 1.623214339645435 
>
> 
>
> **fold 5** 
>
> ==VSE === 0.027944111776447067	==PRE === 0.7241379310344828	==REC === 0.30434782608695654	==f1-s== = 2.053240188756478 
>
> 
>
> Average Validation set error : **0.026999527707965765** 
>
> Average precision : **0.7545584350681801** 
>
> Average recall : **0.2820816864295125** 
>
> Average f1-score : **0.4098449170278425**



각각의 fold에 대해서 위의 4가지 statistics를 계산해내었다. 모든 fold에 대해서 검증을 마친 뒤, 도출된 statistics를 평균하여 값을 도출해 본 결과, 위와 같은 결과를 얻을 수 있었다.

- validation set error 값은 약 0.027으로 도출되었다.
- precision은 0.7546, recall은 0.2821, f1-score는 0.41 이다.

앞서 (a)에서 도출한 validation set error는 약 0.02749이 도출되었는데, 이는 (e)에서 도출한 값보다 다소 크다. 그러나 이는 동 데이터 세트를 어떻게 랜덤 추출하여 5개의 sub set을 구성하는가에 따라 지속적으로 바뀌는 값이므로, 이에 대해 큰 의미를 부여하기 어렵다.

다만, validation set error는 subset의 랜덤 추출에 따른 하나의 RV로서 분포(distribution)를 형성할 것이다. (동 분포에 대해서는 (d)에서 추가로 언급하겠다) 물론, 여러 번 반복 추출하여 얻은 결과를 평균한 값이므로, 앞서 도출한 1회 검증에 따른 validation error rate 0.02749보다 stable한 값이다. (동 RV에 대한 distribution의 mean에 비교적 더 근접할 확률이 높다)



### (d)

다음은 **<u>CV에 대한 customized function을 직접 짠 것</u>**이다. 다른 데이터 셋에도 보편적으로 활용할 수 있지만, 반드시 Logistic Regression이어야 하며, target_column은 binary형태여야 한다. 물론 데이터 셋을 먼저 factorize하여 사용해야 한다.

```python
# Defining a customized CV function
# You should input already factorized dataset
# This CV method uses 'Stratified sampling' as a deafult sampling method

def binary_CV_Logistic_print(target_column:str, predictive_column:list, k:int, data):
  import pandas as pd
  import random
  from sklearn.linear_model import LogisticRegression

  index_1 = list(data.loc[data[target_column]==1].index)
  index_0 = list(data.loc[data[target_column]==0].index)

  random.shuffle(index_1)
  random.shuffle(index_0)

  interval_1 = int(len(index_1)/k)
  interval_0 = int(len(index_0)/k)

  Validation_set_error = 0
  precision = 0
  recall = 0
  f1_score = 0

  for i in range(k):
      # Constructing each train & test set pair
      if i != k-1:
          index_test = index_1[interval_1*i:interval_1*(i+1)] + index_0[interval_0*i:interval_0*(i+1)]
      else:
          index_test = index_1[interval_1*i:] + index_0[interval_0*i:]
      index_train = list(set(range(data.shape[0])) - set(index_test))
      
      # Dividing the whole dataset into two subsets; train & test
      train_set = data.iloc[index_train,:]
      test_set = data.iloc[index_test,:]
      
      # Fitting each Logistic Regression model with sklearn for one epoch
      logreg = LogisticRegression()
      logreg.fit(train_set[predictive_column], train_set[target_column])
      
      # Making a prediction about whether each row data would be default or not by probability
      predicted_default = logreg.predict_proba(test_set[predictive_column])

      # To determine whether the row data belongs to 1(default) or 0(no default)
      predicted_default_b = []
      for a in range(len(predicted_default)):
          if predicted_default[a][1] > 0.5:
              predicted_default_b.append(1)
          else:
              predicted_default_b.append(0)
              
      # Calculating each statistic
      true_positive = 0
      true_negative = 0
      false_positive = 0
      false_negative = 0
      
      predicted_default_b = pd.Series(predicted_default_b)
      for b in range(test_set.shape[0]):
          if test_set[target_column].iloc[b] == 1:
              if predicted_default_b.iloc[b] == 1:
                  true_positive += 1
              else:
                  false_negative += 1
          else:
              if predicted_default_b.iloc[b] == 1:
                  false_positive += 1
              else:
                  true_negative += 1        
      
      # For validation set error
      Validation_set_error += 1-((true_positive + true_negative)/test_set.shape[0])
      
      # For Precision
      precision += true_positive/(true_positive + false_positive)
      
      # For Recall
      recall += true_positive/(true_positive + false_negative)
      
      # For f1-score
      f1_score += 2*((true_positive/(true_positive + false_positive)) * (true_positive/(true_positive + false_negative))) / ((true_positive/(true_positive + false_positive)) + (true_positive/(true_positive + false_negative)))
      
      print(f'fold {i+1}')
      print(f'\tVSE = {1-((true_positive + true_negative)/test_set.shape[0])}\tPRE = {true_positive/(true_positive + false_positive)}\tREC = {true_positive/(true_positive + false_negative)}\tf1-s = {2*(precision * recall) / (precision + recall)}')

  print(f'Average Validation set error : {Validation_set_error / k}\nAverage precision : {precision/k}\nAverage recall : {recall/k}\nAverage f1-score : {f1_score/k}')
  return {'Validation_set_error':Validation_set_error / k, 'precision':precision/k, 'recall':recall/k, 'f1_score':f1_score/k}
```



위 코드가 적절히 구동하는지 확인하기 위해 다음의 테스트를 수행해보았다.

```python
# Testing the customized CV function by k = 5 for 2-variant model

binary_CV_Logistic_print(target_column = 'default', predictive_column = ['income','balance'], k = 5, data = ds2)
```

> ![image-20220406110830345](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220406110830345.png)

```python
# Testing the customized CV function by k = 5 for 3-variant model

binary_CV_Logistic_print(target_column = 'default', predictive_column = ['income','balance','student'], k = 5, data = ds2)
```

> ![image-20220406110908039](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220406110908039.png)

```python
# Testing the customized CV function by k = 8 for 2-variant model

binary_CV_Logistic_print(target_column = 'default', predictive_column = ['income','balance'], k = 8, data = ds2)
```

> ![image-20220406110933311](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220406110933311.png)

```python
# Testing the customized CV function by k = 8 for 3-variant model

binary_CV_Logistic_print(target_column = 'default', predictive_column = ['income','balance','student'], k = 8, data = ds2)
```

> ![image-20220406111008219](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220406111008219.png)



**<u>다시 문제로 돌아오면,</u>**

- 설명변수에 student를 새로이 추가해 5-fold CV를 실행해보겠다.

```python
binary_CV_Logistic_print(target_column = 'default', predictive_column = ['income','balance','student'], k = 5, data = ds2)
```

- 위 코드의 실행 결과 다음과 같은 결과가 도출되었다.

![image-20220406111147977](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220406111147977.png)

- validation set error 값은 약 0.0269으로 도출되었다.
- precision은 0.7631, recall은 0.2793, f1-score는 0.4076 이다.



> - **여기서 변수 3개 model(student 포함)의 Validation set error가 0.033 근방으로 도출되는 경우가 있다. **
> - **그러나 이는 연속형 변수인 income과 balance를 normalize하지 않아 생긴 결과이다.** 
> - 연속형 변수 normalization을 거치게 되면, 위와 유사한 결과가 도출된다. (추후 Random Variable로서 다룸)



 앞선 결과와 비교하여 정리하면, 다음과 같다.

|                      | 변수 2개 model | 변수 3개 model |
| -------------------- | -------------- | -------------- |
| Validation set error | 0.027          | 0.0269         |
| Precision            | 0.7546         | 0.7631         |
| Recall               | 0.2821         | 0.2793         |
| F1-score             | 0.4098         | 0.4076         |



결과를 보면, 검증치의 종류에 따라 소폭 상승하거나 하락하는 경향을 보인다. Validation set error는 소폭 하락,  Precision으 상승, Recall, F1-score는 모두 하락하였다. 그러나 위의 검증치 역시 CV-fold를 수행할 때, 어떻게 fold를 나누는가에 따라 랜덤하게 결정되는 값이므로 위의 일회성 검증만으로 결과를 일반화하기 매우 어렵다. 따라서 위와 같은 CV 검증을 각 모델마다 총 500번 반복 시행하여 4개의 검증치에 대한 distribution을 얻어 이를 그려보았다. 또한 distribution을 얻었으므로, 각 statistics의 mean이 모델별로 얼마나 다르며, 그 차이가 유의한지 여부 또한 one-way ANOVA로 살펴볼 수 있다.



앞서 정의한 customized k-fold CV 함수를 사용했으나, 반복 수행 시, 매번 중간 결과값이 print 되는 것을 방지하고자 **<u>다음과 같이 재정의</u>**하였다. 

```python
# I just redefined the customized CV function in order to make an appropirate function for repeats
# It actually does not print the results

def binary_CV_Logistic(target_column:str, predictive_column:list, k:int, data):
  import pandas as pd
  import random
  from sklearn.linear_model import LogisticRegression

  index_1 = list(data.loc[data[target_column]==1].index)
  index_0 = list(data.loc[data[target_column]==0].index)

  random.shuffle(index_1)
  random.shuffle(index_0)

  interval_1 = int(len(index_1)/k)
  interval_0 = int(len(index_0)/k)

  Validation_set_error = 0
  precision = 0
  recall = 0
  f1_score = 0

  for i in range(k):
      # Constructing each train & test set pair
      if i != k-1:
          index_test = index_1[interval_1*i:interval_1*(i+1)] + index_0[interval_0*i:interval_0*(i+1)]
      else:
          index_test = index_1[interval_1*i:] + index_0[interval_0*i:]
      index_train = list(set(range(data.shape[0])) - set(index_test))
      
      # Dividing the whole dataset into two subsets; train & test
      train_set = data.iloc[index_train,:]
      test_set = data.iloc[index_test,:]
      
      # Fitting each Logistic Regression model with sklearn for one epoch
      logreg = LogisticRegression()
      logreg.fit(train_set[predictive_column], train_set[target_column])
      
      # Making a prediction about whether each row data would be default or not by probability
      predicted_default = logreg.predict_proba(test_set[predictive_column])

      # To determine whether the row data belongs to 1(default) or 0(no default)
      predicted_default_b = []
      for a in range(len(predicted_default)):
          if predicted_default[a][1] > 0.5:
              predicted_default_b.append(1)
          else:
              predicted_default_b.append(0)
              
      # Calculating each statistic
      true_positive = 0
      true_negative = 0
      false_positive = 0
      false_negative = 0
      
      predicted_default_b = pd.Series(predicted_default_b)
      for b in range(test_set.shape[0]):
          if test_set[target_column].iloc[b] == 1:
              if predicted_default_b.iloc[b] == 1:
                  true_positive += 1
              else:
                  false_negative += 1
          else:
              if predicted_default_b.iloc[b] == 1:
                  false_positive += 1
              else:
                  true_negative += 1        
      
      # For validation set error
      Validation_set_error += 1-((true_positive + true_negative)/test_set.shape[0])
      
      # For Precision
      precision += true_positive/(true_positive + false_positive)
      
      # For Recall
      recall += true_positive/(true_positive + false_negative)
      
      # For f1-score
      f1_score += 2*((true_positive/(true_positive + false_positive)) * (true_positive/(true_positive + false_negative))) / ((true_positive/(true_positive + false_positive)) + (true_positive/(true_positive + false_negative)))
      
  return {'Validation_set_error':Validation_set_error / k, 'precision':precision/k, 'recall':recall/k, 'f1_score':f1_score/k}
```



다음은 <u>**CV 검증을 모델마다 총 500번 수행하는 코드**</u>다. 

```python
Validation_set_error_2 = []
precision_2 = []
recall_2 = []
f1_score_2 = []

Validation_set_error_3 = []
precision_3 = []
recall_3 = []
f1_score_3 = []

for repeat in range(500):
    
    # CV method : 2-variant model
    result = binary_CV_Logistic(target_column = 'default', predictive_column = ['income','balance'], k = 5, data = ds2)
    Validation_set_error_2.append(result['Validation_set_error'])
    precision_2.append(result['precision'])
    recall_2.append(result['recall'])
    f1_score_2.append(result['f1_score'])

    
    # CV method : 3-variant model
    result = binary_CV_Logistic(target_column = 'default', predictive_column = ['income','balance','student'], k = 5, data = ds2)
    Validation_set_error_3.append(result['Validation_set_error'])
    precision_3.append(result['precision'])
    recall_3.append(result['recall'])
    f1_score_3.append(result['f1_score'])
```



위 수행을 통해 얻은 결과를 바탕으로, <u>총 4가지의 검증치에 대한 distplot을 모델 간 비교</u>하여 그려보면 다음과 같다. 추가로 one-way ANOVA 역시 같이 적합하였다.

- **Validation set error**

  ```python
  sns.distplot(Validation_set_error_2, label = 'Validation_set_error_2')
  sns.distplot(Validation_set_error_3, label = 'Validation_set_error_3')
  plt.legend()
  plt.show()
  
  print(stats.f_oneway(Validation_set_error_2, Validation_set_error_3))
  ```

  >![image-20220406111951800](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220406111951800.png)



- **Precision**

  ```python
  import scipy.stats as stats
  
  sns.distplot(precision_2, label = 'precision_2')
  sns.distplot(precision_3, label = 'precision_3')
  plt.legend()
  plt.show()
  
  print(stats.f_oneway(precision_2, precision_3))
  ```

  > ![image-20220406112038751](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220406112038751.png)



* **Recall**

  ```python
  sns.distplot(recall_2, label = 'recall_2')
  sns.distplot(recall_3, label = 'recall_3')
  plt.legend()
  plt.show()
  
  print(stats.f_oneway(recall_2, recall_3))
  ```

  > ![image-20220406112139053](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220406112139053.png)



- **F1-score**

  ```python
  sns.distplot(f1_score_2, label = 'f1_score_2')
  sns.distplot(f1_score_3, label = 'f1_score_3')
  plt.legend()
  plt.show()
  
  print(stats.f_oneway(f1_score_2, f1_score_3))
  ```

  > ![image-20220406112223869](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220406112223869.png)



 상기 결과를 종합해보면, 이제 다음과 같은 결론을 명확히 내릴 수 있다.

- **Validation error** :  두 모델의 차이는 없다. one-way ANOVA 결과, p-value가 0.59이므로 영가설을 기각할 수 없다. (신뢰수준 95% 하에서)

  > Validation_set_error_2 mean is  0.02691035547713977 
  >
  > Validation_set_error_3 mean is  0.02691702318224979

- **Precision** : 두 모델 간 차이가 있다. student를 포함한 3변수 모델에서 precision의 평균이 약 0.0063 정도 높게 도출되었고, p-value가 0에 매우 근접하여 영가설을 기각할 수 있다.

  > precision_2 mean is  0.7553568631540901 
  >
  > precision_3 mean is  0.7617068094787046

- **Recall** : 두 모델 간 차이가 있다. 2변수 모델에서 recall의 평균이 약 0.0048 정도 높게 도출되었고, 이 역시 p-value가 0에 매우 근접하여 영가설을 기각할 수 있다.

  > recall_2 mean is  0.28620342555994743 
  >
  > recall_3 mean is  0.28142081686429526

- **F1-score** : 두 모델 간 차이가 있다. 2변수 모델에서 f1-score의 평균이 약 0.0042 정도 높게 도출되었고, 이 또한 p-value가 0에 매우 근접하여 영가설을 기각할 수 있다.

  > f1_score_2 mean is  0.4127000361586952
  >
  > f1_score_3 mean is  0.4084563463565717



 결국, student를 포함하여 LogisticRegression을 적합하게 되면, 전반적인 accuracy를 나타내는 Validation Set Error는 사실상 차이가 없으나, Precision은 상승, Recall은 하락, F1-score도 하락한다고 결론내릴 수 있다. 동 데이터 셋은 default가 1인 경우에 편향되어 있기 때문에 전반적 accuracy만을 고려해 어느 모델이 더 낫다고 결론내릴 수 없으므로, 나머지 지표인 precision, recall, f1-score를 같이 고려해보아야 한다.

 f1-score는 precision, recall의 조화평균으로서 그 값이 1에 가까우면 좋은 지표다. 따라서 변수가 3개인, 즉, student를 포함한 모델은 f1-score에서 더 낮은 값이 도출되었으므로 검증치 전반을 고려해 보았을 때, 변수가 2개인 모델보다 worse하다.



 왜 이런 결과가 발생한 것일까? 그 원인은 새로운 변수를 추가함에 따른 trade-off 관계에 근거한다. 

- 새로운 변수를 추가하여 모델을 적합하게 될 때, MSE는 하락하지 않지만, 이는 과적합(overfitting)의 위험을 가져온다. 
- 물론, 새로운 변수가 동 데이터 셋에 대한 높은 설명력을 보인다면, 이는 bias-variance trade-off 관계에서 bias의 큰 하락을 가져올 수 있다.
- 하지만 동 변수가 다른 컬럼에 대해 redundant 하거나 (이는 곧, multi-collinearity가 존재함을 의미한다),  애당초 설명변수와 아무런 관련성이 없다면 다른 샘플 데이터 셋에 대한 variance가 높아지는 결과를 초래할 수 있다.



 그렇다면 student라는 변수는 결국, 다른 컬럼과의 상관성이 높거나, default에 대한 설명력이 매우 낮을 것이라 기대할 수 있다. 실제로 이를 확인하기 위해, pairplot을 그려보면, 다음과 같다.

```
pairs = sns.PairGrid(ds2)

pairs.map_upper(sns.regplot)
pairs.map_lower(sns.kedplot)
pairs.map_diag(sns.distplot, rug = True)

plt.show()
```

![pairs](C:\Users\user\Downloads\pairs.png)

 위 그림을 보면, 

- student와 income 간 음의 상관성이 다소 짙게 나타난다. 

- student와 default 여부를 보면, student인지의 여부에 따라 default값이 크게 달라진다고 보기도 어렵다. 

  > student라는 변수 자체만으로 default에 대해 크게 설명할 수 없을 뿐더러, income이라는 변수에 대한 다소 redundant한 특성을 보이기 때문에, default 여부를 예측하는 Logistic Regression model에서 위와 CV 검증 결과를 보인 것이다.

상관계수 행렬을 통해서 위의 그림을 숫자로 재확인해볼 수 있다.

```python
ds2.corr()
```

| index   | default               | student              | balance               | income                |
| ------- | --------------------- | -------------------- | --------------------- | --------------------- |
| default | 1\.0                  | 0\.03542047403799172 | 0\.35011924704169506  | -0\.01987144765679436 |
| student | 0\.03542047403799172  | 1\.0                 | 0\.20357773837952942  | -0\.7539853368616161  |
| balance | 0\.35011924704169506  | 0\.20357773837952942 | 1\.0                  | -0\.15224340058606753 |
| income  | -0\.01987144765679436 | -0\.7539853368616161 | -0\.15224340058606753 | 1\.0                  |

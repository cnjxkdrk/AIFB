# 라이브러리 설치
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# !pip install seaborn
import seaborn as sns

# 데이터 로딩
df = pd.read_csv('데이터 경로.csv')

# 데이터 구성 확인
df.describe()
df.head()
df.tail()
df.info()

# 시각화 - 상관관계 분석
corr = df.corr()
sns.heatmap(corr, annot=True)

# 필요없는 데이터 제거/ 결측치(null) 제거 또는 채우기

# replace : 다른 값으로 대체
df['Total'].replace([' '], ['0'], inplace=True)
df['age_itg_cd'].replace(np.nan, 48, inplace=True)

# drop : 지우기     axis=1 : 열, axis=0 : 행
df.drop('CustomerID', axis=1, inplace=True)
df.drop(columns=['new_date', 'opn_nfl_chg_date'], inplace=True)

# 데이터타입 변경
df['Total'] = df['Total'].astype(float)

# 결측치 개수
df.isnull().sum()
df['Total'].isnull().sum()

# 데이터 전처리

# X(Feature)와 Y(Label) 지정
X = df.drop('Total', axis=1).values
Y = df['Total'].values

# 7:3으로 Data 분리(Train:Test)
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, 
    stratify=Y, random_state=42)

# AI 모델링
# 로지스틱 회귀분석 모델링
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
rf.fit(X_train, Y_train)    # 학습
predicted = rf.predict(X_valid)
accuracy_score(Y_valid, predicted)

# 딥러닝 모델링
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
epochs=50
batch_size=10

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(29.)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid')) # sigmoid : 이진분류, softmax : 다중분류

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) # metrics : 평가기준
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

history = model.fit(x=X_train, y=Y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_valid, Y_valid),
                callbacks=[es, mc],
                verbose=1)


# 정확도(acc, loss) 그래프 그리기
history = model.fit(X_train, Y_train, validation_split=0.25, epochs=10, verbose=1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['acc', 'val_acc', 'loss', 'val_loss']) # 범례
plt.show()

# 모델 성능평가
np.mean((y_pred - y_test) ** 2) ** 0.5  # RMSE


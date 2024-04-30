import matplotlib.pyplot as plt
import numpy as np

# 시그모이드 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 시그모이드 도함수
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 순전파 함수
def feed_forward(x, w1, w2, b1, b2):
    a1 = x
    z2 = np.dot(w1, a1) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(w2, a2) + b2
    a3 = sigmoid(z3)
    return a1, a2, a3, z2, z3

# 노드 크기 설정
node_size = {'input_layer_size': 3, 'hidden_layer_size': 3, 'output_layer_size': 1}

# 가중치와 바이어스 초기화
w1 = np.random.random((node_size['hidden_layer_size'], node_size['input_layer_size']))
w2 = np.random.random((node_size['output_layer_size'], node_size['hidden_layer_size']))
b1 = np.random.random(node_size['hidden_layer_size'])
b2 = np.random.random(node_size['output_layer_size'])

# 학습률
learning_rate = 2.0

# 입력 데이터 X와 목표 데이터 Y
X = np.array([[1,0,0], [0,0,1], [0,1,1], [1,0,1], [1,1,0], [0,1,0], [1,1,1]])
Y = np.array([1, 0, 0, 0, 1, 1, 0])

count = 0
# 최대 반복 횟수
max_iteration = 1000

# 데이터셋 크기
dataset_size = len(Y)

list_average_cost = []
# 반복 학습 시작
while count < max_iteration:
    dw1 = np.zeros((node_size['hidden_layer_size'], node_size['input_layer_size']))
    dw2 = np.zeros((node_size['output_layer_size'], node_size['hidden_layer_size']))
    db1 = np.zeros((node_size['hidden_layer_size']))
    db2 = np.zeros((node_size['output_layer_size']))

    average_cost = 0
    # 모든 데이터에 대해 순전파 및 역전파 실행
    for x, y in zip(X, Y):
        a1, a2, a3, z2, z3 = feed_forward(x, w1, w2, b1, b2)
        
        # 역전파 단계
        delta3 = -(y - a3) * sigmoid_derivative(z3)
        average_cost += np.linalg.norm((y - a3), 2)/dataset_size
        delta2 = np.dot(w2.T, delta3) * sigmoid_derivative(z2)

        dw2 += np.dot(delta3[:, np.newaxis], np.transpose(a2[:, np.newaxis]))/dataset_size
        db2 += delta3/dataset_size
        dw1 += np.dot(delta2[:, np.newaxis], np.transpose(a1[:, np.newaxis]))/dataset_size
        db1 += delta2/dataset_size
       
        # 가중치 업데이트
        # 바이어스 업데이트
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        b1 -= learning_rate * db1
        b2 -= learning_rate * db2
        
        list_average_cost.append(average_cost)

        # 반복마다 비용 출력
        if count % 100 == 0:
            print('{}/{} cost : {}'.format(count, max_iteration, average_cost))
        count += 1      
# 학습 완료 후 결과 시각화
fig, ax = plt.subplots()
ax.plot(list_average_cost)
ax.set_title('Cost Over Iterations')
ax.set_xlabel('Iteration Number')
ax.set_ylabel('Cost')
plt.show()

for x,y in zip(x,y):
    a1,a2,a3,z2,z3 = feed_forward(x,w1,w2,b1,b2)
    print(y)
    print(a3)
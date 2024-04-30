import numpy as np

# 시그모이드 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 피드포워드를 수행하는 함수
def feed_forward(x, w1, w2, b1, b2):
    # 입력 계층에서 첫 번째 은닉 계층으로의 신호 전달
    a1 = x
    z2 = np.dot(w1, a1) + b1
    a2 = sigmoid(z2)

    # 첫 번째 은닉 계층에서 출력 계층으로의 신호 전달
    z3 = np.dot(w2, a2) + b2
    a3 = sigmoid(z3)

    return a1, a2, a3, z2, z3

# 노드 사이즈 설정
node_size = {
    'input_layer_size': 3,  # 입력 계층의 노드 수
    'hidden_layer_size': 3, # 은닉 계층의 노드 수
    'output_layer_size': 1  # 출력 계층의 노드 수
}

# 가중치 및 바이어스 초기 무작위 설정
w1 = np.random.rand(node_size['hidden_layer_size'], node_size['input_layer_size'])
w2 = np.random.rand(node_size['output_layer_size'], node_size['hidden_layer_size'])
b1 = np.random.rand(node_size['hidden_layer_size'])
b2 = np.random.rand(node_size['output_layer_size'])

# 입력 데이터 X와 목표 데이터 Y 설정
X = np.array([[1,0,0],[0,0,1],[1,0,1], [0,1,1], [1,1,0], [0,1,0], [1,1,1]])
Y = np.array([[1,0,0,0,1,1,0]])

# 피드포워드 및 오차 계산을 위한 반복 처리
for x, y in zip(X, Y):
    a1, a2, a3, z1, z2 = feed_forward(x, w1, w2, b1, b2)
    # 손실 함수로 L2 Norm을 사용해 오차 계산 및 출력
    print("a3-{} y-{}, Error(L2 Norm)-{}".format(a3, y, np.linalg.norm((y-a3), 2)))

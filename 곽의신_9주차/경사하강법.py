import matplotlib.pyplot as plt
import numpy as np

# 주어진 함수 정의
def f(x):
    return np.power(x-5, 2) - 20

# 주어진 함수의 도함수 정의
def f_derivative(x):
    return 2*x - 10

# 경사 하강법을 구현한 함수
def gradient_descent(next_x, gamma, precision, max_iteration):
    # 단계별로 이동 거리를 저장할 리스트 초기화
    list_step = []
    
    for i in range(max_iteration):  # 최대 반복 횟수만큼 반복
        current_x = next_x
        
        # 현재 위치에서 도함수 값을 사용하여 다음 위치를 업데이트
        next_x = current_x - gamma * f_derivative(current_x)
        
        # 이동 거리 계산
        step = next_x - current_x
        list_step.append(abs(step))
        
        # 반복 상황을 출력
        if i % 50 == 0:
            print("Iteration {:3}: x = {:.5f}, f(x) = {:.5f}, df(x) = {:.5f}".format(i, current_x, f(current_x), f_derivative(current_x)))
            gradient = gamma * f_derivative(current_x)
        # 이동 거리가 정밀도보다 작으면 종료
        if abs(step) <= precision:
            break
    
    print('Min Value of Cost Function is x = {}.' .format(current_x))

    figure, ax = plt.subplots(1, 1)
    ax.plot(list_step)
    ax.title.set_text('step size')
    ax.set_xlabel('Iteration number')
    ax.set_ylabel('Step size')
    plt.show()
    
# 초기값 설정 및 함수 호출
next_x = -10  # 초기 추측값
gamma = 0.01  # 학습률
precision = 0.00001  # 정밀도
max_iteration = 1000  # 최대 반복 횟수

#시작위치가 음수
gradient_descent(next_x, gamma, precision, max_iteration)  

#시작위치가 양수
next_x = 10
gradient_descent(next_x, gamma, precision, max_iteration)  

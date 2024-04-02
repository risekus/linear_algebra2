import numpy as np
import cv2

# 이미지 로드, 변환 행렬 생성
img = cv2.imread('C:/a.jpg', cv2.IMREAD_COLOR)
height, width = img.shape[:2]

# 원근 변환 행렬 생성 
M = np.array([[1, 0,  100], [0, 1, 200], [0, 0, 1]], dtype=float)

# 결과 이미지 생성을 위한 넘파이 배열 생성
dst = np.zeros(img.shape, dtype=np.uint8)

for y in range(height-1):
    for x in range(width-1):
        
        p = np.array([x,y,1])
        P_ = np.dot(M,p)
        
        x_,y_ = P_[:2]
        x_ = int(x_)
        y_ = int(y_)
        
        if x_ > 0 and x_<width and y_>0 and y_<height:
            dst[y_,x_] = img[y,x]

result=cv2.hconcat([img,dst])
cv2.imshow("result", result)
cv2.waitKey(0)

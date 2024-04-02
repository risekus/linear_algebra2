import numpy as np
import cv2

# 이미지 로드, 변환 행렬 생성
img = cv2.imread('C:/a.jpg', cv2.IMREAD_COLOR)
height, width = img.shape[:2]

scale_factor = 0.5
# 원근 변환 행렬 생성 
scaling_matrix = np.array([[scale_factor,0,0],[0,scale_factor,0],[0,0,1]])
translation_matrix = np.array([[1,0,width/4], [0,1,height/4],[0,0,1]])

angle = 45
radian = angle * np.pi/180
c = np.cos(radian)
s = np.sin(radian)
center_x = width/2
center_y = height/2
rotation_matrix = np.array([[c,s,(1-c) * center_x - center_y], 
                            [-s,c,s * center_x + (1-c) * center_y], 
                            [0,0,1]])

T = np.eye(3)
T = np.dot(scaling_matrix,T)
T = np.dot(translation_matrix, T)
T = np.dot(rotation_matrix,T)

# 결과 이미지 생성을 위한 넘파이 배열 생성
dst = np.zeros((height,width,img.shape[2]), dtype=np.uint8)

for y in range(height-1):
    for x in range(width-1):
        
        new_p = np.array([x,y,1])
        inv_scaling_matrix = np.linalg.inv(T)
        old_p = np.dot(inv_scaling_matrix, new_p)
        
        x_,y_ = old_p[:2]
        x_ = int(x_)
        y_ = int(y_)
        
        if x_ > 0 and x_<width and y_>0 and y_<height:
            dst.itemset((y,x,0), img.item(y_, x_, 0))
            dst.itemset((y,x,1), img.item(y_, x_, 1))
            dst.itemset((y,x,2), img.item(y_, x_, 2))
result=cv2.hconcat([img,dst])
cv2.imshow("result", result)
cv2.waitKey(0)

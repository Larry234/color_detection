import cv2
import numpy as np
import pandas as pd
from copy import deepcopy
import math
import imutils

#讀取圖片
imgobj = cv2.imread('./dataset/logo.jpg')
imgobj = cv2.resize(imgobj,(300,300))

#載入色碼表
index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('colors.csv', names=index, header=None,encoding='utf8')

#計算顏色相似度
def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)): #將RGB數值與色碼表中的所有顏色做相減取絕對值
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum): #如果顏色之間的差距小於10000，將距離值d設定成minimum，如果距離值d不再縮小，跳出迴圈，回傳顏色名稱
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

#ALAN ADDED 6/16 ------------------------------------------------------------------------------------------------------

list_of_colors = [[255,255,255],[255,255,0],[0,255,0],[0,0,255],[250,165,0],[255,0,0]]
                  #white         yellow      green     blue      orange      red
loc = np.array(list_of_colors)

list_of_colors_names = ['white','yellow','green','blue','orange','red']
locn = np.array(list_of_colors_names)


def closest(color):
    color = np.array(color)
    distances = np.sqrt(np.sum((loc-color)**2,axis=1))
    index_of_smallest = np.where(distances==np.amin(distances))
    smallest_dist_RGB = loc[index_of_smallest]
    smallest_dist_name = locn[index_of_smallest]
    return smallest_dist_name.tolist()

def getColorNew(R,G,B):
    current_color = [R,G,B]
    return closest(current_color)

#ALAN ADDED 6/16 ------------------------------------------------------------------------------------------------------


#取得路徑點
def draw_function(x,y):
    global b,g,r,xpos,ypos   #全域變數
    xpos = x
    ypos = y
    b,g,r = imgobj[y,x]      #取得在x,y座標下的bgr數值(RGB)
    b = int(b)
    g = int(g)
    r = int(r)

#影像灰階、模糊處理，並進行canny邊緣檢測
gray = cv2.cvtColor(imgobj, cv2.COLOR_BGR2GRAY) #BGR轉灰階
blurred = cv2.GaussianBlur(gray, (3, 3), 0)     #高斯模糊(3x3)
canny = cv2.Canny(blurred, 20, 100)             #最低閾值20，最高閾值100 # https://ithelp.ithome.com.tw/articles/10202295

kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(canny, kernel, iterations=5)  #影像膨脹 #https://shengyu7697.github.io/blog/2020/06/21/Python-OpenCV-erode-dilate/

#尋找Contours輪廓

# cv2.findContours()函式首先返回一個list，list中每個元素都是影象中的一個輪廓，用numpy中的ndarray表示。
# 返回可選的hiararchy結果，這是一個ndarray，其中的元素個數和輪廓個數相同，每個輪廓contours[i]對應4個hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分別表示後一個輪廓、前一個輪廓、父輪廓、內嵌輪廓的索引編號，如果沒有對應項，則該值為負數。
(contours, hierarchy) = cv2.findContours(dilated.copy(),              #複製膨脹後的影像
                                         cv2.RETR_TREE,               #樹狀結構輪廓
                                         cv2.CHAIN_APPROX_SIMPLE)     #壓縮水平方向，垂直方向，對角線方向的元素，只保留該方向的終點座標，例如一個矩形輪廓只需4個點來儲存輪廓資訊

# https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
candidates = []
hierarchy = hierarchy[0]

index = 0
pre_cX = 0
pre_cY = 0
center = []
color_list = []                             #取得顏色清單


# a = [1,2,3]
# b = [4,5,6]
# zipped = zip(a,b)     # list to tuple
# [(1, 4), (2, 5), (3, 6)]

for component in zip(contours, hierarchy):
    global color_text
    contour = component[0]
    peri = cv2.arcLength(contour, True)  # 取得周長，設為True時告訴電腦說該Contours邊界是closed沒有中斷
    approx = cv2.approxPolyDP(contour, 0.1 * peri, True) # 對 contour 做多邊形逼近的目的, 可以想成是用粗一點的線來描邊, 來忽略掉細微的毛邊或雜點. 再用隻程式來實驗:
    area = cv2.contourArea(contour) #取得面積
    corners = len(approx) #取得有幾個邊

    # compute the center of the contour(封閉的區塊)
    # 若Edge線條頭尾相連形成封閉的區塊，那麼它就是Contour，否則就只是Edge。
    
    # https://www.kancloud.cn/aollo/aolloopencv/272892
    # https://blog.csdn.net/shuiyixin/article/details/104646531
    M = cv2.moments(contour)     #取得Contour中心矩
    

    if M["m00"]:
        cX = int(M["m10"] / M["m00"])  #中點x座標
        cY = int(M["m01"] / M["m00"])  #中點y座標
    else:
        cX = None
        cY = None

    # 2000和20000是調整出來的
    if 2000 < area < 20000 and cX is not None:
        tmp = {'index': index, 'cx': cX, 'cy': cY, 'contour': contour}
        center.append(tmp)
        index += 1
        draw_function(cX,cY)   #取得該中心點下的BGR數值
        text = getColorNew(r,g,b) #計算顏色相似距離，取得RGB數值 <---------------------------------------------------------------------------------------changed from getColorName to getColorNew
        # print(text)


        color_list.append(text)

center.sort(key=lambda k: (k.get('cy', 0)))  #針對cy由小到大排列
row1 = center[0:3]
row1.sort(key=lambda k: (k.get('cx', 0)))
row2 = center[3:6]
row2.sort(key=lambda k: (k.get('cx', 0)))
row3 = center[6:9]
row3.sort(key=lambda k: (k.get('cx', 0)))

center.clear()
center = row1 + row2 + row3

for component in center:
    # print(type(component))
    candidates.append(component.get('contour'))

# print(len(candidates))  # 4
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#how-to-draw-the-contours

cv2.drawContours(imgobj, candidates, -1, (0, 0, 255), 2)

#print(color_list) # 初次判斷的顏色清單

tmp_topL = color_list[2]
tmp_topR = color_list[3]
tmp_bottomL = color_list[0]
tmp_bottomR = color_list[1]

# 重新排序後的顏色清單(左上、右上、左下、右下)
final_color_list = [tmp_topL,tmp_topR,tmp_bottomL,tmp_bottomR]
print(final_color_list)

cv2.imshow("image", imgobj)
cv2.waitKey(0)



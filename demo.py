import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

plt.figure(figsize=(8,6))
df = pd.read_csv('heightWeightData.csv', header=None)
data_x = np.array(df[1])
data_y = np.array(df[2])
data = []
for i in range(0,len(data_x)):
    data.append([data_x[i],data_y[i]])
data = np.array(data)


def point_distance(pointA , pointB):
    return np.sqrt(np.sum(np.power((pointA-pointB),2)))
 

def create_center_1(data,k):
    center_index = random.sample(range(0,len(data)),k)
    center_point = [data[i] for i in center_index]
    return center_point

#fxsdwdw
def create_center_2(data,k):
    center_index = random.sample(range(0,len(data)),1)
    center_point = [data[center_index[0]].tolist()]
    if k == 1:
        return center_point
    else :
        center_point_number = 0
        while True:
            max_distance = 0
            for point in data :
                if point.tolist() not in center_point:
                    distance = 0
                    for center in center_point:
                        distance = distance + point_distance(np.array(point) , np.array(center))
                    if distance > max_distance:
                        max_distance = distance
                        max_point = point.tolist()
            if max_point not in center_point:
                center_point.append(max_point)
                center_point_number = center_point_number + 1
            if len(center_point) == k :
                break;
        return center_point
        
def k_means(data,beginning_center_point,k):
    center_point_list = beginning_center_point
    while True:
        data_with_lables = []
        for point in data:
            distance = [point_distance(point , center_point) for center_point in center_point_list]
            data_with_lables.append([point.tolist(),distance.index(min(distance)) + 1])

        cluster = []
        class_number = 1 
        while class_number <= k :
            same_class = []
            for point in data_with_lables:
                if point[1] == class_number:
                    same_class.append(point)
            cluster.append(same_class)
            class_number = class_number + 1
    
        class_number = 1
        new_center_point_list = []
        while class_number <= k:
            total_x = 0
            total_y = 0
            for point in cluster[class_number - 1]:
                total_x = total_x + point[0][0]
                total_y = total_y + point[0][1]
            if len(cluster[class_number - 1]) == 0:
                new_center_point = [total_x , total_y]
            else:
                new_center_point = [total_x / len(cluster[class_number - 1]), total_y/len(cluster[class_number - 1])]
            new_center_point_list.append(new_center_point)
            class_number = class_number + 1
        same_center = True
        for i in range(0,len(new_center_point_list)):
            if new_center_point_list[i] in center_point_list:
                pass
            else :
                same_center = False
        center_point_list = new_center_point_list
        if same_center == True :
            center_point_list = new_center_point_list
            break;
    return [data_with_lables,center_point_list]

k = 2
beginning_center_point = np.array(create_center_2(data,k))
result = k_means(data,beginning_center_point,k)
colors = np.array(["red", "gray", "orange", "pink", "blue", "black"])

for point in result[0]:
    for i in range(1,k+1):
        if point[1] == i:
            plt.subplot(221)
            plt.scatter(point[0][0],point[0][1],color = colors[i])
for center_point in result[1]:
    plt.subplot(221)
    plt.scatter(center_point[0],center_point[1],marker = 's', color = 'black')


julei = KMeans(n_clusters= 4)
julei.fit(np.array(data))
label = julei.labels_.tolist()
index = 0
for i in data :
    plt.subplot(222)
    plt.scatter(i[0],i[1],color = colors[label[index]])
    index = index + 1


def SSE(result, k):
    class_number = 1
    class_distance = 0
    while class_number <= k :
        for point in result[0]:
            if point[1] == class_number:
                class_distance = class_distance + point_distance(np.array(point[0]), np.array(result[1][class_number - 1]))
        class_number = class_number + 1
    return class_distance 

sse_k = []
for i in range(1,10):
    beginning_center_point = np.array(create_center_2(data,i))
    result = k_means(data,beginning_center_point,i)
    sse_k.append(SSE(result,i))
plt.subplot(223)
plt.plot([[i] for i in range(1,10)],sse_k)


plt.show()




 






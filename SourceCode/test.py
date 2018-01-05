import numpy as np
import glob
from PIL import Image # 2.7 sec
#from cv2 import imread, resize # 2.8 sec
import time

images = []
names = []

image_filenames = glob.glob("data/ILSVRC/Data/DET/train/*/*.JPEG")

start = time.time()

count = 0
for image_file in image_filenames:
    #image_data = imresize(imread(image_file), size = (256, 256, 3))#.astype(np.float32)
    #image_data = resize(imread(image_file), output_shape = (256, 256, 3)) #slowest 12 sec for 500 images
    #image_data = resize(imread(image_file), (256,256))
    if count%10 == 0:
        try:
            image_data = np.asarray(Image.open(image_file).resize((32, 32)).convert("RGB"))[:, :, :3]
            #images.append(np.reshape(image_data, 3072)) #196608
            images.append(image_data)
            names.append(image_file)
        except:
            pass
    count += 1
    if count == 25000:
        break
print(time.time() - start)

k = 16
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[*]").setAppName("LSH").set('spark.executor.memory', '32G').set('spark.driver.memory', '32G').set('spark.driver.maxResultSize', '8G')
sc = SparkContext(conf=conf)#spark://ip-172-31-46-196.ec2.internal:7077

rdd1 = sc.parallelize(zip(names,images), 64)


def map1(tuple1):
    name, image = tuple1
    shingles = []
    # shingle = list(image[:k, :k])
    for i in range(image.shape[0] - k):
        for j in range(image.shape[1] - k):
            red_img = image[i:i + k, j:j + k, :].flatten()
            shingles.append(tuple(red_img))  # , :].flatten()))
    # shingles.append(tuple(shingle))

    return (name, shingles)

rdd2 = rdd1.map(map1)

from pyspark import StorageLevel

DISK_ONLY = StorageLevel(True, False, False, False, 1)

rdd2.persist(DISK_ONLY)

rdd3 = rdd2.flatMap(lambda r: r[1])

k_shingle_space = rdd3.distinct().collect()

N = len(k_shingle_space)

print(N)

def map2(tuple1):
    name, image = tuple1
    #bool_vec = bitarray(N)
    #bool_vec.setall(0)
    #bool_vec = np.zeros(N, dtype = np.bool)
    idx_vec = []
    for i in range(N):
        if k_shingle_space[i] in image:
            idx_vec.append(i)
            #bool_vec[i] = 1
            #bool_vec[i] = True
    return (name, idx_vec)

rdd4 = rdd2.map(map2)

a = np.random.randint(low = -10000, high = 10000, size = 100).tolist()
b = np.random.randint(low = -10000, high = 10000, size = 100).tolist()

import sympy
#N = len(k_shingle_space)
for k in range(N):
    if sympy.isprime(k+N+1):
        p = k+N+1
        break

print(p)

import sys
MAX = sys.maxsize

sig_size = 128

rdd4.persist(DISK_ONLY)

def map3(tuple1):
    name, image = tuple1
    M = [MAX]*sig_size
    for i in range(len(image)):
        for j in range(sig_size):
            h = ((a[j]*i + b[j])%p)%N
            if h < M[j]:
                M[j] = h
    return (name, M)

rdd5 = rdd4.map(map3)

sig_matrix = rdd5.collect()

sc.stop()

d = {}
bands = 16
r = 8

def map4(tuple1):
    name, image = tuple1
    for i in range(bands):
        band = tuple(image[i*r:(i+1)*r])
        if band not in d:
            d[band] = [name]
        else:
            d[band].append(name)

for i in sig_matrix:
    map4(i)

sig_dict = dict(sig_matrix)

from sklearn.metrics import jaccard_similarity_score

stripes = {}
for k in d:
    if len(d[k]) > 1 and len(d[k]) < 20:
        for i in range(len(d[k])):
            for j in range(i+1, len(d[k])):
                if i != j:
                    score = jaccard_similarity_score(sig_dict[d[k][i]], sig_dict[d[k][j]])
                    if score > 0.9:
                        if d[k][i] not in stripes:
                            stripes[d[k][i]] = [d[k][j]]
                        else:
                            if d[k][j] not in stripes[d[k][i]]:
                                stripes[d[k][i]].append(d[k][j])

print(stripes)
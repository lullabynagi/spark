import os
import numpy as np
import json
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from sentence_transformers import SentenceTransformer, util
# 創建SparkContext對象
import findspark
from time import process_time, sleep


findspark.init()
spark = SparkSession.builder.master('spark://192.168.56.1:7077').appName('test').getOrCreate()
sc = SparkContext.getOrCreate()




model = SentenceTransformer('paraphrase-distilroberta-base-v1')

songs = []
now_song_name = ""
now_song=[]
next_songs = []


index = 'challenge_set.json'

with open(index, "r") as f:
    result = json.load(f)


print("歌曲讀取中")
print(result["playlists"][1000]["tracks"][0])
for i in range(len(result["playlists"])):#10000
    if len(result["playlists"][i]["tracks"])>0:
        for j in range(len(result["playlists"][i]["tracks"])):
            song=[]

            try:
                song.append(result["playlists"][i]["name"])
            except:
                song.append('none')
            song.append(result["playlists"][i]["num_holdouts"])
            song.append(result["playlists"][i]["pid"])
            song.append(result["playlists"][i]["num_tracks"])
            song.append(result["playlists"][i]["tracks"][j])
            song.append(result["playlists"][i]["num_samples"])
            print(song)
            print("\n")
            songs.append(song)



#現在的歌名
now_song_name="I'm an Albatraoz"
#I'm an Albatraoz
#Little Swing
#Yellow Flicker Beat
#Pure Heroine
print("歌曲尋找中")
for i in range(len(songs)):
  if songs[i][4]["album_name"]==now_song_name:
    now_song.append(songs[i])




print("embeddings")
now_song_embeddings = model.encode(str(now_song[0]))





# 建立 RDD
print("songs_embedding讀取中")
index = 'songs_embedding.json'
with open(index, "r") as f:
    result = json.load(f)
print(result)
small_result=[]
print("songs_embeddingn縮小中")
for i in range(150):
    small_result.append(result[i])

print(len(small_result[0]))
print(type(small_result[0]))
print(len(small_result))
print(type(small_result))
start = process_time()  


# 建立 RDD
rdd = sc.parallelize(small_result)


def do_cos(x):
    cos=util.cos_sim(x,now_song_embeddings)
    print(cos)
    return cos
# 執行操作
result_rdd = rdd.map(do_cos)

print("cos中")
cos = result_rdd.collect()

# 顯示結果
print(cos)
end = process_time()
print(end, start)
print(end-start)

cos_scores_arr=[]

for id in range(len(small_result)):
    cos_scores_arr.append(cos[id][0][0].numpy().tolist())
    print(cos[id])

sort_cos_scores_arr=[]
for id in range(len(small_result)):
    sort_cos_scores_arr.append(cos_scores_arr[id])




sort_cos_scores_arr.sort(reverse=True)

print(cos_scores_arr)
print(sort_cos_scores_arr)

print(cos_scores_arr.index(sort_cos_scores_arr[id]))

for id in range(5):
    next_songs.append(cos_scores_arr.index(sort_cos_scores_arr[id]))
print(next_songs)
for id in range(1,2):
    print("下一首歌:"+songs[next_songs[id]][4]["album_name"])


end = process_time()
print(end, start)
print(end-start)

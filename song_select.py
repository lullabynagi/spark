import numpy as np
import json
from sentence_transformers import SentenceTransformer
from time import process_time, sleep


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




now_song_name="Pure Heroine"
#I'm an Albatraoz
#Little Swing
#Yellow Flicker Beat
#Pure Heroine
print("歌曲尋找中")
for i in range(len(songs)):
  if songs[i][4]["album_name"]==now_song_name:
    now_song.append(songs[i])

str_songs=[]
for i in range(len(songs)):
    str_songs.append(str(songs[i]))

print(now_song)
print(now_song[0])

print("embeddings")
now_song_embeddings = model.encode(str(now_song[0]))

#songs_embeddings=model.encode(str_songs)

#index = 'songs_embedding.json'
#with open(index, "a") as f:
    #json.dump(songs_embeddings.tolist(), f, indent = 4)
print("songs_embedding讀取中")
index = 'songs_embedding.json'
with open(index, "r") as f:
    result = json.load(f)

small_result=[]

print("songs_embeddingn縮小中")
for i in range(150):
    small_result.append(result[i])


from sentence_transformers import SentenceTransformer, util
total_songs = len(small_result)
cos_scores=[]
cos_scores_arr=[]

start = process_time()  
for id in range(total_songs):
    cos_scores.append(util.cos_sim(now_song_embeddings, small_result[id]))
    print(cos_scores[id])
print(cos_scores)


for id in range(total_songs):
    cos_scores_arr.append(cos_scores[id][0][0].numpy().tolist())
    print(cos_scores_arr[id])


print(cos_scores_arr)
sort_cos_scores_arr=[]
for id in range(total_songs):
    sort_cos_scores_arr.append(cos_scores_arr[id])

sort_cos_scores_arr.sort(reverse=True)

print(cos_scores_arr)
print(sort_cos_scores_arr)

print(cos_scores_arr.index(sort_cos_scores_arr[id]))
for id in range(10):
    next_songs.append(cos_scores_arr.index(sort_cos_scores_arr[id]))
print(next_songs)
for id in range(10):
    print(str_songs[next_songs[id]])
for id in range(10):
    print(songs[next_songs[id]][4]["album_name"])

end = process_time()
print(end, start)
print(end-start)
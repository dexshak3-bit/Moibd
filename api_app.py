
from fastapi import FastAPI, HTTPException, Query
from pathlib import Path
import pandas as pd, numpy as np, re, sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
BASE=Path(__file__).resolve().parent
movies=pd.read_csv(BASE/"movies.csv")
ratings=pd.read_csv(BASE/"ratings (1).csv")
ratings=ratings[["userId","movieId","rating"]].dropna()
def clean(s): return re.sub(r"\s+"," ",re.sub(r"\s*\(\d{4}\)\s*"," ",str(s))).strip()
movies["title_clean"]=movies.title.map(clean); movies["genres"]=movies.genres.fillna("")
st=ratings.groupby("movieId").rating.agg(["count","mean"]).reset_index()
movies=movies.merge(st,on="movieId",how="left").fillna({"count":0,"mean":0})
C=ratings.rating.mean(); m=max(int(movies["count"].quantile(.9)),1)
movies["weighted"]=movies["count"]/(movies["count"]+m)*movies["mean"]+m/(movies["count"]+m)*C
ui=ratings.pivot_table(index="userId",columns="movieId",values="rating"); ids=ui.columns.to_list()
from scipy.sparse import csr_matrix
idxmap={int(x):i for i,x in enumerate(ids)}; msim=cosine_similarity(csr_matrix(ui.fillna(0).T.to_numpy()), dense_output=False)
tf=TfidfVectorizer(token_pattern=r"[^ ]+")
cmat=tf.fit_transform(movies.genres.str.replace("|"," ",regex=False))
app=FastAPI(title="Film Recommendation API")
def find(title):
 q=clean(title).casefold(); d=movies[movies.title_clean.str.casefold()==q]
 if d.empty: d=movies[movies.title_clean.str.casefold().str.contains(re.escape(q),na=False)]
 if d.empty: raise HTTPException(404,"Фильм не найден")
 return d.sort_values(["count","mean"],ascending=False).iloc[0]
def output(d):
 d=d.copy()
 if "title_clean" in d: d["title"]=d.title_clean
 return d.replace({np.nan:None}).to_dict("records")
@app.get("/")
def help():
 return {"docs":"/docs","commands":["/popular?limit=10","/similar?title=Toy%20Story&limit=10","/content?title=Toy%20Story&limit=10","/user/1?limit=10","/rate?user_id=1&title=Toy%20Story&rating=4"]}
@app.get("/popular")
def popular(limit:int=Query(10,ge=1,le=100)):
 d=movies[movies["count"]>=20].sort_values(["weighted","count"],ascending=False).head(limit)
 return output(d[["movieId","title_clean","genres","count","mean","weighted"]].rename(columns={"count":"rating_count","mean":"mean_rating"}))
@app.get("/similar")
def similar(title:str,limit:int=Query(10,ge=1,le=100)):
 f=find(title); mid=int(f.movieId)
 if mid not in idxmap: raise HTTPException(404,"Недостаточно оценок")
 scores=msim.getrow(idxmap[mid]).toarray().ravel(); order=[i for i in np.argsort(scores)[::-1] if ids[i]!=mid][:limit]
 d=movies[movies.movieId.isin([ids[i] for i in order])].copy(); d["similarity"]=d.movieId.map({ids[i]:float(scores[i]) for i in order})
 return output(d.sort_values("similarity",ascending=False)[["movieId","title_clean","genres","similarity"]])
@app.get("/content")
def content(title:str,limit:int=Query(10,ge=1,le=100)):
 f=find(title); idx=movies.index[movies.movieId==int(f.movieId)][0]; scores=cosine_similarity(cmat[idx],cmat).ravel()
 order=[i for i in np.argsort(scores)[::-1] if int(movies.iloc[i].movieId)!=int(f.movieId)][:limit]
 d=movies.iloc[order].copy(); d["similarity"]=scores[order]
 return output(d[["movieId","title_clean","genres","similarity"]])
@app.get("/user/{user_id}")
def user_recs(user_id:int,limit:int=Query(10,ge=1,le=100)):
 if user_id not in ui.index: raise HTTPException(404,"Пользователь не найден")
 mat=ui.fillna(0); sims=cosine_similarity(mat.loc[[user_id]],mat).ravel()
 near=pd.Series(sims,index=mat.index).drop(user_id).sort_values(ascending=False).head(30)
 num=pd.Series(0.,index=mat.columns); den=pd.Series(0.,index=mat.columns)
 for uid,s in near.items():
  if s<=0: continue
  vals=ui.loc[uid]; rated=vals.notna(); num.loc[rated]+=vals.loc[rated]*s; den.loc[rated]+=s
 scores=(num/den.replace(0,np.nan)).drop(index=ui.loc[user_id].dropna().index,errors="ignore").dropna().sort_values(ascending=False).head(limit)
 d=movies[movies.movieId.isin(scores.index)].copy(); d["predicted_rating"]=d.movieId.map(scores)
 return output(d.sort_values("predicted_rating",ascending=False)[["movieId","title_clean","genres","predicted_rating"]])
@app.get("/rate")
def rate(user_id:int,title:str,rating:float=Query(...,ge=.5,le=5)):
 f=find(title); con=sqlite3.connect(BASE/"ratings_user.sqlite")
 con.execute("CREATE TABLE IF NOT EXISTS user_ratings(user_id INTEGER,movie_id INTEGER,title TEXT,rating REAL,created_at TEXT DEFAULT CURRENT_TIMESTAMP,UNIQUE(user_id,movie_id))")
 con.execute("INSERT INTO user_ratings(user_id,movie_id,title,rating) VALUES(?,?,?,?) ON CONFLICT(user_id,movie_id) DO UPDATE SET rating=excluded.rating,title=excluded.title",(user_id,int(f.movieId),f.title_clean,float(rating)))
 con.commit(); con.close()
 return {"status":"saved","user_id":user_id,"movie":f.title_clean,"rating":rating}

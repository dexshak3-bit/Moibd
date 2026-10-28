
import streamlit as st, requests, pandas as pd
st.set_page_config(page_title="рекомендации фильмов",layout="wide")
st.title("рекомендательная система фильмов")
api=("http://127.0.0.1:8000")
mode=st.sidebar.selectbox("режим",["топ 10 популярных","похожие по оценкам","по жанрам","персональные рекомендации"])
limit=st.sidebar.slider("Количество",1,100,10)
try:
 if mode=="Топ-10 популярных":
  r=requests.get(api+"/popular",params={"limit":limit},timeout=60)
 elif mode in ("похожие по оценкам","по жанрам"):
  title=st.text_input("название фильма","Hobbit")
  endpoint="/similar" if mode=="похожие по оценкам" else "/content"
  r=requests.get(api+endpoint,params={"title":title,"limit":limit},timeout=60)
 else:
  uid=st.number_input("id пользователя",min_value=1,value=1,step=1)
  r=requests.get(api+f"/user/{int(uid)}",params={"limit":limit},timeout=60)
 if r.ok:
  d=pd.DataFrame(r.json()).rename(columns={"title":"название","genres":"жанры","similarity":"сходство","predicted_rating":"прогноз оценки"})
  st.dataframe(d,use_container_width=True,hide_index=True)
 else: st.warning(r.text)
except requests.RequestException:
 st.error("API недоступен. Запустите uvicorn api_app:app --reload")
st.divider(); st.subheader("поставить оценку фильму")
with st.form("rate"):
 uid=st.number_input("id пользователя",min_value=1,value=1,step=1,key="rate_uid")
 title=st.text_input("название фильма","Toy Story",key="rate_title")
 rating=st.slider("оценка",.5,5.,4.,.5)
 send=st.form_submit_button("сохранить")
 if send:
    try:
        r = requests.get(api+"/rate", params={"user_id":int(uid), "title":title, "rating":rating}, timeout=30)
        if r.ok:
            st.success("оценка сохранена: " + r.json()["movie"])
        else:
            st.error(r.text)
    except requests.RequestException:
        st.error("API недоступен")
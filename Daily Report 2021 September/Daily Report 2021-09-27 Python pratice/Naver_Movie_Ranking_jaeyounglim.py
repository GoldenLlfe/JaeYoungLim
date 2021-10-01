#!/usr/bin/env python
# coding: utf-8

# In[21]:


# request : url로 부터 자료를 가져옴
# beautifulsoup : tag 형식으로 자료변환
# 변환된자료.selevt(), find(), find_all(), select_one -> site의 정보 확인
#네이버 영화 쉰위 출력
#순위 영화명 평점 출력


from bs4 import BeautifulSoup
from urllib import request
movietitle, movie_rank = [],[]
url = 'https://movie.naver.com/movie/sdb/rank/rmovie.naver?sel=cur&date=20210926'
soup = BeautifulSoup(request.urlopen(url))

for item in soup.find_all("tr"):
    movie = item.find('div', class_="tit5")
    if movie:  #자료가 존재하면 movie_title 리스트에 추가
        movietitle.append(movie.get_text().strip("\n")) #movietitle 리스트에 자료를 줄바꿈을(\n)을 기준으로 추가함
        
    rank = item.find('td', class_='point')
    if rank:  #평점이 있다면 moive_rank에 추가 (가끔 자료가 없는 경우가 있는데 그때는 에러 출력같은걸로 처리해야함)
        movie_rank.append(rank.text)


# In[22]:


for i,(movie,rank) in enumerate(zip(movietitle,movie_rank)):
    print(i+1, movie, ' : ', rank)


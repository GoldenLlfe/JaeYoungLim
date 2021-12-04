use adventureworks;

-- sql 의 기본 문법

select		컬럼 표시
from 		데이터의 위치
where 		필터링 조건
group by	집계성 쿼리
order by 	정렬 순서
limit		출력 레코드 수

-- select
select *  -- 모든 컬럼을 다 보열라라는 것
from product  -- 이 경우에는 product의 모든 컬럼을 다 보여준다
--
select productid, name, safetystocklevel
from product -- 위에 지정된 컬럼들만 보여준다

select productid, name, safetystocklevel
from product
where safetystocklevel>=500  -- 필터링 - 안전제고가 500개 이상인 것만 보여준다
		and name like '%crankarm'  -- 이름안의 crankarm 이 있는 것만 검색 
        -- 문자열을 쓸때는 앞에 like를 붙여야하고 따옴표 안에 %--% 를 써야한다
        -- %ㅁㅁ 앞에 ㅁㅁ가 들어가는 것
        -- ㅁㅁ% 뒤에 ㅁㅁ가 들어가는 것
        -- %ㅁㅁ% 앞 뒤로 ㅁㅁ가 있는 것 
        -- ㅁㅁ 오직 ㅁㅁ만 있는것
        
select productid, name, safetystocklevel
from product
where safetystocklevel>=500
order by reorderpoint desc
limit 5; -- 재고가 5백개 넘는것 중에서 위에서부터 보여주고 5개만 출력alter

-- mysql
-- table의 정보의 테일ㅇㄹ 보고 싶을때alter

show tables; -- 현재 스키마의 테이블 목록을 보여줌 
show databases;
show columns from product;
select * from product;

show columns from salesorderdetail;
select * from salesorderdetail;

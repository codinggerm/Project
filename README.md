# 주택분류 로직
## 1. 집합건축물 분류
- 전제 
    - jpk(건축물대장 전유공용면적의 primary key) 단위로 호별 주용도를 정함.
    - 앞서 정해진 jpk의 주용도를 기반으로 ppk(건축물대장 표제부의 primary key) 단위로 건물의 대표 주용도, 건물의 주택 대표 주용도, 건물의 비주택 대표 주용도를 정함.
- 표제부 분류 로직
    - 표제부와 기본개요를 매칭했을 때 매칭이 된 ppk 건은 집합건축물로, 매칭이 되지 않은 ppk 건은 일반건축물로 간주함.
    - jpk 중복이 제거된 전유공용면적과 표제부를 기본개요를 통해 매칭
    - 공동주택공시가격과 오피스텔기준시가를 이용한 건축물대장의 비주택용도 정제
    - ppk 단위로 건물의 대표 주용도, 건물의 주택 대표 주용도, 건물의 비주택 대표 주용도 결정
- 주택분류 로직
    - 그룹핑 테이블 기반으로 단지형아파트, 나홀로아파트로 분류가 된, 주택 대표 주용도명이 아파트인 ppk에 한해서 그에 속하는, 정제된 전유공용면적 상의 주용도코드명이 아파트인 jpk의 주택분류는 해당 ppk의 분류인 단지형 또는 나홀로아파트 분류를 적용함.
        - 주택 대표 주용도명이 아파트가 아닌 집합건축물에 대해서는, 그룹핑 테이블 기반으로 해당 건물의 아파트 전유부 개수와 같은 단지 내 주택 대표 주용도명이 아파트인 아파트 건물수를 계산하여 전유부 단위로 단지형아파트, 나홀로아파트를 분류함.
EX) 3개동으로 이루어진 단지, 각 동의 전유부 수는 총 30호(오피스텔 20호, 아파트 10호)  모든 동의 주택 대표 주용도명은 오피스텔이지만, 각 동의 아파트의 주택분류는 나홀로아파트임
    - 정제된 전유공용면적 상의 주용도코드명이 연립주택 또는 다세대주택인 jpk의 주택분류는 연립다세대로 총칭함.
    - 정제된 전유공용면적 상의 주용도코드명이 오피스텔인 jpk의 주택분류는 오피스텔로 분류함.
    - jpk가 존재하고 해당 전유공용면적 상의 주용도코드명이 단독주택, 다가구주택, 다중주택 중 어느 하나에 속하는 경우 그 jpk 주택분류는 none 값 처리함.

## 2. 일반건축물 분류
- 전제 
    - 층별 주용도를 기반으로 ppk 단위로 건물의 대표 주용도, 건물의 주택 대표 주용도, 건물의 비주택 대표 주용도를 정함.
- 표제부 분류 로직
    - 앞서 집합건축물 분류에서 표제부와 기본개요를 매칭했을 때 매칭이 되지 않는, 일반건축물로 간주되는 ppk건들로 분류 시작
    - 그 중 대장구분코드명이 ‘집합’인 ppk건들을 제외하고 분류 불가 건으로 따로 저장
        - 분류 불가 건 : 기본개요 또는 전유부와 매칭이 안된 집합건축물로서, 아파트의 부속건축물(경비실, 주차장 등)이 대부분임.
    - 주용도 분류 대상 ppk건들을 층별개요와 매칭 후 매칭이 안된 건들을 분류 불가 건으로 따로 저장
    - ppk 단위로 건물의 대표 주용도, 건물의 주택 대표 주용도, 건물의 비주택 대표 주용도 결정
- 주택분류 로직
    - 층별개요가 매칭이 안된 ppk건 중 그 표제부 주용도가 단독주택, 다가구주택, 다중주택 중 어느 하나에 속하는 경우 해당 ppk의 주택분류는 표제부 주용도를 따름. 


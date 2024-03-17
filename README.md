# 주택분류 로직
## 1. 용어 정의
-	(집합)건물의 대표 주용도란 해당 건물의 전유공용면적상의 주용도 중 그 전용면적합이 가장 높은 용도를 뜻함. 다만, 그 합이 같은 용도가 2개 이상 존재할 시 그 중 개수가 가장 많은 주용도를 따름. 개수가 같은 용도가 2개 이상 존재할 시 주용도의 우선순위에 따름.
    - 주용도 우선순위 : 아파트 > 오피스텔 > 연립다세대
-	(집합)건물의 주택 대표 주용도란 해당 건물의 공동주택공시가격과 오피스텔 기준시가로 정제된 전유공용면적상의 주택으로 여겨지는 용도(주택 용도) 중 그 전용면적합이 가장 높은 용도를 뜻함. 다만, 그 합이 같은 용도가 2개 이상 존재할 시 그 중 개수가 가장 많은 주용도를 따름. 개수가 같은 용도가 2개 이상 존재할 시 주용도의 우선순위에 따름.
	주택으로 여겨지는 용도 : 전유공용면적상의 주용도코드명 중 아파트, 연립주택, 다세대주택, 오피스텔
-	(집합)건물의 비주택 대표 주용도란 해당 건물의 공동주택공시가격과 오피스텔 기준시가로 정제된 전유공용면적상의 주택으로 여겨지지 않는 용도(비주택 용도) 중 그 전용면적합이 가장 높은 용도를 뜻함. 다만, 그 합이 같은 용도가 2개 이상 존재할 시 그 중 개수가 가장 큰 주용도를 따름. 개수가 같은 용도가 2개 이상 존재할 시 주용도의 우선순위에 따름.
	주택으로 여겨지지 않는 용도 : 전유공용면적상의 주용도코드명 중 아파트, 연립주택, 다세대주택, 오피스텔 외
-	(일반)건물의 대표 주용도란 해당 건물의 층별개요상의 주용도 중 그 층별면적합이 가장 높은 용도를 뜻함. 다만, 그 합이 같은 용도가 2개 이상 존재할 시 그 중 개수가 가장 많은 주용도를 따름. 개수가 같은 용도가 2개 이상 존재할 시 주용도코드의 오름차순 정렬 후 1순위 주용도에 따름.
-	(일반)건물의 주택 대표 주용도란 해당 건물의 층별개요상의 주택으로 여겨지는 용도(주택 용도) 중 그 층별면적합이 가장 높은 용도를 뜻함. 다만, 그 합이 같은 용도가 2개 이상 존재할 시 그 중 개수가 가장 많은 주용도를 따름. 개수가 같은 용도가 2개 이상 존재할 시 주용도코드의 오름차순 정렬 후 1순위 주용도에 따름.
	주택으로 여겨지는 용도 : 층별개요상의 주용도코드명 중 단독주택, 다가구주택, 다중주택
-	(일반)건물의 비주택 대표 주용도란 해당 건물의 층별개요상의 주택으로 여겨지지 않는 용도(비주택 용도) 중 그 층별면적합이 가장 높은 용도를 뜻함. 다만, 그 합이 같은 용도가 2개 이상 존재할 시 그 중 개수가 가장 큰 주용도를 따름. 개수가 같은 용도가 2개 이상 존재할 시 주용도코드의 오름차순 정렬 후 1순위 주용도에 따름.
	주택으로 여겨지지 않는 용도 : 층별개요상의 주용도코드명 중 단독주택, 다가구주택, 다중주택 외

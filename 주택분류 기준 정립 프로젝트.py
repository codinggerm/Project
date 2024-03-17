# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:37:15 2023

@author: dongkyun you
"""

#%% 경로 및 데이터 설정

import os
import oracledb
import numpy as np
import pandas as pd 
from tqdm import tqdm
import datetime as dt 
from glob import glob
from functools import reduce
from urllib.parse import quote
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta

month = (dt.datetime.now() - relativedelta(months = 2)).strftime('%Y%m')
stym = (dt.datetime.now()).strftime('%Y%m')

# 시도코드
sido_path = r'D:\시도코드.txt'
sido = pd.read_csv(sido_path, engine = 'python', dtype = str, sep = '|')

# 건축물대장 경로
path = 'D:/'
t_path = 'D:/'

# 주용도 수정용 공시가격 데이터
of_path = 'D:/'
gp_path = 'D:/'
doro_path = 'D:/'

# 결과물 저장 경로
sv_path1 = 'D:/'
sv_path2 = 'D:/'
sv_path3 = 'D:/'
j_sv_path = 'D:/'
pjb_path = 'D:/'
rst_path = 'D:/'

for p in [sv_path1, sv_path2, sv_path3, j_sv_path] : 

    if os.path.isdir(p) == False : 
        os.makedirs(p)
    else :
        pass

# 로그 파일
log_file = sv_path1 + '시도별_통계_로그.txt'
log_file1 = sv_path2 + '시도별_통계_로그.txt'

def get_columns_from_db(conn, schema, table):

    query = f"""
    select column_name
    from information_schema.columns
    where table_schema = '{schema}'
    and table_name = '{table}'
    and column_default is null
    order by ordinal_position;
    """

    with conn.cursor() as cursor:
        cursor.execute(query)
        columns = cursor.fetchall()
    return str(tuple(column[0] for column in columns)).replace("'", '"')

def copy_from_file(conn, schema, table, file, delimiter, encoding='cp949', opened=True, header=True):
    
    cols = get_columns_from_db(conn, schema, table)

    if header:
        query = f"COPY {schema}.{table} {cols} FROM STDIN csv DELIMITER '{delimiter}' HEADER"
    else:
        query = f"COPY {schema}.{table} {cols} FROM STDIN csv DELIMITER '{delimiter}'"

    if opened:
        with conn.cursor() as cursor:
            cursor.copy_expert(query, file)
            conn.commit()
            print(f'{table} upload 작업완료')
    else:
        with open(file, encoding=encoding) as openfile:
            with conn.cursor() as cursor:
                cursor.copy_expert(query, openfile)
                conn.commit()
                print(f'{table} upload 작업완료')

def chunker(dataframe, size):
    return [dataframe.iloc[pos : pos + size] for pos in range(0, len(dataframe), size)]

def oracle_insert(conn, table, generator):
    
    with conn.cursor() as cursor:
        
        for df in generator:
            cols = tuple(df.columns)
            vals = tuple(f":{i}" for i in range(1, len(df.columns)+1))
            sql = f"insert into {table} {cols} values {vals}".replace("'", "")

            cursor.executemany(sql, df.values.tolist())
            
        conn.commit()
    print(f"{table} 입력 완료")

#%% 집합건축물, 일반건축물 분류

pnu_error_df = pd.DataFrame()
final_result_all = pd.DataFrame() # 집합 전체 
final_result_all1 = pd.DataFrame() # 일반 전체
jy_cat_result = pd.DataFrame()
no_cat_result_all = pd.DataFrame() # 기타1, 기타2, 그룹 id 없는 것

# i = 0

for i in tqdm(range(len(sido))) : 
    
    print('\n' + '='*10 + ' ' + sido['sido_nm_2'][i] + ' ' + '='*10)
    
    sido_cd = sido['sido_cd'][i]
    # DW 정제된 전유공용면적 불러오기
    conn = create_engine('postgresql://postgres:' + quote_plus('') + '@:5432/')
    
    query = f'''
            SELECT
            
            jpk as V0,
            purpose_code,
            purpose_name,
            private_area
            
            FROM 건축물_대장.tb_private_public_area
            WHERE standard_date = '{stym}' and SUBSTR(jpk, 1, 2) = '{sido_cd}';
            '''
    
    jy = pd.read_sql(query, conn)
    jy.columns = ['V0', 'V34', 'V35', 'V37']    
    jy['V37'] = jy['V37'].fillna(0)
    
    bc = pd.read_csv(path + sido['sido_ind'][i] + '_basic.txt', sep = '|', encoding = 'cp949', dtype = str, header = None, prefix = 'B')
    bc = bc[['B0', 'B1']]
    
    # 일반건축물 분류용
    fl = pd.read_csv(path + sido['sido_ind'][i] + '_floor.txt', sep = '|', encoding = 'cp949', dtype = str, header = None, prefix = 'F')
    fl = fl[['F0', 'F25', 'F26', 'F28']]
    
    # 표제부 원천
    pjb = pd.read_parquet(path + sido['sido_ind'][i] + '_pjb.parquet')
    pjb = pjb[['V0', 'V1', 'V34', 'V35']]
    
    # 표제부, 기본개요, 전유공용면적 매칭
    pjb1 = pd.merge(pjb, bc, left_on = 'V0', right_on = 'B1', how = 'left', indicator = True)
        
    pjb2 = pjb1[pjb1['_merge'] == 'both']
    pjb2.drop('_merge', axis = 1, inplace = True)
    
    # 표제부, 기본개요 미매칭 건 --> 전유부가 없다는 뜻이니 일반건축물로 간주
    ilban = pjb1[pjb1['_merge'] == 'left_only']
    ilban.drop(['B0', 'B1', '_merge'], axis = 1, inplace = True)
    
    trial = pd.merge(pjb2, jy, left_on = 'B0', right_on = 'V0', how = 'left')
    trial.drop(['B0', 'B1'], axis = 1, inplace = True)
    trial['전유_주용도코드_원천'] = trial['V34_y']
    trial['전유_주용도명_원천'] = trial['V35_y']
    
    # ========================= 공시가격으로 주용도 정제하기 =========================
    trial1 = trial.copy()
    
    # 공동주택공시가격 불러와서 주용도 덮어씌우기 / 건축물대장상 주용도가 공동주택용도가 아닌 경우
    gp = pd.read_csv(gp_path + sido['sido_ind'][i] + '_' + sido['sido_nm_2'][i] + '.txt', sep = '|', encoding = 'utf-8', dtype = str, usecols = ['jpk', '공동주택구분명'])
    trial2 = pd.merge(trial1, gp, left_on = 'V0_y', right_on = 'jpk', how = 'left', indicator = True)
    # 건축물대장 상 비주택인 주용도를 공시가격과 매칭하는데 비주택에 단독주택, 다가구주택 따위를 포함함. 왜냐하면 주용도가 단독주택인데 공시가격과 매칭했을 때 다세대주택인 것이 있기 때문. ex) jpk : 11110-50075
    t_check = trial2[(trial2['_merge'] == 'both') & (~trial2['V34_y'].isin(['02001', '02002', '02003', '14202']))] 

    # 공시가격 다세대 덮어씌우기
    upd = t_check[t_check['공동주택구분명'] == '다세대']['V0_y'].to_list()
    upd_tg = trial1['V0_y'].isin(upd)
    trial1.loc[upd_tg, 'V34_y'] = '02002'
    trial1.loc[upd_tg, 'V35_y'] = '연립주택'
    
    # 공시가격 빌라 덮어씌우기
    upd = t_check[t_check['공동주택구분명'] == '빌라']['V0_y'].to_list()
    upd_tg = trial1['V0_y'].isin(upd)
    trial1.loc[upd_tg, 'V34_y'] = '02002'
    trial1.loc[upd_tg, 'V35_y'] = '연립주택'
    
    # 공시가격 아파트 덮어씌우기
    upd = t_check[t_check['공동주택구분명'] == '아파트']['V0_y'].to_list()
    upd_tg = trial1['V0_y'].isin(upd)
    trial1.loc[upd_tg, 'V34_y'] = '02001'
    trial1.loc[upd_tg, 'V35_y'] = '아파트'
    
    # 오피스텔 기준시가 추가
    ofgp = pd.read_csv(of_path + '오피스텔_pk_식별.txt', sep = '|', encoding = 'cp949', dtype = str, usecols = ['JPK', '상가종류코드'])
    trial2 = pd.merge(trial1, ofgp, left_on = 'V0_y', right_on = 'JPK', how = 'left', indicator = True)
    t_check = trial2[(trial2['_merge'] == 'both') & (~trial2['V34_y'].isin(['02001', '02002', '02003', '14202']))]
    
    # 기준시가 오피스텔 덮어씌우기
    upd = t_check['V0_y'].to_list()
    upd_tg = trial1['V0_y'].isin(upd)
    trial1.loc[upd_tg, 'V34_y'] = '14202'
    trial1.loc[upd_tg, 'V35_y'] = '오피스텔'

    # jpk 단위 주용도가 있어야 된다는 지용SM의 요청에 의해 추가. 2023.10.23
    mid_jpk_result = trial1[['V0_x', 'V0_y','V34_y', 'V35_y', '전유_주용도코드_원천', '전유_주용도명_원천']]
    mid_jpk_result.columns = ['ppk', 'jpk', '전유_주용도코드', '전유_주용도명', '전유_주용도코드_원천', '전유_주용도명_원천']
    mid_jpk_result = mid_jpk_result.drop_duplicates(['ppk', 'jpk'], keep = 'first')

    jy_cat_result = pd.concat([jy_cat_result, mid_jpk_result], axis = 0, ignore_index = True)
    
    mid_jpk_result.to_csv(j_sv_path + sido['sido_ind'][i] + '_' + sido['sido_nm_2'][i] + '.txt', sep = '|', encoding = 'cp949', index = False)
    
    # 연립주택과 다세대주택을 하나로 계산
    trial1.drop(['전유_주용도명_원천', '전유_주용도코드_원천'], axis = 1, inplace = True)
    
    trial1['V34_ori'] = trial1['V34_y']
    trial1['V35_ori'] = trial1['V35_y']
    trial1['V34_y'] = trial1['V34_y'].str.replace('02003', '02002')
    trial1['V35_y'] = trial1['V35_y'].str.replace('다세대주택', '연립주택')
        
    # ========================= 건물 대표 주용도 정하기 =========================
    building = trial1[trial1['V34_y'].notnull()].reset_index(drop = True)  

    building['area_sum'] = building.groupby(['V0_x', 'V34_y', 'V35_y'])['V37'].transform('sum')
    building['area_sum1'] = building['area_sum'].apply(lambda x : round(x, 2))
    building['pp_cnt'] = building.groupby(['V0_x', 'V34_y', 'V35_y'])['V0_y'].transform('count')
    building['prior'] = building['V35_y'].replace({'아파트' : '00001', '오피스텔' : '00002', '연립주택' : '00003'}) # '다세대주택' : '00003'

    target = ~building['prior'].isin(['00001', '00002', '00003'])
    building.loc[target, 'prior'] = building.loc[target, 'V34_y']
    
    building1 = building.sort_values(['V0_x', 'area_sum1', 'pp_cnt', 'prior'], ascending = [True, False, False, True])

    building_result = building1.drop_duplicates('V0_x', keep = 'first')
    building_result = building_result[['V0_x', 'V34_ori', 'V35_ori']]
    building_result.columns = ['ppk', '건물_대표_주용도코드', '건물_대표_주용도명']

    # ========================= 주택 대표 주용도 정하기 =========================
    house = trial1[(trial1['V34_ori'].isin(['02001', '02002', '02003', '14202'])) & (trial1['V34_ori'].notnull())].reset_index(drop = True)

    house['area_sum'] = house.groupby(['V0_x', 'V34_y', 'V35_y'])['V37'].transform('sum')
    house['area_sum1'] = house['area_sum'].apply(lambda x : round(x, 2))
    house['pp_cnt'] = house.groupby(['V0_x', 'V34_y', 'V35_y'])['V0_y'].transform('count')
    house['prior'] = house['V35_y'].replace({'아파트' : '00001', '오피스텔' : '00002', '연립주택' : '00003'}) # '다세대주택' : '00003'
    
    target = ~house['prior'].isin(['00001', '00002', '00003'])
    house.loc[target, 'prior'] = house.loc[target, 'V34_y']
    
    house1 = house.sort_values(['V0_x', 'area_sum1', 'pp_cnt', 'prior'], ascending = [True, False, False, True])
    
    house_result = house1.drop_duplicates('V0_x', keep = 'first')
    house_result = house_result[['V0_x', 'V34_ori', 'V35_ori']]
    house_result.columns = ['ppk', '주택_대표_주용도코드', '주택_대표_주용도명']
    
    # ======================== 비주택 대표 주용도 정하기 ========================
    # 이렇게 한다는 것은 단독, 다가구, 다중을 비주택으로 본다는 건데.. 집합건축물을 분류하는 데 있어 전유에 저 세가지 용도가 있다는 것은 이론적으로 불가하니까 그냥 비주택으로 간주.
    non_house = trial1[(~trial1['V34_ori'].isin(['02001', '02002', '02003', '14202'])) & (trial1['V34_ori'].notnull())].reset_index(drop = True) # '01000', '01001', '01002', '01003'

    non_house['area_sum'] = non_house.groupby(['V0_x', 'V34_y', 'V35_y'])['V37'].transform('sum')
    non_house['area_sum1'] = non_house['area_sum'].apply(lambda x : round(x, 2))
    non_house['pp_cnt'] = non_house.groupby(['V0_x', 'V34_y', 'V35_y'])['V0_y'].transform('count')
    non_house['prior'] = non_house['V35_y'].replace({'아파트' : '00001', '오피스텔' : '00002', '연립주택' : '00003'}) # '다세대주택' : '00003'
    
    target = ~non_house['prior'].isin(['00001', '00002', '00003'])
    non_house.loc[target, 'prior'] = non_house.loc[target, 'V34_y']
    
    non_house1 = non_house.sort_values(['V0_x', 'area_sum1', 'pp_cnt', 'prior'], ascending = [True, False, False, True])
    
    non_house_result = non_house1.drop_duplicates('V0_x', keep = 'first')
    non_house_result = non_house_result[['V0_x', 'V34_ori', 'V35_ori']]
    non_house_result.columns = ['ppk', '비주택_대표_주용도코드', '비주택_대표_주용도명']

    # 매칭
    mpjb = trial[['V0_x', 'V34_x', 'V35_x']].drop_duplicates().rename(columns = {'V0_x' : 'ppk', 'V34_x' : '표제부_주용도코드', 'V35_x' : '표제부_주용도명'})    
    jiphap_final = reduce(lambda x, y : pd.merge(x, y, on = 'ppk', how = 'left'), [mpjb, building_result, house_result, non_house_result])
    
    # 결과물 저장
    jiphap_final.to_csv(sv_path1 + sido['sido_ind'][i] + '_' + sido['sido_nm_2'][i] + '.txt', sep = '|', encoding = 'cp949', index = False)
    
    # 전국 결과물
    final_result_all = pd.concat([final_result_all, jiphap_final], axis = 0, ignore_index = True)
     
    # 시도별 결과 요약
    
    if ilban['V0'].nunique() + trial['V0_x'].nunique() == pjb['V0'].nunique() : 
        
        print('\n')
        print('========== ' + sido['sido_nm'][i] + ' // 전체 표제부 PPK 총 ' + str(pjb['V0'].nunique()) + ' 개    (A) ==========', file=open(log_file, 'a'))
        print('========== ' + sido['sido_nm'][i] + ' // 일반건축물 간주 PPK 총 ' + str(ilban['V0'].nunique()) + ' 개 (B) ==========', file=open(log_file, 'a'))
        print('========== ' + sido['sido_nm'][i] + ' // 집합건축물 간주 PPK 총 ' + str(trial['V0_x'].nunique()) + ' 개 (C) ==========', file=open(log_file, 'a'))  
        print('========== ' + sido['sido_nm'][i] + ' // 집합건축물 주택분류 비율 총 ' + str(round(trial['V0_x'].nunique()/pjb['V0'].nunique()*100, 2)) + ' % (D/A) ==========', file=open(log_file, 'a'))
        print('                 ========== ' + ' A = B + C ' + ' ==========', file=open(log_file, 'a'))    
    
    else : 
        raise Exception('계산이 맞지 않음. 확인 요망!!')
    
    # ////////////////////////////////  일반건축물 분류  \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    # 일반건축물로 여겨지는 ppk 건과 전유부 미매칭 건 합치기
    
    ilban_jiphap = ilban[ilban['V1'] == '2'] # 기본개요 또는 전유부와 매칭이 안된 집합건축물 --> 아파트의 부속건축물(경비실, 주차장 따위)이 대부분
    ilban_jiphap1 = pd.merge(ilban_jiphap[['V0']], pjb[['V0', 'V34', 'V35']], on = 'V0', how = 'left')
    ilban_jiphap1['탈락사유'] = '집합 일반건축물'
    ilban_jiphap1.columns = ['ppk', '표제부_주용도코드', '표제부_주용도명', '탈락사유']    

    single = ilban.drop(ilban_jiphap.index, axis = 0).reset_index(drop = True)

    # 층별개요와 매칭  
    single_fl = pd.merge(single, fl, left_on = 'V0', right_on = 'F0', how = 'left', indicator = True)
    
    # 층별개요와 매칭 안된 ppk 건
    no_floor = single_fl[single_fl['_merge'] == 'left_only'] 
    no_floor = pd.merge(no_floor[['V0']], pjb[['V0', 'V34', 'V35']], on = 'V0', how = 'left')
    no_floor['탈락사유'] = '층별개요 미매칭'
    no_floor.columns = ['ppk', '표제부_주용도코드', '표제부_주용도명', '탈락사유']    

    # 주택 분류 안 되는 건들
    no_cat_result_all = pd.concat([no_cat_result_all, ilban_jiphap1, no_floor], axis = 0, ignore_index = True)

    single_fl = single_fl[single_fl['_merge'] == 'both'] 
    single_fl['F28'] = single_fl['F28'].astype('float64')
    
    # ========================= 건물 대표 주용도 정하기 =========================
    building = single_fl[single_fl['F26'].notnull()].reset_index(drop = True)  
    
    building['area_sum'] = building.groupby(['V0', 'F25', 'F26'])['F28'].transform('sum')
    building['area_sum1'] = building['area_sum'].apply(lambda x : round(x, 2))
    building['pp_cnt'] = building.groupby(['V0', 'F25', 'F26'])['V0'].transform('count')
    building['prior'] = building['F25']
    
    building1 = building.sort_values(['V0', 'area_sum1', 'pp_cnt', 'prior'], ascending = [True, False, False, True])
    
    building_result = building1.drop_duplicates('V0', keep = 'first')
    building_result = building_result[['V0', 'F25', 'F26']]
    building_result.columns = ['ppk', '건물_대표_주용도코드', '건물_대표_주용도명']
    
    # ========================= 주택 대표 주용도 정하기 =========================
    house = single_fl[single_fl['F26'].notnull()].reset_index(drop = True)  
    house1 = house[house['F25'].isin(['01000', '01001', '01002', '01003'])]
    
    house1['area_sum'] = house1.groupby(['V0', 'F25', 'F26'])['F28'].transform('sum')
    house1['area_sum1'] = house1['area_sum'].apply(lambda x : round(x, 2))
    house1['pp_cnt'] = house1.groupby(['V0', 'F25', 'F26'])['V0'].transform('count')
    house1['prior'] = house1['F25']
    
    house2 = house1.sort_values(['V0', 'area_sum1', 'pp_cnt', 'prior'], ascending = [True, False, False, True])
    
    house_result = house2.drop_duplicates('V0', keep = 'first')
    house_result = house_result[['V0', 'F25', 'F26']]
    house_result.columns = ['ppk', '주택_대표_주용도코드', '주택_대표_주용도명']
    
    # ========================= 주택 외 대표 주용도 정하기 =========================
    non_house = single_fl[single_fl['F26'].notnull()].reset_index(drop = True)  
    non_house1 = non_house[~non_house['F25'].isin(['01000', '01001', '01002', '01003'])]
    
    non_house1['area_sum'] = non_house1.groupby(['V0', 'F25', 'F26'])['F28'].transform('sum')
    non_house1['area_sum1'] = non_house1['area_sum'].apply(lambda x : round(x, 2))
    non_house1['pp_cnt'] = non_house1.groupby(['V0', 'F25', 'F26'])['V0'].transform('count')
    non_house1['prior'] = non_house1['F25']
    
    non_house2 = non_house1.sort_values(['V0', 'area_sum1', 'pp_cnt', 'prior'], ascending = [True, False, False, True])
    
    non_house_result = non_house2.drop_duplicates('V0', keep = 'first')
    non_house_result = non_house_result[['V0', 'F25', 'F26']]
    non_house_result.columns = ['ppk', '비주택_대표_주용도코드', '비주택_대표_주용도명']
    
    # 매칭
    mpjb = single_fl[['V0', 'V34', 'V35']].drop_duplicates().rename(columns = {'V0' : 'ppk', 'V34' : '표제부_주용도코드', 'V35' : '표제부_주용도명'})    
    ilban_final = reduce(lambda x, y : pd.merge(x, y, on = 'ppk', how = 'left'), [mpjb, building_result, house_result, non_house_result])
    
    # 결과물 저장
    ilban_final.to_csv(sv_path2 + sido['sido_ind'][i] + '_' + sido['sido_nm_2'][i] + '.txt', sep = '|', encoding = 'cp949', index = False)
    
    # 전국 결과물
    final_result_all1 = pd.concat([final_result_all1, ilban_final], axis = 0, ignore_index = True)
    
    if ilban_jiphap.shape[0] + no_floor.shape[0] + ilban_final.shape[0] == ilban.shape[0] :
        
        print('\n')
        print('========== ' + sido['sido_nm'][i] + ' // 일반건축물 간주 PPK 총 ' + str(ilban['V0'].nunique()) + ' 개    (A) ==========', file = open(log_file1, 'a'))
        print('========== ' + sido['sido_nm'][i] + ' // 집합 부속건축물 PPK 총 ' + str(ilban_jiphap1['ppk'].nunique()) + ' 개 (B) ==========', file = open(log_file1, 'a'))
        print('========== ' + sido['sido_nm'][i] + ' // 층별개요 미매칭 PPK 총 ' + str(no_floor['ppk'].nunique()) + ' 개 (C) ==========', file = open(log_file1, 'a'))
        print('========== ' + sido['sido_nm'][i] + ' // 일반건축물 분류 PPK 총 ' + str(ilban_final['ppk'].nunique()) + ' 개 (D) ==========', file = open(log_file1, 'a'))
        print('                 ========== ' + ' A = B + C + D' + ' ==========\n', file=open(log_file1, 'a'))    
    
    else : 
        raise Exception('계산이 맞지 않음. 확인 요망!!')
        
final_result_all.to_csv(sv_path1 + '전국.txt', sep = '|', encoding = 'cp949', index = False)
final_result_all1.to_csv(sv_path2 + '전국.txt', sep = '|', encoding = 'cp949', index = False)
jy_cat_result.to_csv(j_sv_path + '전국.txt', sep = '|', encoding = 'cp949', index = False)
no_cat_result_all.to_csv(sv_path3 + '전국.txt', sep = '|', encoding = 'cp949', index = False)
    
#%% 그룹핑 테이블로 단지형, 나홀로 판별

bld = pd.read_csv(sv_path1 + '전국.txt', sep = '|', encoding = 'cp949', dtype = str)
ilban = pd.read_csv(sv_path2 + '전국.txt', sep = '|', encoding = 'cp949', dtype = str)
jy_cat = pd.read_csv(j_sv_path + '전국.txt', sep = '|', encoding = 'cp949', dtype = str)
no_cat = pd.read_csv(sv_path3 + '전국.txt', sep = '|', encoding = 'cp949', dtype = str, usecols = ['ppk' ,'표제부_주용도코드', '표제부_주용도명'])

pjb = pd.read_csv(pjb_path + 'mart_djy_03.txt', sep = '|', encoding = 'cp949', dtype = str, header = None, usecols = [0])

# 그룹키 가져오기
connection = create_engine('postgresql://postgres:' + quote_plus('') + '@:/')
group = pd.read_sql(f"select complex_key, ppk from 건축물_대장.tb_complex_key where standard_date ='{stym}';", connection)
# group = pd.read_sql(f"select group_id as complex_key, ppk from general.tb_group_key where standard_ym ='{stym}';", connection)

assert type(group['complex_key'][0]) == str
assert len(group) > 0

bld1 = pd.merge(bld, group, on = 'ppk', how = 'left')

# 단지형, 나홀로아파트 판별용 아파트 세대수 계산
hh = jy_cat[jy_cat['전유_주용도코드'] == '02001']
pjb_hh = hh.groupby('ppk')['jpk'].nunique().reset_index(name = '아파트_세대수')

bld2 = pd.merge(bld1[['complex_key', 'ppk', '주택_대표_주용도명']], pjb_hh, on = 'ppk', how = 'inner')

bld2['아파트_건물수'] = bld2[bld2['주택_대표_주용도명'] == '아파트'].groupby('complex_key')['ppk'].transform('count')
bld2['아파트_건물수_통일'] = bld2.groupby('complex_key')['아파트_건물수'].transform('max')
bld2['아파트_건물수_통일'] = bld2['아파트_건물수_통일'].fillna(0)

bld2['아파트_건물수'] = np.where(bld2['아파트_건물수'].isna(), bld2['아파트_건물수_통일'], bld2['아파트_건물수'])
bld2['아파트_총세대수'] = bld2.groupby('complex_key')['아파트_세대수'].transform('sum')

bld2['cat'] = ''

danji_target = (bld2['아파트_건물수'] >= 3) | (bld2['아파트_총세대수'] >= 50)
bld2.loc[danji_target, 'cat'] = '단지형아파트'

holo_target = (bld2['아파트_건물수'] <= 2) & (bld2['아파트_총세대수'] < 50)
bld2.loc[holo_target, 'cat'] = '나홀로아파트'

# 단지형, 나홀로 분류 결과
dj_hl_cat = bld2[['ppk', 'cat']]

# 표제부 분류와, 전유부 분류 매칭
holo_dj_cat = pd.merge(jy_cat, dj_hl_cat, on = 'ppk', how = 'left')

apt_target = holo_dj_cat['전유_주용도코드'] == '02001'
holo_dj_cat.loc[apt_target, '전유_주용도명'] = holo_dj_cat.loc[apt_target, 'cat']
holo_dj_cat['전유_주용도명'] = holo_dj_cat['전유_주용도명'].str.replace('연립주택', '연립다세대').str.replace('다세대주택', '연립다세대')

jy_final = holo_dj_cat[['ppk', 'jpk', '전유_주용도코드_원천', '전유_주용도명_원천', '전유_주용도코드', '전유_주용도명']]

# 집합건축물, 일반건축물, 이상치, 전유부 분류 통합 결과물
total = pd.concat([bld, ilban, no_cat], axis = 0, ignore_index = True)

# 마지막 정제 / postgresql insert 용
total_df = pd.merge(total, jy_final, on = 'ppk', how = 'left')
total_df1 = total_df.copy()

# 주택 대표 주용도가 단독, 다가구, 다중으로 뽑힌 것 중에서 jpk 값이 있는 건이 없어야 함
dd_check = total_df[total_df['주택_대표_주용도명'].isin(['단독주택', '다가구주택', '다중주택']) & total_df['jpk'].notnull()]
assert len(dd_check) == 0

dd_target = total_df['주택_대표_주용도명'].isin(['단독주택', '다가구주택', '다중주택'])
total_df.loc[dd_target, '전유_주용도명'] = total_df.loc[dd_target, '주택_대표_주용도명']

# 층별개요 미매칭된 표제부 주용도가 단독다가구인 건에 대해서 주택분류를 표제부 주용도를 따름
dd_target1 = total_df['표제부_주용도명'].isin(['단독주택', '다가구주택', '다중주택']) & total_df['건물_대표_주용도명'].isna()
total_df.loc[dd_target1, '전유_주용도명'] = total_df.loc[dd_target1, '표제부_주용도명']

jy_target = ~total_df['전유_주용도명'].isin(['단지형아파트', '나홀로아파트', '연립다세대', '오피스텔', '단독주택', '다가구주택', '다중주택'])
total_df.loc[jy_target, '전유_주용도명'] = None

# jpk 값이 있는데 전유부 주용도 원천이 다가구주택인 것에 대한 처리 / ex) ppk : 41480-100359658, jpk : 41480-100362842
jy_target1 = total_df['전유_주용도명'].isin(['단독주택', '다가구주택', '다중주택']) & total_df['jpk'].notnull()
total_df.loc[jy_target1, '전유_주용도명'] = None

# jpk 값이 있으면서 표제부 주용도가 단독주택인 건의 주택분류 중 단독주택 없어야 함
dd_check1 = list(total_df[(total_df['jpk'].notnull()) & (total_df['표제부_주용도명'] == '단독주택')]['전유_주용도명'].unique())
assert '단독주택' not in dd_check1

total_df['전유_주용도명'].unique() # 주택 7종 들어가 있는지 확인

total_df['standard_ym'] = stym
total_df = total_df.replace(np.nan, None)
total_df['jpk'] = total_df['jpk'].fillna('none')

total_df = total_df[['ppk', 'jpk', 'standard_ym', '표제부_주용도코드', '표제부_주용도명', '전유_주용도코드_원천', '전유_주용도명_원천', '건물_대표_주용도코드', 
                     '건물_대표_주용도명', '주택_대표_주용도코드', '주택_대표_주용도명', '비주택_대표_주용도코드', '비주택_대표_주용도명', '전유_주용도명']]

total_df.columns = ['ppk', 'jpk', 'standard_ym', 'purpose_code_title', 'purpose_name_title', 'purpose_code_private_public_area', 
                    'purpose_name_private_public_area', 'building_purpose_code', 'building_purpose_name', 'house_purpose_code', 
                    'house_purpose_name', 'non_house_purpose_code', 'non_house_purpose_name', 'house_division_name']

total_df.to_csv(rst_path + 'total_df_psql_' + month + '.txt', sep = '|', encoding = 'utf-8', index = False)

# 오라클 insert용
dd_target11 = total_df1['표제부_주용도명'].isin(['단독주택', '다가구주택', '다중주택']) & total_df1['건물_대표_주용도명'].isna()
total_df1.loc[dd_target11, '주택_대표_주용도명'] = total_df1.loc[dd_target11, '표제부_주용도명']

# jpk 값이 있는데 층별개요 미매칭되고 표제부 주용도가 단독주택인 것에 대한 처리 / ex) ppk : 41590-100349419, jpk : 41590-100349420
jy_target11 = total_df1['표제부_주용도명'].isin(['단독주택', '다가구주택', '다중주택']) & total_df1['jpk'].notnull()
total_df1.loc[jy_target11, '주택_대표_주용도명'] = None

jy_target3 = ~total_df1['전유_주용도명'].isin(['단지형아파트', '나홀로아파트', '연립다세대', '오피스텔'])
total_df1.loc[jy_target3, '전유_주용도명'] = None

total_df1['STANDARD_YM'] = stym
total_df1 = total_df1.replace(np.nan, None)
total_df1['jpk'] = total_df1['jpk'].fillna('none')

total_df1 = total_df1[['STANDARD_YM', 'ppk', 'jpk', '표제부_주용도코드', '표제부_주용도명', '건물_대표_주용도코드', '건물_대표_주용도명', 
                       '주택_대표_주용도코드', '주택_대표_주용도명', '비주택_대표_주용도코드', '비주택_대표_주용도명', '전유_주용도명']]

total_df1.columns = ['STANDARD_YM', 'PPK', 'JPK', 'HEADLINE_MAIN_PURPOSE_CODE', 'HEADLINE_MAIN_PURPOSE_NAME', 'BUILDING_MAIN_PURPOSE_CODE', 
                     'BUILDING_MAIN_PURPOSE_NAME', 'HOUSE_MAIN_PURPOSE_CODE', 'HOUSE_MAIN_PURPOSE_NAME', 'NON_HOUSE_MAIN_PURPOSE_CODE', 
                     'NON_HOUSE_MAIN_PURPOSE_NAME', 'EXCLUSIVE_MAIN_PURPOSE_NAME']

total_df1.to_csv(rst_path + 'total_df_oracle_' + month + '.txt', sep = '|', encoding = 'utf-8', index = False)

# 표제부 불러와서 개수 비교해보기
assert pjb.shape[0] == total_df['ppk'].nunique()

''' #################### 결과물 postgresql insert #################### '''

user = ''
password = quote('')
host = ''
port = 5432
dbname = ''

postgres_engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')

postgres_engine.execute(f'truncate table 건축물_대장.tb_house_division_{stym};')

conn = postgres_engine.raw_connection()
copy_from_file(conn, '건축물_대장', 'tb_house_division', rst_path + 'total_df_psql_' + month + '.txt', '|', 'utf-8', opened = False, header = True)

''' #################### 결과물 oracle insert #################### '''
    
oracle_engine = create_engine('oracle+cx_oracle://:' + quote_plus('') + '@:1521/')
conn = oracle_engine.raw_connection()

# insert 앞서 해당월 데이터 존재하면 삭제
del_sql = f"DELETE FROM TB_NEW_HOUSE_DIVISION_V2 WHERE STANDARD_YM = '{stym}'"
with conn.cursor() as cursor:
    cursor.execute(del_sql)
conn.commit()

oracle_insert(conn, 'admin.tb_new_house_division_v2', chunker(total_df1, 100000))

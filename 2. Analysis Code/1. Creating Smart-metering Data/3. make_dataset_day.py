### Code : 스마트미터링 자료 시, 일, 월 단위 생성 - 일
### Writer : Donghyeon Kim
### Date : 2022.07.03

# 0. 라이브러리 실행
from pathlib import Path
import os
import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime, date
import holidays

# 1. 파일의 상위-상위 경로 설정
def get_project_root() -> Path:
    return Path(__file__).parent.parent

# 2. data 폴더 안에 사용자 이름 획득
def get_name_root():
    root = get_project_root()
    folder_root = os.path.join(root, 'data') # 루트 + 'data' 경로 설정
    user_name = os.listdir(folder_root) # 경로 안에 있는 사용자명
    return user_name

# 3. 주말 여부 확인 함수
def check_weekend(date):
    weekday = date.weekday()
    if weekday <= 4:
        return False
    if weekday > 4:
        return True

# 4. 공휴일 여부 체크
kr_holidays = holidays.KR()
def check_holiday(date):
    mydate = date
    if mydate in kr_holidays:
        return True
    if mydate not in kr_holidays:
        return False

# 5. 사용자 이름 폴더 안에 파일 하나씩 적용 - 최종 데이터프레임에 값 채우기
# 태양광 사용 가구
def get_value_on_use_df():
    root = get_project_root() # 루트
    user_name = get_name_root() # 사용자명
    folder_root = os.path.join(root, 'data_hour') # 폴더 경로

    print('----------------------------------')
    print('분석할 xlsx 파일 로딩 시작')

    dir_file = os.path.join(folder_root, 'hour_data.xlsx') # 파일 경로
    rawdata = pd.read_excel(dir_file)

    print('분석할 xlsx 파일 로드 완료')
    print('----------------------------------')

    # 태양광 사용 가구 리스트
    solar_use = ['오OO', '박OO', '이OO', '유OO', '임OO', '김OO', '이OO', '최OO', '오OO', '최OO', '김OO',
                 '고OO', '송OO', '이OO', '변OO', '서OO', '민OO', '조OO', '임OO', '윤OO']

    for i in range(len(user_name)):

        # 결과 Dictionary 생성
        data_time = {}
        data_time['id_hh'] = [] # 가구 ID
        data_time['id_hs'] = [] # 주택 ID
        data_time['PV'] = []  # 태양광 설치 여부
        data_time['year'] = []  # 연도
        data_time['month'] = []  # 월
        data_time['day'] = []  # 일
        data_time['weekend'] = []  # 주말 여부
        data_time['holiday'] = []  # 공휴일 여부
        data_time['consumption'] = []  # 전력 소비량
        data_time['production'] = []  # 전력 생산량
        data_time['export'] = []  # 전력 수출량
        data_time['import'] = []  # 수전량

        if user_name[i] not in solar_use: # 태양광 미사용 가구는 본 코드에서 생략함.
            continue

        print(f'{user_name[i]} 태양광 사용 가구 dataset 생성 시작')

        cond = (rawdata.owner == user_name[i])
        rawdata_filter = rawdata[cond]

        # Dictionary에 값 대입을 위한 필요한 변수 값 도출
        # 1) 연도 변수 설정
        u_year = rawdata_filter.year.unique()

        # 2) 연도 - 월 - 일 필터링 후 일일 단위 데이터 수집
        for y in u_year:
            date_cond1 = (rawdata_filter.year == y)
            day_filter1 = rawdata_filter[date_cond1]
            u_month = day_filter1.month.unique()

            for m in u_month:
                date_cond2 = (day_filter1.month == m)
                day_filter2 = day_filter1[date_cond2]
                u_day = day_filter2.day.unique()

                for d in u_day:
                    date_cond3 = (day_filter2.day == d)
                    day_filter3 = day_filter2[date_cond3]

                    consum_ = sum(day_filter3['import']) # 일일 그리드 소비 -> 수전 전력량
                    yield_ = sum(day_filter3['production']) # 일일 에너지 수율 -> 자가 발전량(전력 생산량)
                    export_ = sum(day_filter3['export']) # 일일 수출 된 에너지 -> 에너지 소비량
                    time_total = sum(day_filter3['consumption']) # 전력 소비량(consumption)
                    consum_self = yield_ - export_ # 자가 발전량(전력 생산량) 중 자가발전소비량

                    idhh = rawdata_filter.id_hh.unique().tolist()[0] # 가구 ID
                    idhs = rawdata_filter.id_hs.unique().tolist()[0] # 주택 ID

                    # 값 대입
                    data_time['id_hh'].append(idhh)
                    data_time['id_hs'].append(idhs)
                    data_time['PV'].append(1)
                    data_time['year'].append(y)
                    data_time['month'].append(m)
                    data_time['day'].append(d)

                    if check_weekend(date(y, m, d)) == True:
                        data_time['weekend'].append(1)
                    else:
                        data_time['weekend'].append(0)

                    if check_holiday(date(y, m, d)) == True:
                        data_time['holiday'].append(1)
                    else:
                        data_time['holiday'].append(0)

                    data_time['consumption'].append(time_total)
                    data_time['production'].append(yield_)
                    data_time['export'].append(export_)
                    data_time['import'].append(consum_)

            data_frame_time = pd.DataFrame(data_time)

        df_root = os.path.join(root, 'result_by_user')
        if not os.path.isdir(df_root):
            os.makedirs(df_root)

        xlsx_name = df_root + '/' + f'{user_name[i]}_dataset_day.xlsx'
        data_frame_time.to_excel(xlsx_name, sheet_name='day')
        print(f'{user_name[i]} 태양광 사용 가구 dataset 생성 완료')

    print("태양광 사용 가구 dataset 모두 생성 완료")
    return

# 6-2. 사용자 이름 폴더 안에 파일 하나씩 적용 - 최종 데이터프레임에 값 채우기
# 태양광 미사용 가구
def get_value_on_not_df():
    root = get_project_root() # 루트
    user_name = get_name_root() # 사용자명
    folder_root = os.path.join(root, 'data_hour') # 폴더 경로

    print('----------------------------------')
    print('분석할 xlsx 파일 로딩 시작')

    dir_file = os.path.join(folder_root, 'hour_data.xlsx') # 파일 경로
    rawdata = pd.read_excel(dir_file)

    print('분석할 xlsx 파일 로드 완료')
    print('----------------------------------')

    # 태양광 사용 가구 리스트
    solar_use = ['오OO', '박OO', '이OO', '유OO', '임OO', '김OO', '이OO', '최OO', '오OO', '최OO', '김OO',
                 '고OO', '송OO', '이OO', '변OO', '서OO', '민OO', '조OO', '임OO', '윤OO']

    for i in range(len(user_name)):

        # 결과 Dictionary 생성
        data_time = {}
        data_time['id_hh'] = []  # 가구 ID
        data_time['id_hs'] = []  # 주택 ID
        data_time['PV'] = []  # 태양광 설치 여부
        data_time['year'] = []  # 연도
        data_time['month'] = []  # 월
        data_time['day'] = []  # 일
        data_time['weekend'] = []  # 주말 여부
        data_time['holiday'] = []  # 공휴일 여부
        data_time['consumption'] = []  # 전력 소비량
        data_time['production'] = []  # 전력 생산량
        data_time['export'] = []  # 전력 수출량
        data_time['import'] = []  # 수전량

        if user_name[i] in solar_use:  # 태양광 사용 가구는 본 코드에서 생략함.
            continue

        print(f'{user_name[i]} 태양광 미사용 가구 dataset 생성 시작')

        cond = (rawdata.owner == user_name[i])
        rawdata_filter = rawdata[cond]

        # Dictionary에 값 대입을 위한 필요한 변수 값 도출
        # 1) 연도 변수 설정
        u_year = rawdata_filter.year.unique()

        # 2) 연도 - 월 - 일 필터링 후 일일 단위 데이터 수집
        for y in u_year:
            date_cond1 = (rawdata_filter.year == y)
            day_filter1 = rawdata_filter[date_cond1]
            u_month = day_filter1.month.unique()

            for m in u_month:
                date_cond2 = (day_filter1.month == m)
                day_filter2 = day_filter1[date_cond2]
                u_day = day_filter2.day.unique()

                for d in u_day:
                    date_cond3 = (day_filter2.day == d)
                    day_filter3 = day_filter2[date_cond3]

                    consum_ = sum(day_filter3['import'])  # 일일 그리드 소비 -> 수전 전력량
                    yield_ = sum(day_filter3['production'])  # 일일 에너지 수율 -> 자가 발전량(전력 생산량)
                    export_ = sum(day_filter3['export'])  # 일일 수출 된 에너지 -> 에너지 소비량
                    time_total = sum(day_filter3['consumption'])  # 전력 소비량(consumption)
                    # consum_self = yield_ - export_  # 자가 발전량(전력 생산량) 중 자가발전소비량

                    idhh = rawdata_filter.id_hh.unique().tolist()[0]  # 가구 ID
                    idhs = rawdata_filter.id_hs.unique().tolist()[0]  # 주택 ID

                    # 값 대입
                    data_time['id_hh'].append(idhh)
                    data_time['id_hs'].append(idhs)
                    data_time['PV'].append(0)
                    data_time['year'].append(y)
                    data_time['month'].append(m)
                    data_time['day'].append(d)

                    if check_weekend(date(y, m, d)) == True:
                        data_time['weekend'].append(1)
                    else:
                        data_time['weekend'].append(0)

                    if check_holiday(date(y, m, d)) == True:
                        data_time['holiday'].append(1)
                    else:
                        data_time['holiday'].append(0)

                    data_time['consumption'].append(time_total)
                    data_time['production'].append(yield_)
                    data_time['export'].append(export_)
                    data_time['import'].append(consum_)

            data_frame_time = pd.DataFrame(data_time)

        df_root = os.path.join(root, 'result_by_user')
        if not os.path.isdir(df_root):
            os.makedirs(df_root)

        xlsx_name = df_root + '/' + f'{user_name[i]}_dataset_day.xlsx'
        data_frame_time.to_excel(xlsx_name, sheet_name='day')
        print(f'{user_name[i]} 태양광 미사용 가구 dataset 생성 완료')

    print("태양광 미사용 가구 dataset 모두 생성 완료")
    return

# 실행부(Pycharm : Ctrl + Shift + F10)
if __name__ == '__main__':
    tmp = get_value_on_use_df()
    print(tmp)

    tmp2 = get_value_on_not_df()
    print(tmp2)

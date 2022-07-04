### Code : 스마트미터링 자료 연도별 생성(추가 작업)
### Writer : Donghyeon Kim
### Date : 2022.07.05

## 방법론
# rawdata에서 각 minute별로 consumption 계산.
# 다음, 각 사용자마다 data를 concatenate 진행.
# 그 후, 연도별로 max - min 계산 진행 -> consumption, import, production, export 모두 적용.

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

# 3. 연도별 주말 횟수
def weekend_num(dy):
    result = 0
    for dm in range(1, 13):
        if dm in [1, 3, 5, 7, 8, 10, 12]:
            for dd in range(1, 32):
                mydate = date(dy, dm, dd)
                weekday = mydate.weekday()
                if weekday <= 4:
                    continue
                if weekday > 4:
                    result += 1
        elif dm in [4, 6, 9, 11]:
            for dd in range(1, 31):
                mydate = date(dy, dm, dd)
                weekday = mydate.weekday()
                if weekday <= 4:
                    continue
                if weekday > 4:
                    result += 1
        else:
            for dd in range(1, 29):
                mydate = date(dy, dm, dd)
                weekday = mydate.weekday()
                if weekday <= 4:
                    continue
                if weekday > 4:
                    result += 1
    return result

# 4. 연도별 공휴일 횟수
kr_holidays = holidays.KR()
def holiday_num(dy):
    result = 0
    for dm in range(1, 13):
        if dm in [1, 3, 5, 7, 8, 10, 12]:
            for dd in range(1, 32):
                mydate = date(dy, dm, dd)
                if mydate in kr_holidays:
                    result += 1
                if mydate not in kr_holidays:
                    continue
        elif dm in [4, 6, 9, 11]:
            for dd in range(1, 31):
                mydate = date(dy, dm, dd)
                if mydate in kr_holidays:
                    result += 1
                if mydate not in kr_holidays:
                    continue
        else:
            for dd in range(1, 29):
                mydate = date(dy, dm, dd)
                if mydate in kr_holidays:
                    result += 1
                if mydate not in kr_holidays:
                    continue
    return result

# 5. 날짜 리스트 생성(2021/1/1 ~ 2021/8/8)
def makedate(year = 2021):
    mydate = []
    for m in range(1, 9):
        if m in [1, 3, 5, 7]:
            for d in range(1, 32):
                mydate.append(f'{year}/{m}/{d}')
        elif m in [4, 6]:
            for d in range(1, 31):
                mydate.append(f'{year}/{m}/{d}')
        elif m == 2:
            for d in range(1, 29):
                mydate.append(f'{year}/{m}/{d}')
        else:
            for d in range(1, 9):
                mydate.append(f'{year}/{m}/{d}')
    return mydate

# 6-1. 사용자 이름 폴더 안에 파일 하나씩 적용 - 최종 데이터프레임에 값 채우기
# 태양광 사용 가구
def get_value_on_use_df():
    root = get_project_root() # 루트
    user_name = get_name_root() # 사용자명

    # 태양광 사용 가구 리스트('윤OO' 제외)
    solar_use = ['오OO', '박OO', '이OO', '유OO', '임OO', '김OO', '이OO', '최OO', '오OO', '최OO', '김OO',
                 '고OO', '송OO', '이OO', '변OO', '서OO', '민OO', '조OO', '임OO']

    for i in range(len(user_name)):

        # 결과 Dictionary 생성
        data_time = {}
        data_time['PV'] = [] # 태양광 설치 여부
        data_time['year'] = [] # 연도
        data_time['weekend'] = [] # 주말 횟수
        data_time['holiday'] = [] # 공휴일 횟수
        data_time['consumption'] = [] # 전력 소비량
        data_time['production'] = [] # 전력 생산량
        data_time['export'] = [] # 전력 수출량
        data_time['import'] = [] # 수전량

        if user_name[i] == '윤OO': # 윤OO data는 변수명이 다르므로 본 코드에서는 생략함.
            continue

        if user_name[i] not in solar_use: # 태양광 미사용 가구일 경우, 본 코드에서는 실행하지 않음.
            continue

        print(f'{user_name[i]} 태양광 사용 가구 dataset 생성 시작')
        folder_root = os.path.join(root, 'data', user_name[i]) # 폴더 경로
        dir_list = os.listdir(folder_root) # 경로 안에 있는 파일 리스트

        for j in range(len(dir_list)):
            dir_file = os.path.join(folder_root, dir_list[j])
            file_name = dir_list[j]
            user_data = pd.read_excel(dir_file)

            date_val = f'{file_name[0:4]}/{int(file_name[5:7])}/{int(file_name[8:10])}' # 날짜 형식(yyyy/mm/dd)
            for idx, tmp in enumerate(user_data.iloc[:, 0]): # index와 내용(tmp) 추출
                if str(tmp) in ['nan', '마지막 업데이트']: # 리스트 안에 해당되는 내용일 경우 코드 실행 생략
                    continue
                if tmp.split(' ')[0] == f"{date_val}":
                    break

            rawdata = pd.read_excel(dir_file, header=idx)  # idx로 header 재설정

            rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True) # 컬럼 이름 변경
            rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정
            rawdata['year'] = rawdata['date'].dt.year
            rawdata['month'] = rawdata['date'].dt.month
            rawdata['day'] = rawdata['date'].dt.day
            rawdata['hour'] = rawdata['date'].dt.hour
            rawdata['minute'] = rawdata['date'].dt.minute

            if "Grid Consumption(kWh)" in rawdata.columns:
                rawdata.rename(columns={"Grid Consumption(kWh)": '그리드 소비(kWh)'}, inplace=True) # 컬럼 이름 변경
                rawdata.rename(columns={"Exported Energy(kWh)": '수출 된 에너지(kWh)'}, inplace=True) # 컬럼 이름 변경
                rawdata.rename(columns={"Yield Energy(kWh)": '에너지 수율(kWh)'}, inplace=True) # 컬럼 이름 변경

            rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str) # 타입 문자열로 변경
            rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str) # 타입 문자열로 변경
            rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].astype(str) # 타입 문자열로 변경

            rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
            rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
            rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경

            # 각 minute 별로 consumption 계산(= import + production - export, = 그리드 소비 + 에너지 수율 - 수출 된 에너지)
            rawdata['전력 소비량(kWh)'] = rawdata['그리드 소비(kWh)'] + rawdata['에너지 수율(kWh)'] - rawdata['수출 된 에너지(kWh)']

            # 'yyyy-mm-dd' Data 모두 합치기
            if j == 0:
                final_rawdata = rawdata
            else:
                final_rawdata = pd.concat([final_rawdata, rawdata])

            # Dictionary에 값 대입을 위한 필요한 변수 값 도출
            if j == (len(dir_list) - 1): # j가 last index인 경우에만 아래 코드 실행
                u_year = final_rawdata.year.unique().tolist()

                for y in u_year:
                    date_cond1 = (final_rawdata.year == y)
                    day_filter1 = final_rawdata[date_cond1]

                    # 1) 그리드 소비(수전 전력량) : import
                    after_c = day_filter1['그리드 소비(kWh)'].max()
                    before_c = day_filter1['그리드 소비(kWh)'].min()
                    consum_ = after_c - before_c # 일일 그리드 소비 -> 수전 전력량

                    # 2) 에너지 수율(자가 발전량) : production
                    after_y = day_filter1['에너지 수율(kWh)'].max()
                    before_y = day_filter1['에너지 수율(kWh)'].min()
                    yield_ = after_y - before_y # 일일 에너지 수율 -> 자가 발전량(전력 생산량)

                    # 3) 수출 된 에너지(에너지 소비량) : export
                    after_e = day_filter1['수출 된 에너지(kWh)'].max()
                    before_e = day_filter1['수출 된 에너지(kWh)'].min()
                    export_ = after_e - before_e # 일일 수출 된 에너지 -> 에너지 소비량

                    # 4) 전력 소비량(total) : consumption
                    after_t = day_filter1['전력 소비량(kWh)'].max()
                    before_t = day_filter1['전력 소비량(kWh)'].min()
                    time_total = after_t - before_t # 일일 전력 소비량

                    # 5) 자가발전소비량
                    # consum_self = yield_ - export_ # 자가 발전량(전력 생산량) 중 자가발전소비량

                    # 값 대입
                    data_time['PV'].append(1)
                    data_time['year'].append(y)
                    data_time['weekend'].append(weekend_num(y))
                    data_time['holiday'].append(holiday_num(y))
                    data_time['consumption'].append(time_total)
                    data_time['production'].append(yield_)
                    data_time['export'].append(export_)
                    data_time['import'].append(consum_)
                data_frame_time = pd.DataFrame(data_time)
            if not j == (len(dir_list) - 1):
                continue

        df_root = os.path.join(root, 'result_by_user')
        if not os.path.isdir(df_root):
            os.makedirs(df_root)

        xlsx_name = df_root + '/' + f'{user_name[i]}_dataset_year.xlsx'
        data_frame_time.to_excel(xlsx_name, sheet_name='year')
        print(f'{user_name[i]} 태양광 사용 가구 dataset 생성 완료')

    print("태양광 사용 가구 dataset 모두 생성 완료")
    return

# 6-2. 사용자 이름 폴더 안에 파일 하나씩 적용 - 최종 데이터프레임에 값 채우기
# 태양광 미사용 가구
def get_value_on_not_df():
    root = get_project_root() # 루트
    user_name = get_name_root() # 사용자명

    # 태양광 사용 가구 리스트('윤OO' 제외)
    solar_use = ['오OO', '박OO', '이OO', '유OO', '임OO', '김OO', '이OO', '최OO', '오OO', '최OO', '김OO',
                 '고OO', '송OO', '이OO', '변OO', '서OO', '민OO', '조OO', '임OO']

    for i in range(len(user_name)):

        # 결과 Dictionary 생성
        data_time = {}
        data_time['PV'] = []  # 태양광 설치 여부
        data_time['year'] = []  # 연도
        data_time['weekend'] = []  # 주말 횟수
        data_time['holiday'] = []  # 공휴일 횟수
        data_time['consumption'] = []  # 전력 소비량
        data_time['production'] = []  # 전력 생산량
        data_time['export'] = []  # 전력 수출량
        data_time['import'] = []  # 수전량

        if user_name[i] == '윤OO':  # 윤OO data는 변수명이 다르므로 본 코드에서는 생략함.
            continue

        if user_name[i] in solar_use:  # 태양광 사용 가구일 경우, 본 코드에서는 실행하지 않음.
            continue

        print(f'{user_name[i]} 태양광 미사용 가구 dataset 생성 시작')
        folder_root = os.path.join(root, 'data', user_name[i])  # 폴더 경로
        dir_list = os.listdir(folder_root)  # 경로 안에 있는 파일 리스트

        for j in range(len(dir_list)):
            dir_file = os.path.join(folder_root, dir_list[j])
            file_name = dir_list[j]
            user_data = pd.read_excel(dir_file)

            date_val = f'{file_name[0:4]}/{int(file_name[5:7])}/{int(file_name[8:10])}'  # 날짜 형식(yyyy/mm/dd)
            for idx, tmp in enumerate(user_data.iloc[:, 0]):  # index와 내용(tmp) 추출
                if str(tmp) in ['nan', '마지막 업데이트']:  # 리스트 안에 해당되는 내용일 경우 코드 실행 생략
                    continue
                if tmp.split(' ')[0] == f"{date_val}":
                    break

            rawdata = pd.read_excel(dir_file, header=idx)  # idx로 header 재설정

            rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True)  # 컬럼 이름 변경
            rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S')  # 날짜 포맷 설정
            rawdata['year'] = rawdata['date'].dt.year
            rawdata['month'] = rawdata['date'].dt.month
            rawdata['day'] = rawdata['date'].dt.day
            rawdata['hour'] = rawdata['date'].dt.hour
            rawdata['minute'] = rawdata['date'].dt.minute

            if "Grid Consumption(kWh)" in rawdata.columns:
                rawdata.rename(columns={"Grid Consumption(kWh)": '그리드 소비(kWh)'}, inplace=True) # 컬럼 이름 변경
                rawdata.rename(columns={"Exported Energy(kWh)": '수출 된 에너지(kWh)'}, inplace=True)
                # rawdata.rename(columns={"Yield Energy(kWh)": '에너지 수율(kWh)'}, inplace=True)

            rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str) # 타입 문자열로 변경
            rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str)
            # rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].astype(str)

            rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
            rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32')
            # rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].str.replace(',', '').astype('float32')

            # 각 minute 별로 consumption 계산(= import + production - export, = 그리드 소비 + 에너지 수율 - 수출 된 에너지)
            rawdata['전력 소비량(kWh)'] = rawdata['그리드 소비(kWh)'] - rawdata['수출 된 에너지(kWh)']

            # 'yyyy-mm-dd' Data 모두 합치기
            if j == 0:
                final_rawdata = rawdata
            else:
                final_rawdata = pd.concat([final_rawdata, rawdata])

            # Dictionary에 값 대입을 위한 필요한 변수 값 도출
            if j == (len(dir_list) - 1):  # j가 last index인 경우에만 아래 코드 실행
                u_year = final_rawdata.year.unique().tolist()

                for y in u_year:
                    date_cond1 = (final_rawdata.year == y)
                    day_filter1 = final_rawdata[date_cond1]

                    # 1) 그리드 소비(수전 전력량) : import
                    after_c = day_filter1['그리드 소비(kWh)'].max()
                    before_c = day_filter1['그리드 소비(kWh)'].min()
                    consum_ = after_c - before_c  # 일일 그리드 소비 -> 수전 전력량

                    # 2) 에너지 수율(자가 발전량) : production
                    # after_y = day_filter1['에너지 수율(kWh)'].max()
                    # before_y = day_filter1['에너지 수율(kWh)'].min()
                    # yield_ = after_y - before_y  # 일일 에너지 수율 -> 자가 발전량(전력 생산량)

                    # 3) 수출 된 에너지(에너지 소비량) : export
                    after_e = day_filter1['수출 된 에너지(kWh)'].max()
                    before_e = day_filter1['수출 된 에너지(kWh)'].min()
                    export_ = after_e - before_e  # 일일 수출 된 에너지 -> 에너지 소비량

                    # 4) 전력 소비량(total) : consumption
                    after_t = day_filter1['전력 소비량(kWh)'].max()
                    before_t = day_filter1['전력 소비량(kWh)'].min()
                    time_total = after_t - before_t  # 일일 전력 소비량

                    # 5) 자가발전소비량
                    # consum_self = yield_ - export_ # 자가 발전량(전력 생산량) 중 자가발전소비량

                    # 값 대입
                    data_time['PV'].append(0)
                    data_time['year'].append(y)
                    data_time['weekend'].append(weekend_num(y))
                    data_time['holiday'].append(holiday_num(y))
                    data_time['consumption'].append(time_total)
                    data_time['production'].append(np.nan)
                    data_time['export'].append(export_)
                    data_time['import'].append(consum_)
                data_frame_time = pd.DataFrame(data_time)
            if not j == (len(dir_list) - 1):
                continue

        df_root = os.path.join(root, 'result_by_user')
        if not os.path.isdir(df_root):
            os.makedirs(df_root)

        xlsx_name = df_root + '/' + f'{user_name[i]}_dataset_year.xlsx'
        data_frame_time.to_excel(xlsx_name, sheet_name='year')
        print(f'{user_name[i]} 태양광 미사용 가구 dataset 생성 완료')

    print("태양광 미사용 가구 dataset 모두 생성 완료")
    return

# 6-3. 사용자 이름 폴더 안에 파일 하나씩 적용 - 최종 데이터프레임에 값 채우기
# 태양광 사용 가구 중 special case - '윤OO'
# 2021-08-08까지 변수 '에너지 수율' -> 태양광 발전, 변수 '부하 에너지' -> 그리드 소비를 의미함.
# 2021-08-09부터는 다른 data와 마찬가지로 형태가 동일하므로 코드 그대로 적용 가능.
def get_value_on_use_df_special():
    root = get_project_root() # 루트
    user_name = get_name_root() # 사용자명

    # 태양광 사용 가구 리스트('윤OO' 제외)
    # solar_use = ['오OO', '박OO', '이OO', '유OO', '임OO', '김OO', '이OO', '최OO', '오OO', '최OO', '김OO',
    #             '고OO', '송OO', '이OO', '변OO', '서OO', '민OO', '조OO', '임OO']

    for i in range(len(user_name)):

        # 결과 Dictionary 생성
        data_time = {}
        data_time['PV'] = [] # 태양광 설치 여부
        data_time['year'] = [] # 연도
        data_time['weekend'] = [] # 주말 횟수
        data_time['holiday'] = [] # 공휴일 횟수
        data_time['consumption'] = [] # 전력 소비량
        data_time['production'] = [] # 전력 생산량
        data_time['export'] = [] # 전력 수출량
        data_time['import'] = [] # 수전량

        if user_name[i] != '윤OO': # 윤OO data가 아니면 본 코드는 실행하지 않음.
            continue

        print(f'{user_name[i]} 태양광 사용 가구 dataset 생성 시작')
        folder_root = os.path.join(root, 'data', user_name[i]) # 폴더 경로
        dir_list = os.listdir(folder_root) # 경로 안에 있는 파일 리스트

        for j in range(len(dir_list)):
            dir_file = os.path.join(folder_root, dir_list[j])
            file_name = dir_list[j]
            user_data = pd.read_excel(dir_file)

            date_val = f'{file_name[0:4]}/{int(file_name[5:7])}/{int(file_name[8:10])}' # 날짜 형식(yyyy/mm/dd)
            for idx, tmp in enumerate(user_data.iloc[:, 0]): # index와 내용(tmp) 추출
                if str(tmp) in ['nan', '마지막 업데이트']: # 리스트 안에 해당되는 내용일 경우 코드 실행 생략
                    continue
                if tmp.split(' ')[0] == f"{date_val}":
                    break

            rawdata = pd.read_excel(dir_file, header=idx) # idx로 header 재설정

            date_2021_list = makedate()
            if date_val in date_2021_list: # 2021 날짜 리스트(2021/1/1 ~ 2021/8/8)에 해당되면 코드 실행
                rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True) # 컬럼 이름 변경
                rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정
                rawdata['year'] = rawdata['date'].dt.year
                rawdata['month'] = rawdata['date'].dt.month
                rawdata['day'] = rawdata['date'].dt.day
                rawdata['hour'] = rawdata['date'].dt.hour
                rawdata['minute'] = rawdata['date'].dt.minute

                if "Grid Consumption(kWh)" in rawdata.columns:
                    rawdata.rename(columns={"Grid Consumption(kWh)": '그리드 소비(kWh)'}, inplace=True) # 컬럼 이름 변경
                    # rawdata.rename(columns={"Exported Energy(kWh)": '수출 된 에너지(kWh)'}, inplace=True)
                    rawdata.rename(columns={"Yield Energy(kWh)": '에너지 수율(kWh)'}, inplace=True)

                if "부하 에너지(kWh)" in rawdata.columns:
                    rawdata.rename(columns={"부하 에너지(kWh)": '그리드 소비(kWh)'}, inplace=True)

                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str) # 타입 문자열로 변경
                # rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str)
                rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].astype(str)

                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
                # rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32')
                rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].str.replace(',', '').astype('float32')

                # 각 minute 별로 consumption 계산(= import + production - export, = 그리드 소비 + 에너지 수율 - 수출 된 에너지)
                rawdata['전력 소비량(kWh)'] = rawdata['그리드 소비(kWh)'] + rawdata['에너지 수율(kWh)']

                # 'yyyy-mm-dd' Data 모두 합치기
                if j == 0:
                    final_rawdata = rawdata
                else:
                    final_rawdata = pd.concat([final_rawdata, rawdata])

                # Dictionary에 값 대입을 위한 필요한 변수 값 도출
                if j == (len(dir_list) - 1):  # j가 last index인 경우에만 아래 코드 실행
                    u_year = final_rawdata.year.unique().tolist()

                    for y in u_year:
                        date_cond1 = (final_rawdata.year == y)
                        day_filter1 = final_rawdata[date_cond1]

                        # 1) 그리드 소비(수전 전력량) : import
                        after_c = day_filter1['그리드 소비(kWh)'].max()
                        before_c = day_filter1['그리드 소비(kWh)'].min()
                        consum_ = after_c - before_c  # 일일 그리드 소비 -> 수전 전력량

                        # 2) 에너지 수율(자가 발전량) : production
                        after_y = day_filter1['에너지 수율(kWh)'].max()
                        before_y = day_filter1['에너지 수율(kWh)'].min()
                        yield_ = after_y - before_y  # 일일 에너지 수율 -> 자가 발전량(전력 생산량)

                        # 3) 수출 된 에너지(에너지 소비량) : export
                        # after_e = day_filter1['수출 된 에너지(kWh)'].max()
                        # before_e = day_filter1['수출 된 에너지(kWh)'].min()
                        # export_ = after_e - before_e  # 일일 수출 된 에너지 -> 에너지 소비량

                        # 4) 전력 소비량(total) : consumption
                        after_t = day_filter1['전력 소비량(kWh)'].max()
                        before_t = day_filter1['전력 소비량(kWh)'].min()
                        time_total = after_t - before_t  # 일일 전력 소비량

                        # 5) 자가발전소비량
                        # consum_self = yield_ - export_ # 자가 발전량(전력 생산량) 중 자가발전소비량

                        # 값 대입
                        data_time['PV'].append(1)
                        data_time['year'].append(y)
                        data_time['weekend'].append(weekend_num(y))
                        data_time['holiday'].append(holiday_num(y))
                        data_time['consumption'].append(time_total)
                        data_time['production'].append(yield_)
                        data_time['export'].append(np.nan)
                        data_time['import'].append(consum_)
                    data_frame_time = pd.DataFrame(data_time)
                if not j == (len(dir_list) - 1):
                    continue
            else:
                rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True) # 컬럼 이름 변경
                rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정
                rawdata['year'] = rawdata['date'].dt.year
                rawdata['month'] = rawdata['date'].dt.month
                rawdata['day'] = rawdata['date'].dt.day
                rawdata['hour'] = rawdata['date'].dt.hour
                rawdata['minute'] = rawdata['date'].dt.minute

                if "Grid Consumption(kWh)" in rawdata.columns:
                    rawdata.rename(columns={"Grid Consumption(kWh)": '그리드 소비(kWh)'}, inplace=True) # 컬럼 이름 변경
                    rawdata.rename(columns={"Exported Energy(kWh)": '수출 된 에너지(kWh)'}, inplace=True)
                    rawdata.rename(columns={"Yield Energy(kWh)": '에너지 수율(kWh)'}, inplace=True)

                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str) # 타입 문자열로 변경
                rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str)
                rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].astype(str)

                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
                rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32')
                rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].str.replace(',', '').astype('float32')

                # 각 minute 별로 consumption 계산(= import + production - export, = 그리드 소비 + 에너지 수율 - 수출 된 에너지)
                rawdata['전력 소비량(kWh)'] = rawdata['그리드 소비(kWh)'] + rawdata['에너지 수율(kWh)'] - rawdata['수출 된 에너지(kWh)']

                # 'yyyy-mm-dd' Data 모두 합치기
                if j == 0:
                    final_rawdata = rawdata
                else:
                    final_rawdata = pd.concat([final_rawdata, rawdata])

                # Dictionary에 값 대입을 위한 필요한 변수 값 도출
                if j == (len(dir_list) - 1):  # j가 last index인 경우에만 아래 코드 실행
                    u_year = final_rawdata.year.unique().tolist()

                    for y in u_year:
                        date_cond1 = (final_rawdata.year == y)
                        day_filter1 = final_rawdata[date_cond1]

                        # 1) 그리드 소비(수전 전력량) : import
                        after_c = day_filter1['그리드 소비(kWh)'].max()
                        before_c = day_filter1['그리드 소비(kWh)'].min()
                        consum_ = after_c - before_c  # 일일 그리드 소비 -> 수전 전력량

                        # 2) 에너지 수율(자가 발전량) : production
                        after_y = day_filter1['에너지 수율(kWh)'].max()
                        before_y = day_filter1['에너지 수율(kWh)'].min()
                        yield_ = after_y - before_y  # 일일 에너지 수율 -> 자가 발전량(전력 생산량)

                        # 3) 수출 된 에너지(에너지 소비량) : export
                        after_e = day_filter1['수출 된 에너지(kWh)'].max()
                        before_e = day_filter1['수출 된 에너지(kWh)'].min()
                        export_ = after_e - before_e  # 일일 수출 된 에너지 -> 에너지 소비량

                        # 4) 전력 소비량(total) : consumption
                        after_t = day_filter1['전력 소비량(kWh)'].max()
                        before_t = day_filter1['전력 소비량(kWh)'].min()
                        time_total = after_t - before_t  # 일일 전력 소비량

                        # 5) 자가발전소비량
                        # consum_self = yield_ - export_ # 자가 발전량(전력 생산량) 중 자가발전소비량

                        # 값 대입
                        data_time['PV'].append(1)
                        data_time['year'].append(y)
                        data_time['weekend'].append(weekend_num(y))
                        data_time['holiday'].append(holiday_num(y))
                        data_time['consumption'].append(time_total)
                        data_time['production'].append(yield_)
                        data_time['export'].append(export_)
                        data_time['import'].append(consum_)
                    data_frame_time = pd.DataFrame(data_time)
                if not j == (len(dir_list) - 1):
                    continue

        df_root = os.path.join(root, 'result_by_user')
        if not os.path.isdir(df_root):
            os.makedirs(df_root)

        xlsx_name = df_root + '/' + f'{user_name[i]}_dataset_year.xlsx'
        data_frame_time.to_excel(xlsx_name, sheet_name='year')
        print(f'{user_name[i]} 태양광 사용 가구 dataset 생성 완료')

    print("태양광 사용 가구 dataset special case 모두 생성 완료")
    return

# 실행부(Pycharm : Ctrl + Shift + F10)
if __name__ == '__main__':
    tmp = get_value_on_use_df()
    print(tmp)

    tmp2 = get_value_on_not_df()
    print(tmp2)

    tmp3 = get_value_on_use_df_special()
    print(tmp3)

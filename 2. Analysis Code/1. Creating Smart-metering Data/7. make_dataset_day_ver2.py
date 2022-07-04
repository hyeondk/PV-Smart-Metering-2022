### Code : 스마트미터링 자료 시, 일, 월 단위 생성 - 일 version2
### Writer : Donghyeon Kim
### Date : 2022.07.04

## 방법론
# rawdata에서 각 minute별로 consumption 계산.
# 그 후, 일별로 max - min 계산 진행 -> consumption, import, production, export 모두 적용.
# 이러면 rawdata 전체로 max - min으로 진행했을 시 나오는 'consumption 값이 마이너스(음수, -)가 나오는 문제'는 해결될 것으로 보임.

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
        data_time['month'] = [] # 월
        data_time['day'] = [] # 일
        data_time['weekend'] = [] # 주말 여부
        data_time['holiday'] = [] # 공휴일 여부
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
            rawdata['전력 소비량(kWh)'] = rawdata.iloc[:, 4] + rawdata.iloc[:, 9] - rawdata.iloc[:, 5]

            # Dictionary에 값 대입을 위한 필요한 변수 값 도출
            # 1) 그리드 소비(수전 전력량) : import
            after_c = rawdata.iloc[:, 4].max()
            before_c = rawdata.iloc[:, 4].min()
            consum_ = after_c - before_c # 일일 그리드 소비 -> 수전 전력량

            # 2) 에너지 수율(자가 발전량) : production
            after_y = rawdata.iloc[:, 9].max()
            before_y = rawdata.iloc[:, 9].min()
            yield_ = after_y - before_y # 일일 에너지 수율 -> 자가 발전량(전력 생산량)

            # 3) 수출 된 에너지(에너지 소비량) : export
            after_e = rawdata.iloc[:, 5].max()
            before_e = rawdata.iloc[:, 5].min()
            export_ = after_e - before_e # 일일 수출 된 에너지 -> 에너지 소비량

            # 4) 전력 소비량(total) : consumption
            after_t = rawdata.iloc[:, -1].max()
            before_t = rawdata.iloc[:, -1].min()
            time_total = after_t - before_t # 일일 전력 소비량

            # 5) 자가발전소비량
            consum_self = yield_ - export_ # 자가 발전량(전력 생산량) 중 자가발전소비량

            # 6) 날짜
            y = rawdata.year.unique().tolist()[0]
            m = rawdata.month.unique().tolist()[0]
            d = rawdata.day.unique().tolist()[0]

            # 값 대입
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

        xlsx_name = df_root + '/' + f'{user_name[i]}_dataset_day_v2.xlsx'
        data_frame_time.to_excel(xlsx_name, sheet_name='day')
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
        data_time['PV'] = [] # 태양광 설치 여부
        data_time['year'] = [] # 연도
        data_time['month'] = [] # 월
        data_time['day'] = [] # 일
        data_time['weekend'] = [] # 주말 여부
        data_time['holiday'] = [] # 공휴일 여부
        data_time['consumption'] = [] # 전력 소비량
        data_time['production'] = [] # 전력 생산량
        data_time['export'] = [] # 전력 수출량
        data_time['import'] = [] # 수전량

        if user_name[i] == '윤OO': # 윤OO data는 변수명이 다르므로 본 코드에서는 생략함.
            continue

        if user_name[i] in solar_use: # 태양광 사용 가구일 경우, 본 코드에서는 실행하지 않음.
            continue

        print(f'{user_name[i]} 태양광 미사용 가구 dataset 생성 시작')
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
                # rawdata.rename(columns={"Yield Energy(kWh)": '에너지 수율(kWh)'}, inplace=True) # 컬럼 이름 변경

            rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str) # 타입 문자열로 변경
            rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str) # 타입 문자열로 변경
            # rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].astype(str) # 타입 문자열로 변경

            rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
            rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
            # rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경

            # 각 minute 별로 consumption 계산(= import + production - export, = 그리드 소비 + 에너지 수율 - 수출 된 에너지)
            rawdata['전력 소비량(kWh)'] = rawdata.iloc[:, 4] - rawdata.iloc[:, 5]

            # Dictionary에 값 대입을 위한 필요한 변수 값 도출
            # 1) 그리드 소비(수전 전력량) : import
            after_c = rawdata.iloc[:, 4].max()
            before_c = rawdata.iloc[:, 4].min()
            consum_ = after_c - before_c # 일일 그리드 소비 -> 수전 전력량

            # 2) 에너지 수율(자가 발전량) : production
            # after_y = rawdata.iloc[:, 9].max()
            # before_y = rawdata.iloc[:, 9].min()
            # yield_ = after_y - before_y # 일일 에너지 수율 -> 자가 발전량(전력 생산량)

            # 3) 수출 된 에너지(에너지 소비량) : export
            after_e = rawdata.iloc[:, 5].max()
            before_e = rawdata.iloc[:, 5].min()
            export_ = after_e - before_e # 일일 수출 된 에너지 -> 에너지 소비량

            # 4) 전력 소비량(total) : consumption
            after_t = rawdata.iloc[:, -1].max()
            before_t = rawdata.iloc[:, -1].min()
            time_total = after_t - before_t # 일일 전력 소비량

            # 5) 자가발전소비량
            # consum_self = yield_ - export_ # 자가 발전량(전력 생산량) 중 자가발전소비량

            # 6) 날짜
            y = rawdata.year.unique().tolist()[0]
            m = rawdata.month.unique().tolist()[0]
            d = rawdata.day.unique().tolist()[0]

            # 값 대입
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
            data_time['production'].append(np.nan)
            data_time['export'].append(export_)
            data_time['import'].append(consum_)

        data_frame_time = pd.DataFrame(data_time)

        df_root = os.path.join(root, 'result_by_user')
        if not os.path.isdir(df_root):
            os.makedirs(df_root)

        xlsx_name = df_root + '/' + f'{user_name[i]}_dataset_day_v2.xlsx'
        data_frame_time.to_excel(xlsx_name, sheet_name='day')
        print(f'{user_name[i]} 태양광 미사용 가구 dataset 생성 완료')

    print("태양광 미사용 가구 dataset 모두 생성 완료")
    return

# 6-3. 사용자 이름 폴더 안에 파일 하나씩 적용 - 최종 데이터프레임에 값 채우기
# 태양광 사용 가구 중 special case - '윤정자'
# 2021-08-08까지 변수 '에너지 수율' -> 태양광 발전, 변수 '부하 에너지' -> 그리드 소비를 의미함.
# 2021-08-09부터는 다른 data와 마찬가지로 형태가 동일하므로 코드 그대로 적용 가능.
def get_value_on_use_df_special():
    root = get_project_root() # 루트
    user_name = get_name_root() # 사용자명

    # 태양광 사용 가구 리스트('윤OO' 제외)
    solar_use = ['오OO', '박OO', '이OO', '유OO', '임OO', '김OO', '이OO', '최OO', '오OO', '최OO', '김OO',
                 '고OO', '송OO', '이OO', '변OO', '서OO', '민OO', '조OO', '임OO']

    # 태양광 사용 가구 special case : 윤OO
    solar_use_special = ['윤OO']

    for i in range(len(user_name)):

        # 결과 Dictionary 생성
        data_time = {}
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

        if user_name[i] != '윤OO': # 윤OO data가 아니면 본 코드 실행 안함.
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
                    rawdata.rename(columns={"Grid Consumption(kWh)": '그리드 소비(kWh)'}, inplace=True)
                    # rawdata.rename(columns={"Exported Energy(kWh)": '수출 된 에너지(kWh)'}, inplace = True)
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
                rawdata['전력 소비량(kWh)'] = rawdata.iloc[:, 8] + rawdata.iloc[:, 4]

                # Dictionary에 값 대입을 위한 필요한 변수 값 도출
                # 1) 그리드 소비(수전 전력량) : import
                after_c = rawdata.iloc[:, 8].max()
                before_c = rawdata.iloc[:, 8].min()
                consum_ = after_c - before_c  # 일일 그리드 소비 -> 수전 전력량

                # 2) 에너지 수율(자가 발전량) : production
                after_y = rawdata.iloc[:, 4].max()
                before_y = rawdata.iloc[:, 4].min()
                yield_ = after_y - before_y  # 일일 에너지 수율 -> 자가 발전량(전력 생산량)

                # 3) 수출 된 에너지(에너지 소비량) : export
                # after_e = rawdata.iloc[:, 5].max()
                # before_e = rawdata.iloc[:, 5].min()
                # export_ = after_e - before_e  # 일일 수출 된 에너지 -> 에너지 소비량

                # 4) 전력 소비량(total) : consumption
                after_t = rawdata.iloc[:, -1].max()
                before_t = rawdata.iloc[:, -1].min()
                time_total = after_t - before_t  # 일일 전력 소비량

                # 5) 자가발전소비량
                # consum_self = yield_ - export_  # 자가 발전량(전력 생산량) 중 자가발전소비량

                # 6) 날짜
                y = rawdata.year.unique().tolist()[0]
                m = rawdata.month.unique().tolist()[0]
                d = rawdata.day.unique().tolist()[0]

                # 값 대입
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
                data_time['export'].append(np.nan)
                data_time['import'].append(consum_)

                data_frame_time = pd.DataFrame(data_time)
            else:
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
                    rawdata.rename(columns={"Yield Energy(kWh)": '에너지 수율(kWh)'}, inplace=True)

                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str) # 타입 문자열로 변경
                rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str)
                rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].astype(str)

                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
                rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32')
                rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].str.replace(',', '').astype('float32')

                # 각 minute 별로 consumption 계산(= import + production - export, = 그리드 소비 + 에너지 수율 - 수출 된 에너지)
                rawdata['전력 소비량(kWh)'] = rawdata.iloc[:, 4] + rawdata.iloc[:, 9] - rawdata.iloc[:, 5]

                # Dictionary에 값 대입을 위한 필요한 변수 값 도출
                # 1) 그리드 소비(수전 전력량) : import
                after_c = rawdata.iloc[:, 4].max()
                before_c = rawdata.iloc[:, 4].min()
                consum_ = after_c - before_c  # 일일 그리드 소비 -> 수전 전력량

                # 2) 에너지 수율(자가 발전량) : production
                after_y = rawdata.iloc[:, 9].max()
                before_y = rawdata.iloc[:, 9].min()
                yield_ = after_y - before_y  # 일일 에너지 수율 -> 자가 발전량(전력 생산량)

                # 3) 수출 된 에너지(에너지 소비량) : export
                after_e = rawdata.iloc[:, 5].max()
                before_e = rawdata.iloc[:, 5].min()
                export_ = after_e - before_e  # 일일 수출 된 에너지 -> 에너지 소비량

                # 4) 전력 소비량(total) : consumption
                after_t = rawdata.iloc[:, -1].max()
                before_t = rawdata.iloc[:, -1].min()
                time_total = after_t - before_t  # 일일 전력 소비량

                # 5) 자가발전소비량
                consum_self = yield_ - export_  # 자가 발전량(전력 생산량) 중 자가발전소비량

                # 6) 날짜
                y = rawdata.year.unique().tolist()[0]
                m = rawdata.month.unique().tolist()[0]
                d = rawdata.day.unique().tolist()[0]

                # 값 대입
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

        xlsx_name = df_root + '/' + f'{user_name[i]}_dataset_day_v2.xlsx'
        data_frame_time.to_excel(xlsx_name, sheet_name='day')
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

### Code : Rawdata Interpolation(분 단위)
### Writer : Donghyeon Kim
### Date : 2022.07.11

## 데이터 형태 ##
# 태양광 사용 가구 : 20가구
# 태양광 미사용 가구 : 33가구
# 총 가구수 : 53가구

## 방법론 ##
# 해당 날짜에 0:00부터 23:59분까지 60분 X 24시간 = 1,440개의 행이 모두 존재한다 -> 완전한 데이터로 취급 / 이 외에는 불완전한 데이터로 취급.
# 완전한 데이터 -> linear interpolation method를 활용하여 보정 진행(단, 앞부분과 뒷부분 보정은 생략. 이는 모델을 통해 보정 예정.)
# 불완전한 데이터 -> interpolation 생략

## 참고 사항 ##
# 기존 data가 '.xls' 형태로 저장되어 있어 같은 형태로 저장하고자 함.
# 오류는 'FutureWarning'이므로, 실행하는데는 문제가 없음. 다만, 추후 본 코드를 활용하고자 할 경우에는 확장자를 '.xlsx'로 변경할 필요는 있음.

# 0. 라이브러리 실행
from pathlib import Path
import os
import pandas as pd
import numpy as np
import openpyxl
import xlwt
from datetime import datetime, date

# 1. 파일의 상위-상위 경로 설정
def get_project_root() -> Path:
    return Path(__file__).parent.parent

# 2. data 폴더 안에 사용자 이름 획득
def get_name_root():
    root = get_project_root()
    folder_root = os.path.join(root, 'data') # 루트 + 'data' 경로 설정
    user_name = os.listdir(folder_root) # 경로 안에 있는 사용자명
    return user_name

# 3. 날짜 리스트 생성(2021/1/1 ~ 2021/8/8)
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

# 4-1. 사용자 이름 폴더 안에 파일 하나씩 적용 - NA 값 대체
# 태양광 사용 가구
def get_value_on_use_df():
    root = get_project_root() # 루트
    user_name = get_name_root() # 사용자명

    # 태양광 사용 가구 리스트('윤OO' 제외)
    solar_use = ['오OO', '박OO', '이OO', '유OO', '임OO', '김OO', '이OO', '최OO', '오OO', '최OO', '김OO',
                 '고OO', '송OO', '이OO', '변OO', '서OO', '민OO', '조OO', '임OO']

    for i in range(len(user_name)):

        if user_name[i] == '윤OO': # 윤OO data는 변수명이 다르므로 본 코드를 실행하지 않음.
            continue

        if user_name[i] not in solar_use: # 태양광 미사용 가구 데이터는 본 코드를 실행하지 않음.
            continue

        print(f'{user_name[i]} 태양광 사용 가구 dataset 보정 시작')
        folder_root = os.path.join(root, 'data', user_name[i]) # 폴더 경로
        dir_list = os.listdir(folder_root) # 경로 안에 있는 파일 리스트

        for j in range(len(dir_list)):
            dir_file = os.path.join(folder_root, dir_list[j])
            file_name = dir_list[j]
            user_data = pd.read_excel(dir_file)

            date_val = f'{file_name[0:4]}/{int(file_name[5:7])}/{int(file_name[8:10])}' # 날짜 형식(yyyy/mm/dd)
            for idx, tmp in enumerate(user_data.iloc[:, 0]): # index와 내용(tmp) 추출
                if str(tmp) in ['nan', '마지막 업데이트']: # 리스트 안에 해당되는 내용일 경우 코드 실행하지 않음.
                    continue
                if tmp.split(' ')[0] == f"{date_val}":
                    break

            rawdata = pd.read_excel(dir_file, header=idx)  # idx로 header 재설정

            rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True) # 컬럼 이름 변경
            rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정
            #rawdata['year'] = rawdata['date'].dt.year
            #rawdata['month'] = rawdata['date'].dt.month
            #rawdata['day'] = rawdata['date'].dt.day
            #rawdata['hour'] = rawdata['date'].dt.hour
            #rawdata['minute'] = rawdata['date'].dt.minute

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

            # 1분 단위 데이터 보정(단위 변환 X / 1분 단위 데이터 그대로 유지)
            if len(rawdata) != 1440: # 해당 날짜 데이터를 불렀을 때, 행이 1440개가 되지 않으면 생략함.
                continue
            elif len(rawdata) == 1440:
                # Linear Interpolation
                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].fillna(rawdata['그리드 소비(kWh)'].interpolate())
                rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].fillna(rawdata['수출 된 에너지(kWh)'].interpolate())
                rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].fillna(rawdata['에너지 수율(kWh)'].interpolate())

                df_root = os.path.join(root, 'data_revised_use')
                if not os.path.isdir(df_root):
                    os.makedirs(df_root)

                df_root_user = os.path.join(df_root, user_name[i])
                if not os.path.isdir(df_root_user):
                    os.makedirs(df_root_user)

                xls_name = df_root_user + '\\' + file_name
                rawdata.to_excel(xls_name, sheet_name='revised', index=False)

        print(f'{user_name[i]} 태양광 사용 가구 dataset 보정 완료')

    print("태양광 사용 가구 dataset 모두 보정 완료")
    return

# 4-2. 사용자 이름 폴더 안에 파일 하나씩 적용 - NA 값 대체
# 태양광 미사용 가구
def get_value_on_not_df():
    root = get_project_root() # 루트
    user_name = get_name_root() # 사용자명

    # 태양광 사용 가구 리스트('윤OO' 제외)
    solar_use = ['오OO', '박OO', '이OO', '유OO', '임OO', '김OO', '이OO', '최OO', '오OO', '최OO', '김OO',
                 '고OO', '송OO', '이OO', '변OO', '서OO', '민OO', '조OO', '임OO']

    for i in range(len(user_name)):

        if user_name[i] == '윤OO': # 윤OO data는 변수명이 다르므로 본 코드를 실행하지 않음.
            continue

        if user_name[i] in solar_use: # 태양광 사용 가구 데이터는 본 코드를 실행하지 않음.
            continue

        print(f'{user_name[i]} 태양광 미사용 가구 dataset 보정 시작')
        folder_root = os.path.join(root, 'data', user_name[i]) # 폴더 경로
        dir_list = os.listdir(folder_root) # 경로 안에 있는 파일 리스트

        for j in range(len(dir_list)):
            dir_file = os.path.join(folder_root, dir_list[j])
            file_name = dir_list[j]
            user_data = pd.read_excel(dir_file)

            date_val = f'{file_name[0:4]}/{int(file_name[5:7])}/{int(file_name[8:10])}' # 날짜 형식(yyyy/mm/dd)
            for idx, tmp in enumerate(user_data.iloc[:, 0]): # index와 내용(tmp) 추출
                if str(tmp) in ['nan', '마지막 업데이트']: # 리스트 안에 해당되는 내용일 경우 코드 실행하지 않음.
                    continue
                if tmp.split(' ')[0] == f"{date_val}":
                    break

            rawdata = pd.read_excel(dir_file, header=idx)  # idx로 header 재설정

            rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True) # 컬럼 이름 변경
            rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정
            #rawdata['year'] = rawdata['date'].dt.year
            #rawdata['month'] = rawdata['date'].dt.month
            #rawdata['day'] = rawdata['date'].dt.day
            #rawdata['hour'] = rawdata['date'].dt.hour
            #rawdata['minute'] = rawdata['date'].dt.minute

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

            # 1분 단위 데이터 보정(단위 변환 X / 1분 단위 데이터 그대로 유지)
            if len(rawdata) != 1440: # 해당 날짜 데이터를 불렀을 때, 행이 1440개가 되지 않으면 생략함.
                continue
            elif len(rawdata) == 1440:
                # Linear Interpolation
                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].fillna(rawdata['그리드 소비(kWh)'].interpolate())
                rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].fillna(rawdata['수출 된 에너지(kWh)'].interpolate())
                # rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].fillna(rawdata['에너지 수율(kWh)'].interpolate())

                df_root = os.path.join(root, 'data_revised_not')
                if not os.path.isdir(df_root):
                    os.makedirs(df_root)

                df_root_user = os.path.join(df_root, user_name[i])
                if not os.path.isdir(df_root_user):
                    os.makedirs(df_root_user)

                xls_name = df_root_user + '\\' + file_name
                rawdata.to_excel(xls_name, sheet_name='revised', index=False)

        print(f'{user_name[i]} 태양광 미사용 가구 dataset 보정 완료')

    print("태양광 미사용 가구 dataset 모두 보정 완료")
    return

# 4-3. 사용자 이름 폴더 안에 파일 하나씩 적용 - NA 값 대체
# 태양광 사용 가구 중 special case - '윤OO'
# 2021-08-08까지 변수 '에너지 수율' -> 태양광 발전, 변수 '부하 에너지' -> 그리드 소비를 의미함.
# 2021-08-09부터는 다른 data와 마찬가지로 형태가 동일하므로 코드 그대로 적용 가능.
def get_value_on_use_df_special():
    root = get_project_root() # 루트
    user_name = get_name_root() # 사용자명

    for i in range(len(user_name)):

        if user_name[i] != '윤OO': # 윤OO data가 아닌 경우 본 코드를 실행하지 않음.
            continue

        print(f'{user_name[i]} 태양광 사용 가구 dataset 보정 시작')
        folder_root = os.path.join(root, 'data', user_name[i]) # 폴더 경로
        dir_list = os.listdir(folder_root) # 경로 안에 있는 파일 리스트

        for j in range(len(dir_list)):
            dir_file = os.path.join(folder_root, dir_list[j])
            file_name = dir_list[j]
            user_data = pd.read_excel(dir_file)

            date_val = f'{file_name[0:4]}/{int(file_name[5:7])}/{int(file_name[8:10])}' # 날짜 형식(yyyy/mm/dd)
            for idx, tmp in enumerate(user_data.iloc[:, 0]): # index와 내용(tmp) 추출
                if str(tmp) in ['nan', '마지막 업데이트']: # 리스트 안에 해당되는 내용일 경우 코드 실행하지 않음.
                    continue
                if tmp.split(' ')[0] == f"{date_val}":
                    break

            rawdata = pd.read_excel(dir_file, header=idx)  # idx로 header 재설정

            date_2021_list = makedate()
            if date_val in date_2021_list: # 2021 날짜 리스트(2021/1/1 ~ 2021/8/8)에 해당되면 코드 실행
                rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True) # 컬럼 이름 변경
                rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정
                # rawdata['year'] = rawdata['date'].dt.year
                # rawdata['month'] = rawdata['date'].dt.month
                # rawdata['day'] = rawdata['date'].dt.day
                # rawdata['hour'] = rawdata['date'].dt.hour
                # rawdata['minute'] = rawdata['date'].dt.minute

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

                # 1분 단위 데이터 보정(단위 변환 X / 1분 단위 데이터 그대로 유지)
                if len(rawdata) != 1440:  # 해당 날짜 데이터를 불렀을 때, 행이 1440개가 되지 않으면 생략함.
                    continue
                elif len(rawdata) == 1440:
                    # Linear Interpolation
                    rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].fillna(rawdata['그리드 소비(kWh)'].interpolate())
                    # rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].fillna(rawdata['수출 된 에너지(kWh)'].interpolate())
                    rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].fillna(rawdata['에너지 수율(kWh)'].interpolate())

                    df_root = os.path.join(root, 'data_revised_use')
                    if not os.path.isdir(df_root):
                        os.makedirs(df_root)

                    df_root_user = os.path.join(df_root, user_name[i])
                    if not os.path.isdir(df_root_user):
                        os.makedirs(df_root_user)

                    xls_name = df_root_user + '\\' + file_name
                    rawdata.to_excel(xls_name, sheet_name='revised', index=False)
            else:
                if date_val in date_2021_list: # 2021 날짜 리스트(2021/1/1 ~ 2021/8/8)에 해당되면 코드 실행
                    rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True) # 컬럼 이름 변경
                    rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정
                    # rawdata['year'] = rawdata['date'].dt.year
                    # rawdata['month'] = rawdata['date'].dt.month
                    # rawdata['day'] = rawdata['date'].dt.day
                    # rawdata['hour'] = rawdata['date'].dt.hour
                    # rawdata['minute'] = rawdata['date'].dt.minute

                    if "Grid Consumption(kWh)" in rawdata.columns:
                        rawdata.rename(columns={"Grid Consumption(kWh)": '그리드 소비(kWh)'}, inplace=True)  # 컬럼 이름 변경
                        # rawdata.rename(columns={"Exported Energy(kWh)": '수출 된 에너지(kWh)'}, inplace=True)
                        rawdata.rename(columns={"Yield Energy(kWh)": '에너지 수율(kWh)'}, inplace=True)

                    if "부하 에너지(kWh)" in rawdata.columns:
                        rawdata.rename(columns={"부하 에너지(kWh)": '그리드 소비(kWh)'}, inplace=True)

                    rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str)  # 타입 문자열로 변경
                    # rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str)
                    rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].astype(str)

                    rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype(
                        'float32')  # 반점 제거 & 타입 float32로 변경
                    # rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32')
                    rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].str.replace(',', '').astype('float32')

                    # 1분 단위 데이터 보정(단위 변환 X / 1분 단위 데이터 그대로 유지)
                    if len(rawdata) != 1440:  # 해당 날짜 데이터를 불렀을 때, 행이 1440개가 되지 않으면 생략함.
                        continue
                    elif len(rawdata) == 1440:
                        # Linear Interpolation
                        rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].fillna(rawdata['그리드 소비(kWh)'].interpolate())
                        # rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].fillna(rawdata['수출 된 에너지(kWh)'].interpolate())
                        rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].fillna(rawdata['에너지 수율(kWh)'].interpolate())

                        df_root = os.path.join(root, 'data_revised_use')
                        if not os.path.isdir(df_root):
                            os.makedirs(df_root)

                        df_root_user = os.path.join(df_root, user_name[i])
                        if not os.path.isdir(df_root_user):
                            os.makedirs(df_root_user)

                        xls_name = df_root_user + '\\' + file_name
                        rawdata.to_excel(xls_name, sheet_name='revised', index=False)
                else:
                    rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True) # 컬럼 이름 변경
                    rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정
                    # rawdata['year'] = rawdata['date'].dt.year
                    # rawdata['month'] = rawdata['date'].dt.month
                    # rawdata['day'] = rawdata['date'].dt.day
                    # rawdata['hour'] = rawdata['date'].dt.hour
                    # rawdata['minute'] = rawdata['date'].dt.minute

                    if "Grid Consumption(kWh)" in rawdata.columns:
                        rawdata.rename(columns={"Grid Consumption(kWh)": '그리드 소비(kWh)'}, inplace=True) # 컬럼 이름 변경
                        rawdata.rename(columns={"Exported Energy(kWh)": '수출 된 에너지(kWh)'}, inplace=True)
                        rawdata.rename(columns={"Yield Energy(kWh)": '에너지 수율(kWh)'}, inplace=True)

                    if "부하 에너지(kWh)" in rawdata.columns:
                        rawdata.rename(columns={"부하 에너지(kWh)": '그리드 소비(kWh)'}, inplace=True)

                    rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str) # 타입 문자열로 변경
                    rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str)
                    rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].astype(str)

                    rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
                    rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32')
                    rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].str.replace(',', '').astype('float32')

                    # 1분 단위 데이터 보정(단위 변환 X / 1분 단위 데이터 그대로 유지)
                    if len(rawdata) != 1440:  # 해당 날짜 데이터를 불렀을 때, 행이 1440개가 되지 않으면 생략함.
                        continue
                    elif len(rawdata) == 1440:
                        # Linear Interpolation
                        rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].fillna(rawdata['그리드 소비(kWh)'].interpolate())
                        rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].fillna(rawdata['수출 된 에너지(kWh)'].interpolate())
                        rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].fillna(rawdata['에너지 수율(kWh)'].interpolate())

                        df_root = os.path.join(root, 'data_revised_use')
                        if not os.path.isdir(df_root):
                            os.makedirs(df_root)

                        df_root_user = os.path.join(df_root, user_name[i])
                        if not os.path.isdir(df_root_user):
                            os.makedirs(df_root_user)

                        xls_name = df_root_user + '\\' + file_name
                        rawdata.to_excel(xls_name, sheet_name='revised', index=False)

        print(f'{user_name[i]} 태양광 사용 가구 dataset 보정 완료')

    print("태양광 사용 가구 dataset 모두 보정 완료")
    return

# 실행부(Pycharm : Ctrl + Shift + F10)
if __name__ == '__main__':
    tmp = get_value_on_use_df()
    print(tmp)

    tmp2 = get_value_on_not_df()
    print(tmp2)

    tmp3 = get_value_on_use_df_special()
    print(tmp3)

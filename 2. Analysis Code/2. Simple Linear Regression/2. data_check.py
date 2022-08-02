### Code : Rawdata Check(Before Regression Analysis)
### Writer : Donghyeon Kim
### Date : 2022.07.22

## 데이터 형태 ##
# 태양광 사용 가구 : 20가구
# 태양광 미사용 가구 : 33가구
# 총 가구수 : 53가구

## 방법론 ##
# 참고 : 회귀분석(Regression Analysis) 진행을 위한 데이터 사전 체크 작업임.
# 1. 1440개 행 데이터가 가장 많이 있는 가구 추출(1440 = 60분 X 24시간)
# 2. NA가 적은지도 체크

# 0. 라이브러리 실행
from pathlib import Path
import os
import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime, date
from sklearn.linear_model import LinearRegression

# 1. 파일의 상위-상위 경로 설정
def get_project_root() -> Path:
    return Path(__file__).parent.parent

##########
# 2-1. data 폴더 안에 태양광 사용 가구 이름 획득
def get_name_root_use():
    root = get_project_root()
    folder_root = os.path.join(root, 'data_revised_use') # 루트 + 'data_revised_use' 경로 설정
    user_name = os.listdir(folder_root) # 경로 안에 있는 사용자명
    return user_name

# 2-2. data 폴더 안에 태양광 미사용 가구 이름 획득
def get_name_root_not():
    root = get_project_root()
    folder_root = os.path.join(root, 'data_revised_not') # 루트 + 'data_revised_not' 경로 설정
    user_name = os.listdir(folder_root)
    return user_name
##########

##########
# 3-1. 1440개 행이 가장 많은 태양광 사용 가구 이름 추출
def get_household_use():
    root = get_project_root() # 루트
    user_name = get_name_root_use() # 사용자명
    user_len = [] # 사용자별 데이터 길이(엑셀 개수)

    for i in range(len(user_name)):
        folder_root = os.path.join(root, 'data_revised_use', user_name[i]) # 폴더 경로
        dir_list = os.listdir(folder_root) # 경로 안에 있는 파일 리스트
        user_len.append(len(dir_list)) # 파일 리스트 개수 카운트 -> 리스트에 추가

    user_result = user_name[np.argmax(user_len)]

    return user_result

# 3-2. 1440개 행이 가장 많은 태양광 미사용 가구 이름 추출
def get_household_not():
    root = get_project_root() # 루트
    user_name = get_name_root_not() # 사용자명
    user_len = [] # 사용자별 데이터 길이(엑셀 개수)

    for i in range(len(user_name)):
        folder_root = os.path.join(root, 'data_revised_not', user_name[i]) # 폴더 경로
        dir_list = os.listdir(folder_root) # 경로 안에 있는 파일 리스트
        user_len.append(len(dir_list)) # 파일 리스트 개수 카운트 -> 리스트에 추가

    user_result = user_name[np.argmax(user_len)]

    return user_result
##########

##########
# 4-1. 태양광 사용 가구 '김OO'에 대한 NA 체크
def get_data_na_use():
    root = get_project_root()
    user_name = get_household_use() # 가구명 : '김OO'
    folder_root = os.path.join(root, 'data_revised_all', user_name) # 폴더 경로
    dir_list = os.listdir(folder_root) # 경로 안에 있는 파일 리스트

    # NA 개수 확인을 위한 딕셔너리 생성
    na_len_dict = {}
    na_len_dict['그리드 소비'] = 0
    na_len_dict['수출 된 에너지'] = 0
    na_len_dict['에너지 수율'] = 0

    print(f'{user_name} 태양광 사용 가구 NA 체크 시작')

    for j in range(len(dir_list)):
        dir_file = os.path.join(folder_root, dir_list[j])
        file_name = dir_list[j]
        user_data = pd.read_excel(dir_file)

        if 'date' not in user_data.columns: # 컬럼에 'date'라는 변수가 없은 경우 -> Rawdata이므로 다시 읽어줘야 함.
            date_val = f'{file_name[0:4]}/{int(file_name[5:7])}/{int(file_name[8:10])}' # 날짜 형식(yyyy/mm/dd)
            for idx, tmp in enumerate(user_data.iloc[:, 0]): # index와 내용(tmp) 추출
                if str(tmp) in ['nan', '마지막 업데이트']: # 리스트 안에 해당되는 내용일 경우 코드 실행하지 않음.
                    continue
                if tmp.split(' ')[0] == f"{date_val}":
                    break

            rawdata = pd.read_excel(dir_file, header=idx) # idx로 header 재설정

            rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True) # 컬럼 이름 변경
            rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정

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

            na_len_dict['그리드 소비'] += rawdata['그리드 소비(kWh)'].isna().sum()
            na_len_dict['수출 된 에너지'] += rawdata['수출 된 에너지(kWh)'].isna().sum()
            na_len_dict['에너지 수율'] += rawdata['에너지 수율(kWh)'].isna().sum()
        else:
            rawdata = user_data # if문 안에 있는 데이터 변수명과 동일하게 설정

            na_len_dict['그리드 소비'] += rawdata['그리드 소비(kWh)'].isna().sum()
            na_len_dict['수출 된 에너지'] += rawdata['수출 된 에너지(kWh)'].isna().sum()
            na_len_dict['에너지 수율'] += rawdata['에너지 수율(kWh)'].isna().sum()

    print(f'{user_name} 태양광 사용 가구 NA 체크 종료')

    return na_len_dict

# 4-2. 태양광 미사용 가구 '고OO'에 대한 NA 체크
def get_data_na_not():
    root = get_project_root()
    user_name = get_household_not() # 가구명 : '고OO'
    folder_root = os.path.join(root, 'data_revised_all', user_name) # 폴더 경로
    dir_list = os.listdir(folder_root) # 경로 안에 있는 파일 리스트

    # NA 개수 확인을 위한 딕셔너리 생성
    na_len_dict = {}
    na_len_dict['그리드 소비'] = 0
    na_len_dict['수출 된 에너지'] = 0
    # na_len_dict['에너지 수율'] = 0

    print(f'{user_name} 태양광 미사용 가구 NA 체크 시작')

    for j in range(len(dir_list)):
        dir_file = os.path.join(folder_root, dir_list[j])
        file_name = dir_list[j]
        user_data = pd.read_excel(dir_file)

        if 'date' not in user_data.columns: # 컬럼에 'date'라는 변수가 없은 경우 -> Rawdata이므로 다시 읽어줘야 함.
            date_val = f'{file_name[0:4]}/{int(file_name[5:7])}/{int(file_name[8:10])}' # 날짜 형식(yyyy/mm/dd)
            for idx, tmp in enumerate(user_data.iloc[:, 0]): # index와 내용(tmp) 추출
                if str(tmp) in ['nan', '마지막 업데이트']: # 리스트 안에 해당되는 내용일 경우 코드 실행하지 않음.
                    continue
                if tmp.split(' ')[0] == f"{date_val}":
                    break

            rawdata = pd.read_excel(dir_file, header=idx) # idx로 header 재설정

            rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True) # 컬럼 이름 변경
            rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정

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

            na_len_dict['그리드 소비'] += rawdata['그리드 소비(kWh)'].isna().sum()
            na_len_dict['수출 된 에너지'] += rawdata['수출 된 에너지(kWh)'].isna().sum()
            # na_len_dict['에너지 수율'] += rawdata['에너지 수율(kWh)'].isna().sum()
        else:
            rawdata = user_data # if문 안에 있는 데이터 변수명과 동일하게 설정

            na_len_dict['그리드 소비'] += rawdata['그리드 소비(kWh)'].isna().sum()
            na_len_dict['수출 된 에너지'] += rawdata['수출 된 에너지(kWh)'].isna().sum()
            # na_len_dict['에너지 수율'] += rawdata['에너지 수율(kWh)'].isna().sum()

    print(f'{user_name} 태양광 미사용 가구 NA 체크 종료')

    return na_len_dict
##########


# 실행부(Pycharm : Ctrl + Shift + F10)
if __name__ == '__main__':
    tmp = get_data_na_use()
    print(tmp)

    tmp2 = get_data_na_not()
    print(tmp2)

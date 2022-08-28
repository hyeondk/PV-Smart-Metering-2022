### Code : Interpolated Data에 대한 1시간 단위 Dataset 생성
### Writer : Donghyeon Kim
### Date : 2022.07.24

# 0. 라이브러리 실행
from pathlib import Path
import os
import pandas as pd
import numpy as np
import openpyxl
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

# 4-1. 사용자 이름 폴더 안에 파일 하나씩 적용
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
        data_time['date'] = [] # 날짜
        data_time['year'] = []  # 연도
        data_time['month'] = []  # 월
        data_time['day'] = []  # 일
        data_time['hour'] = []  # 시간
        data_time['그리드 소비(kWh)'] = []  # 전력 소비량
        data_time['수출 된 에너지(kWh)'] = []  # 전력 수출량
        data_time['에너지 수율(kWh)'] = []  # 전력 생산량
        update = 0

        if user_name[i] == '윤OO': # 윤OO data일 경우 본 코드 실행하지 않음.
            continue

        if user_name[i] not in solar_use: # 태양광 미사용 가구일 경우 본 코드 실행하지 않음.
            continue

        print(f'{user_name[i]} 태양광 사용 가구 dataset 생성 시작')
        folder_root = os.path.join(root, 'data_revised_all', user_name[i]) # 폴더 경로
        dir_list = os.listdir(folder_root) # 경로 안에 있는 파일 리스트

        for j in range(len(dir_list)):
            dir_file = os.path.join(folder_root, dir_list[j])
            file_name = dir_list[j]
            user_data = pd.read_excel(dir_file)

            date_val = f'{file_name[0:4]}/{int(file_name[5:7])}/{int(file_name[8:10])}' # 날짜 형식(yyyy/mm/dd)

            if 'date' not in user_data.columns: # 컬럼에 'date'라는 변수가 없은 경우 -> Rawdata이므로 다시 읽어줘야 함.
                for idx, tmp in enumerate(user_data.iloc[:, 0]): # index와 내용(tmp) 추출
                    if str(tmp) in ['nan', '마지막 업데이트']: # 리스트 안에 해당되는 내용일 경우 코드 실행 X
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
                    rawdata.rename(columns={"Exported Energy(kWh)": '수출 된 에너지(kWh)'}, inplace=True)
                    rawdata.rename(columns={"Yield Energy(kWh)": '에너지 수율(kWh)'}, inplace=True)

                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str) # 타입 문자열로 변경
                rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str)
                rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].astype(str)

                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
                rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32')
                rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].str.replace(',', '').astype('float32')
            else:
                rawdata = user_data # if문 안에 있는 데이터 변수명과 동일하게 설정

                rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정
                rawdata['year'] = rawdata['date'].dt.year
                rawdata['month'] = rawdata['date'].dt.month
                rawdata['day'] = rawdata['date'].dt.day
                rawdata['hour'] = rawdata['date'].dt.hour
                rawdata['minute'] = rawdata['date'].dt.minute

            # Dictionary에 값 대입을 위한 필요한 변수 값 도출
            for row in rawdata.itertuples():
                if update and row.minute == 59:
                    after_c = row._5 # 그리드 소비
                    after_e = row._6 # 수출 된 에너지
                    after_y = row._10 # 에너지 수율

                    consum_ = after_c - before_c # 시간당 그리드 소비 -> 수전 전력량
                    yield_ = after_y - before_y # 시간당 에너지 수율 -> 자가 발전량(전력 생산량)
                    export_ = after_e - before_e # 시간당 수출 된 에너지 -> 에너지 소비량

                    time_total = consum_ + yield_ - export_ # 전력 소비량(consumption)
                    consum_self = yield_ - export_ # 자가 발전량(전력 생산량) 중 자가발전소비량

                    # 값 대입
                    data_time['date'].append(row.date)
                    data_time['year'].append(row.year)
                    data_time['month'].append(row.month)
                    data_time['day'].append(row.day)
                    data_time['hour'].append(row.hour)
                    data_time['그리드 소비(kWh)'].append(consum_)
                    data_time['수출 된 에너지(kWh)'].append(export_)
                    data_time['에너지 수율(kWh)'].append(yield_)

                    # 초기값 변경
                    before_c = after_c
                    before_e = after_e
                    before_y = after_y
                    update -= 1

                # 초기값
                if not update and row.minute == 0:
                    before_c = row._5
                    before_e = row._6
                    before_y = row._10
                    update += 1
            data_frame_time = pd.DataFrame(data_time)

        df_root = os.path.join(root, 'data_revised_hour')
        if not os.path.isdir(df_root):
            os.makedirs(df_root)

        xlsx_name = df_root + '/' + f'{user_name[i]}_dataset_revised_hour.xlsx'
        data_frame_time.to_excel(xlsx_name, sheet_name='hour', index=False)
        print(f'{user_name[i]} 태양광 사용 가구 dataset 생성 완료')

    print("태양광 사용 가구 dataset 모두 생성 완료")
    return

# 6-2. 사용자 이름 폴더 안에 파일 하나씩 적용
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
        data_time['date'] = []  # 날짜
        data_time['year'] = []  # 연도
        data_time['month'] = []  # 월
        data_time['day'] = []  # 일
        data_time['hour'] = []  # 시간
        data_time['그리드 소비(kWh)'] = []  # 전력 소비량
        data_time['수출 된 에너지(kWh)'] = []  # 전력 수출량
        # data_time['에너지 수율(kWh)'] = []  # 전력 생산량
        update = 0

        if user_name[i] == '윤OO':  # 윤OO data일 경우 본 코드 실행하지 않음.
            continue

        if user_name[i] in solar_use:  # 태양광 사용 가구일 경우 본 코드 실행하지 않음.
            continue

        print(f'{user_name[i]} 태양광 미사용 가구 dataset 생성 시작')
        folder_root = os.path.join(root, 'data_revised_all', user_name[i])  # 폴더 경로
        dir_list = os.listdir(folder_root)  # 경로 안에 있는 파일 리스트

        for j in range(len(dir_list)):
            dir_file = os.path.join(folder_root, dir_list[j])
            file_name = dir_list[j]
            user_data = pd.read_excel(dir_file)

            date_val = f'{file_name[0:4]}/{int(file_name[5:7])}/{int(file_name[8:10])}' # 날짜 형식(yyyy/mm/dd)

            if 'date' not in user_data.columns: # 컬럼에 'date'라는 변수가 없은 경우 -> Rawdata이므로 다시 읽어줘야 함.
                for idx, tmp in enumerate(user_data.iloc[:, 0]): # index와 내용(tmp) 추출
                    if str(tmp) in ['nan', '마지막 업데이트']: # 리스트 안에 해당되는 내용일 경우 코드 실행 X
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
                    rawdata.rename(columns={"Exported Energy(kWh)": '수출 된 에너지(kWh)'}, inplace=True)
                    # rawdata.rename(columns={"Yield Energy(kWh)": '에너지 수율(kWh)'}, inplace=True)

                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str) # 타입 문자열로 변경
                rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str)
                # rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].astype(str)

                rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
                rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32')
                # rawdata['에너지 수율(kWh)'] = rawdata['에너지 수율(kWh)'].str.replace(',', '').astype('float32')
            else:
                rawdata = user_data  # if문 안에 있는 데이터 변수명과 동일하게 설정

                rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S')  # 날짜 포맷 설정
                rawdata['year'] = rawdata['date'].dt.year
                rawdata['month'] = rawdata['date'].dt.month
                rawdata['day'] = rawdata['date'].dt.day
                rawdata['hour'] = rawdata['date'].dt.hour
                rawdata['minute'] = rawdata['date'].dt.minute

            # Dictionary에 값 대입을 위한 필요한 변수 값 도출
            for row in rawdata.itertuples():
                if update and row.minute == 59:
                    after_c = row._5  # 그리드 소비
                    after_e = row._6  # 수출 된 에너지
                    # after_y = row._10  # 에너지 수율

                    consum_ = after_c - before_c  # 시간당 그리드 소비 -> 수전 전력량
                    # yield_ = after_y - before_y  # 시간당 에너지 수율 -> 자가 발전량(전력 생산량)
                    export_ = after_e - before_e  # 시간당 수출 된 에너지 -> 에너지 소비량

                    # time_total = consum_ + yield_ - export_  # 전력 소비량(consumption)
                    # consum_self = yield_ - export_  # 자가 발전량(전력 생산량) 중 자가발전소비량

                    # 값 대입
                    data_time['date'].append(row.date)
                    data_time['year'].append(row.year)
                    data_time['month'].append(row.month)
                    data_time['day'].append(row.day)
                    data_time['hour'].append(row.hour)
                    data_time['그리드 소비(kWh)'].append(consum_)
                    data_time['수출 된 에너지(kWh)'].append(export_)
                    # data_time['에너지 수율(kWh)'].append(yield_)

                    # 초기값 변경
                    before_c = after_c
                    before_e = after_e
                    # before_y = after_y
                    update -= 1

                # 초기값
                if not update and row.minute == 0:
                    before_c = row._5
                    before_e = row._6
                    # before_y = row._10
                    update += 1
            data_frame_time = pd.DataFrame(data_time)

        df_root = os.path.join(root, 'data_revised_hour')
        if not os.path.isdir(df_root):
            os.makedirs(df_root)

        xlsx_name = df_root + '/' + f'{user_name[i]}_dataset_revised_hour.xlsx'
        data_frame_time.to_excel(xlsx_name, sheet_name='hour', index=False)
        print(f'{user_name[i]} 태양광 미사용 가구 dataset 생성 완료')

    print("태양광 미사용 가구 dataset 모두 생성 완료")
    return

# 6-3. 사용자 이름 폴더 안에 파일 하나씩 적용
# 태양광 사용 가구 중 special case - '윤OO'
# 2021-08-08까지 변수 '에너지 수율' -> 태양광 발전, 변수 '부하 에너지' -> 그리드 소비를 의미함.
# 2021-08-09부터는 다른 data와 마찬가지로 형태가 동일하므로 코드 그대로 적용 가능.
def get_value_on_use_df_special():
    root = get_project_root() # 루트
    user_name = get_name_root() # 사용자명

    # 2021/1/1 ~ 2021/8/8 날짜 리스트
    date_2021 = makedate()

    for i in range(len(user_name)):

        # 결과 Dictionary 생성
        data_time = {}
        data_time['date'] = []  # 날짜
        data_time['year'] = []  # 연도
        data_time['month'] = []  # 월
        data_time['day'] = []  # 일
        data_time['hour'] = []  # 시간
        data_time['그리드 소비(kWh)'] = []  # 전력 소비량
        data_time['수출 된 에너지(kWh)'] = []  # 전력 수출량
        data_time['에너지 수율(kWh)'] = []  # 전력 생산량
        update = 0

        if user_name[i] != '윤OO': # 윤OO data가 아닐 경우 코드 실행하지 않음.
            continue

        print(f'{user_name[i]} 태양광 사용 가구 dataset 생성 시작')
        folder_root = os.path.join(root, 'data_revised_all', user_name[i]) # 폴더 경로
        dir_list = os.listdir(folder_root) # 경로 안에 있는 파일 리스트

        for j in range(len(dir_list)):
            dir_file = os.path.join(folder_root, dir_list[j])
            file_name = dir_list[j]
            user_data = pd.read_excel(dir_file)

            date_val = f'{file_name[0:4]}/{int(file_name[5:7])}/{int(file_name[8:10])}' # 날짜 형식(yyyy/mm/dd)

            if 'date' not in user_data.columns: # 컬럼에 'date'라는 변수가 없은 경우 -> Rawdata이므로 다시 읽어줘야 함.
                for idx, tmp in enumerate(user_data.iloc[:, 0]): # index와 내용(tmp) 추출
                    if str(tmp) in ['nan', '마지막 업데이트']: # 리스트 안에 해당되는 내용일 경우 코드 실행 X
                        continue
                    if tmp.split(' ')[0] == f"{date_val}":
                        break

                rawdata = pd.read_excel(dir_file, header=idx) # idx로 header 재설정

                if date_val in date_2021:
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

                    # Dictionary에 값 대입을 위한 필요한 변수 값 도출
                    for row in rawdata.itertuples():
                        if update and row.minute == 59:
                            after_c = row._9  # 그리드 소비
                            # after_e = row._6  # 수출 된 에너지
                            after_y = row._5  # 에너지 수율

                            consum_ = after_c - before_c  # 시간당 그리드 소비 -> 수전 전력량
                            yield_ = after_y - before_y  # 시간당 에너지 수율 -> 자가 발전량(전력 생산량)
                            # export_ = after_e - before_e  # 시간당 수출 된 에너지 -> 에너지 소비량

                            # time_total = consum_ + yield_ - export_  # 전력 소비량(consumption)
                            # consum_self = yield_ - export_  # 자가 발전량(전력 생산량) 중 자가발전소비량

                            # 값 대입
                            data_time['date'].append(row.date)
                            data_time['year'].append(row.year)
                            data_time['month'].append(row.month)
                            data_time['day'].append(row.day)
                            data_time['hour'].append(row.hour)
                            data_time['그리드 소비(kWh)'].append(consum_)
                            data_time['수출 된 에너지(kWh)'].append(np.nan)
                            data_time['에너지 수율(kWh)'].append(yield_)

                            # 초기값 변경
                            before_c = after_c
                            # before_e = after_e
                            before_y = after_y
                            update -= 1

                        # 초기값
                        if not update and row.minute == 0:
                            before_c = row._9
                            # before_e = row._6
                            before_y = row._5
                            update += 1
                elif date_val not in date_2021:
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

                    # Dictionary에 값 대입을 위한 필요한 변수 값 도출
                    for row in rawdata.itertuples():
                        if update and row.minute == 59:
                            after_c = row._5  # 그리드 소비
                            after_e = row._6  # 수출 된 에너지
                            after_y = row._10  # 에너지 수율

                            consum_ = after_c - before_c  # 시간당 그리드 소비 -> 수전 전력량
                            yield_ = after_y - before_y  # 시간당 에너지 수율 -> 자가 발전량(전력 생산량)
                            export_ = after_e - before_e  # 시간당 수출 된 에너지 -> 에너지 소비량

                            time_total = consum_ + yield_ - export_  # 전력 소비량(consumption)
                            consum_self = yield_ - export_  # 자가 발전량(전력 생산량) 중 자가발전소비량

                            # 값 대입
                            data_time['date'].append(row.date)
                            data_time['year'].append(row.year)
                            data_time['month'].append(row.month)
                            data_time['day'].append(row.day)
                            data_time['hour'].append(row.hour)
                            data_time['그리드 소비(kWh)'].append(consum_)
                            data_time['수출 된 에너지(kWh)'].append(export_)
                            data_time['에너지 수율(kWh)'].append(yield_)

                            # 초기값 변경
                            before_c = after_c
                            before_e = after_e
                            before_y = after_y
                            update -= 1

                        # 초기값
                        if not update and row.minute == 0:
                            before_c = row._5
                            before_e = row._6
                            before_y = row._10
                            update += 1
            else:
                if date_val in date_2021:
                    rawdata = user_data  # if문 안에 있는 데이터 변수명과 동일하게 설정

                    rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S')  # 날짜 포맷 설정
                    rawdata['year'] = rawdata['date'].dt.year
                    rawdata['month'] = rawdata['date'].dt.month
                    rawdata['day'] = rawdata['date'].dt.day
                    rawdata['hour'] = rawdata['date'].dt.hour
                    rawdata['minute'] = rawdata['date'].dt.minute

                    # Dictionary에 값 대입을 위한 필요한 변수 값 도출
                    for row in rawdata.itertuples():
                        if update and row.minute == 59:
                            after_c = row._9  # 그리드 소비
                            # after_e = row._6  # 수출 된 에너지
                            after_y = row._5  # 에너지 수율

                            consum_ = after_c - before_c  # 시간당 그리드 소비 -> 수전 전력량
                            yield_ = after_y - before_y  # 시간당 에너지 수율 -> 자가 발전량(전력 생산량)
                            # export_ = after_e - before_e  # 시간당 수출 된 에너지 -> 에너지 소비량

                            # time_total = consum_ + yield_ - export_  # 전력 소비량(consumption)
                            # consum_self = yield_ - export_  # 자가 발전량(전력 생산량) 중 자가발전소비량

                            # 값 대입
                            data_time['date'].append(row.date)
                            data_time['year'].append(row.year)
                            data_time['month'].append(row.month)
                            data_time['day'].append(row.day)
                            data_time['hour'].append(row.hour)
                            data_time['그리드 소비(kWh)'].append(consum_)
                            data_time['수출 된 에너지(kWh)'].append(np.nan)
                            data_time['에너지 수율(kWh)'].append(yield_)

                            # 초기값 변경
                            before_c = after_c
                            # before_e = after_e
                            before_y = after_y
                            update -= 1

                        # 초기값
                        if not update and row.minute == 0:
                            before_c = row._9
                            # before_e = row._6
                            before_y = row._5
                            update += 1
                elif date_val not in date_2021:
                    rawdata = user_data  # if문 안에 있는 데이터 변수명과 동일하게 설정

                    rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S')  # 날짜 포맷 설정
                    rawdata['year'] = rawdata['date'].dt.year
                    rawdata['month'] = rawdata['date'].dt.month
                    rawdata['day'] = rawdata['date'].dt.day
                    rawdata['hour'] = rawdata['date'].dt.hour
                    rawdata['minute'] = rawdata['date'].dt.minute

                    # Dictionary에 값 대입을 위한 필요한 변수 값 도출
                    for row in rawdata.itertuples():
                        if update and row.minute == 59:
                            after_c = row._5  # 그리드 소비
                            after_e = row._6  # 수출 된 에너지
                            after_y = row._10  # 에너지 수율

                            consum_ = after_c - before_c  # 시간당 그리드 소비 -> 수전 전력량
                            yield_ = after_y - before_y  # 시간당 에너지 수율 -> 자가 발전량(전력 생산량)
                            export_ = after_e - before_e  # 시간당 수출 된 에너지 -> 에너지 소비량

                            time_total = consum_ + yield_ - export_  # 전력 소비량(consumption)
                            consum_self = yield_ - export_  # 자가 발전량(전력 생산량) 중 자가발전소비량

                            # 값 대입
                            data_time['date'].append(row.date)
                            data_time['year'].append(row.year)
                            data_time['month'].append(row.month)
                            data_time['day'].append(row.day)
                            data_time['hour'].append(row.hour)
                            data_time['그리드 소비(kWh)'].append(consum_)
                            data_time['수출 된 에너지(kWh)'].append(export_)
                            data_time['에너지 수율(kWh)'].append(yield_)

                            # 초기값 변경
                            before_c = after_c
                            before_e = after_e
                            before_y = after_y
                            update -= 1

                        # 초기값
                        if not update and row.minute == 0:
                            before_c = row._5
                            before_e = row._6
                            before_y = row._10
                            update += 1

        data_frame_time = pd.DataFrame(data_time)
        df_root = os.path.join(root, 'data_revised_hour')
        if not os.path.isdir(df_root):
            os.makedirs(df_root)

        xlsx_name = df_root + '/' + f'{user_name[i]}_dataset_revised_hour.xlsx'
        data_frame_time.to_excel(xlsx_name, sheet_name='hour', index=False)
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

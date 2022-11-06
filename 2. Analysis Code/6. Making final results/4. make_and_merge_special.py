### [Special Case : 윤OO]
### Code : Updated data로 1시간 단위 변경 및 Merging with weather data
### Writer : Donghyeon Kim
### Date : 2022.11.06

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

# 2. 1시간 단위 데이터 생성(변수: 수출된 에너지)
# 태양광 사용 가구 중 special case - '윤OO'
# 2021-03-01 ~ 2021-08-08 : 수출된 에너지(kWh) 변수
# 기존에 누락이었던 데이터 부분에 대해 업데이트 자료를 받았음. 해당 자료를 활용하여 NA를 채우고자 함.
def get_value_on_use_df_special():
    root = get_project_root() # 루트
    user_name = '윤OO'
    file_name = '윤OO_0316-0815.csv'

    # 결과 Dictionary 생성
    data_time = {}
    data_time['date'] = []  # 날짜
    data_time['year'] = []  # 연도
    data_time['month'] = []  # 월
    data_time['day'] = []  # 일
    data_time['hour'] = []  # 시간
    data_time['그리드 소비(kWh)'] = []  # 전력 소비량
    data_time['수출 된 에너지(kWh)'] = []  # 전력 수출량
    update = 0

    print(f'{user_name} 태양광 사용 가구 dataset : 1시간 단위 변경 시작')
    folder_root = os.path.join(root, 'data_2')
    dir_name = os.path.join(folder_root, file_name)
    rawdata = pd.read_csv(dir_name, skiprows=2, encoding='cp949')

    rawdata.rename(columns={"Last Updated": 'date'}, inplace=True) # 컬럼 이름 변경

    rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정
    rawdata['year'] = rawdata['date'].dt.year
    rawdata['month'] = rawdata['date'].dt.month
    rawdata['day'] = rawdata['date'].dt.day
    rawdata['hour'] = rawdata['date'].dt.hour
    rawdata['minute'] = rawdata['date'].dt.minute

    if "Grid Consumption(kWh)" in rawdata.columns:
        rawdata.rename(columns={"Grid Consumption(kWh)": '그리드 소비(kWh)'}, inplace=True) # 컬럼 이름 변경
        rawdata.rename(columns={"Exported Energy(kWh)": '수출 된 에너지(kWh)'}, inplace=True)

    if "부하 에너지(kWh)" in rawdata.columns:
        rawdata.rename(columns={"부하 에너지(kWh)": '그리드 소비(kWh)'}, inplace=True)

    rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str) # 타입 문자열로 변경
    rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str)

    rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
    rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32')

    # Dictionary에 값 대입을 위한 필요한 변수 값 도출
    for row in rawdata.itertuples():
        if update and row.minute == 55:
            after_c = row._6  # 그리드 소비
            after_e = row._7  # 수출 된 에너지

            consum_ = after_c - before_c  # 시간당 그리드 소비 -> 수전 전력량
            export_ = after_e - before_e  # 시간당 수출 된 에너지 -> 에너지 소비량

            # 값 대입
            data_time['date'].append(row.date)
            data_time['year'].append(row.year)
            data_time['month'].append(row.month)
            data_time['day'].append(row.day)
            data_time['hour'].append(row.hour)
            data_time['그리드 소비(kWh)'].append(consum_)
            data_time['수출 된 에너지(kWh)'].append(export_)

            # 초기값 변경
            before_c = after_c
            before_e = after_e
            update -= 5

        # 초기값
        if not update and row.minute == 0:
            before_c = row._6
            before_e = row._7
            update += 5

        data_frame_time = pd.DataFrame(data_time)

    xlsx_name = folder_root + '/' + f'{user_name}_dataset_hour_0316-0815.xlsx'
    data_frame_time.to_excel(xlsx_name, sheet_name='hour', index=False)
    print(f'{user_name} 태양광 사용 가구 dataset : 1시간 단위 변경 종료')
    return

# 3. Merging with weather data
def dataset_weather_merge():
    root = get_project_root()
    user_name = '윤OO'

    # 파일 1 : User Data
    folder_root = os.path.join(root, 'data_2')
    file_name = os.path.join(folder_root, '윤OO_dataset_revised_hour.xlsx')
    df1 = pd.read_excel(file_name)

    # 파일 2 : weather data
    weather_folder_root = os.path.join(root, 'data_preload', 'data_weather')
    csv_name = os.path.join(weather_folder_root, 'keei_ldaps.csv')
    df2 = pd.read_csv(csv_name, encoding='cp949')

    df2['dt'] = pd.to_datetime(df2['dt'], format='%Y/%m/%d %H:%M:%S')
    df2['year'] = df2['dt'].dt.year
    df2['month'] = df2['dt'].dt.month
    df2['day'] = df2['dt'].dt.day
    df2['hour'] = df2['dt'].dt.hour

    # temperature 변수 : Kelvin -> Celsius
    df2['temperature'] = df2['temperature'] - 273.15

    df2_filter = df2[['owner', 'temperature', 'uws_10m', 'vws_10m', 'ghi', 'precipitation',
                      'relative_humidity_1p5m', 'specific_humidity_1p5m',
                      'id_hh', 'id_hs', 'year', 'month', 'day', 'hour']]

    df2_final = df2_filter[df2_filter.owner == user_name]

    print(f'{user_name} dataset과 weather data와의 merge 시작')

    # merge 실행
    result = pd.merge(df1, df2_final, how='left', on=['year', 'month', 'day', 'hour'])

    # xlsx 저장
    xlsx_name = folder_root + '/' + f'{user_name}_final_merge_wt.xlsx'
    result.to_excel(xlsx_name, sheet_name='hour', index=False)

    print(f'{user_name} dataset과 weather data와의 merge 완료')

    return

# 실행부(Pycharm : Ctrl + Shift + F10)
if __name__ == '__main__':
    tmp = dataset_weather_merge()
    print(tmp)

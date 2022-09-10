### Code : 태양광 미사용 가구 - Variable Interpolation
### Writer : Donghyeon Kim
### Date : 2022.09.04.

# 0. 라이브러리 설정
import os
import pandas as pd
import numpy as np
import openpyxl

# 1. 사용자 함수 호출
# get_project_root : 파일의 상위-상위 경로 호출
# get_name_root : 모든 사용자 이름 호출
# get_name_root_use : 태양광 사용자 이름 호출
# get_name_root_not : 태양광 미사용자 이름 호출
from pack_utils import get_project_root, get_name_root, get_name_root_not

# 2. 각 사용자 모든 파일 Concatenation
def user_concat_not(user):
    root = get_project_root()
    df_result = pd.DataFrame()

    print(f'{user} 태양광 미사용 가구 Dataset : Concatenation & Interpolation 시작')
    folder_root = os.path.join(root, 'data', user)  # 폴더 경로
    dir_list = os.listdir(folder_root)  # 경로 안에 있는 파일 리스트

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

        rawdata = pd.read_excel(dir_file, header=idx) # idx로 header 재설정

        rawdata.rename(columns={'Unnamed: 0': 'date'}, inplace=True) # 컬럼 이름 변경
        rawdata['date'] = pd.to_datetime(rawdata['date'], format='%Y/%m/%d %H:%M:%S') # 날짜 포맷 설정

        if "Grid Consumption(kWh)" in rawdata.columns:
            rawdata.rename(columns={"Grid Consumption(kWh)": '그리드 소비(kWh)'}, inplace=True) # 컬럼 이름 변경
            rawdata.rename(columns={"Exported Energy(kWh)": '수출 된 에너지(kWh)'}, inplace=True)
            rawdata.rename(columns={"Voltage(V)": '전압(V)'}, inplace=True)
            rawdata.rename(columns={"Current(A)": '전류(A)'}, inplace=True)
            rawdata.rename(columns={"Power(W)": '전력(W)'}, inplace=True)

        rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].astype(str) # 타입 문자열로 변경
        rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].astype(str)

        rawdata['그리드 소비(kWh)'] = rawdata['그리드 소비(kWh)'].str.replace(',', '').astype('float32') # 반점 제거 & 타입 float32로 변경
        rawdata['수출 된 에너지(kWh)'] = rawdata['수출 된 에너지(kWh)'].str.replace(',', '').astype('float32')

        df_result = pd.concat([df_result, rawdata])

    df_result['그리드 소비(kWh)'] = df_result['그리드 소비(kWh)'].fillna(df_result['그리드 소비(kWh)'].interpolate())
    df_result['수출 된 에너지(kWh)'] = df_result['수출 된 에너지(kWh)'].fillna(df_result['수출 된 에너지(kWh)'].interpolate())

    # 결과물 저장
    df_root = os.path.join(root, 'data_concat_not')
    if not os.path.isdir(df_root):
        os.makedirs(df_root)

    xlsx_name = df_root + '\\' + f'{user}_concat.xlsx'
    df_result.to_excel(xlsx_name, sheet_name='concat', index=False)
    print(f'{user} 태양광 미사용 가구 Dataset : Concatenation & Interpolation 종료')
    return

# 3. Interpolated Data - 1시간 단위 Dataset 생성
def make_dataset_hour_not(user):
    root = get_project_root()

    print(f'{user} 태양광 미사용 가구 Dataset : 1시간 단위 변경 시작')

    # 결과 Dictionary 생성
    data_time = {}
    data_time['date'] = [] # 날짜
    data_time['year'] = [] # 연도
    data_time['month'] = [] # 월
    data_time['day'] = [] # 일
    data_time['hour'] = [] # 시간
    data_time['그리드 소비(kWh)'] = [] # 수전 전력량
    data_time['수출 된 에너지(kWh)'] = [] # 잉여 전력량
    data_time['전력 소비량(kWh)'] = [] # 전력 소비량
    update = 0

    folder_root = os.path.join(root, 'data_concat_not')
    file_name = folder_root + '\\' + f'{user}_concat.xlsx'
    rawdata = pd.read_excel(file_name)

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

            consum_ = after_c - before_c # 시간당 그리드 소비 -> 수전 전력량
            export_ = after_e - before_e # 시간당 수출 된 에너지 -> 잉여 전력량
            time_total = consum_ - export_ # 시간당 전력 소비량

            # 값 대입
            data_time['date'].append(row.date)
            data_time['year'].append(row.year)
            data_time['month'].append(row.month)
            data_time['day'].append(row.day)
            data_time['hour'].append(row.hour)
            data_time['그리드 소비(kWh)'].append(consum_)
            data_time['수출 된 에너지(kWh)'].append(export_)
            data_time['전력 소비량(kWh)'].append(time_total)

            # 초기값 변경
            before_c = after_c
            before_e = after_e
            update -= 1

        # 초기값
        if not update and row.minute == 0:
            before_c = row._5
            before_e = row._6
            update += 1
    data_frame_time = pd.DataFrame(data_time)

    df_root = os.path.join(root, 'data_hour_f2')
    if not os.path.isdir(df_root):
        os.makedirs(df_root)

    xlsx_name = df_root + '/' + f'{user}_hour.xlsx'
    data_frame_time.to_excel(xlsx_name, sheet_name='hour', index=False)
    print(f'{user} 태양광 미사용 가구 Dataset : 1시간 단위 변경 종료')
    return

# 4. Interpolated Data - 1일 단위 Dataset 생성
def make_dataset_day_not(user):
    root = get_project_root()

    print(f'{user} 태양광 미사용 가구 Dataset : 1일 단위 변경 시작')

    # 결과 Dictionary 생성
    data_time = {}
    data_time['year'] = [] # 연도
    data_time['month'] = [] # 월
    data_time['day'] = [] # 일
    data_time['그리드 소비(kWh)'] = [] # 수전 전력량
    data_time['수출 된 에너지(kWh)'] = [] # 잉여 전력량
    data_time['전력 소비량(kWh)'] = [] # 전력 소비량

    folder_root = os.path.join(root, 'data_hour_f2')
    file_name = folder_root + '\\' + f'{user}_hour.xlsx'
    rawdata = pd.read_excel(file_name)

    u_year = rawdata.year.unique().tolist()

    for y in u_year:
        date_cond1 = (rawdata.year == y)
        day_filter1 = rawdata[date_cond1]
        u_month = day_filter1.month.unique().tolist()

        for m in u_month:
            date_cond2 = (day_filter1.month == m)
            day_filter2 = day_filter1[date_cond2]
            u_day = day_filter2.day.unique().tolist()

            for d in u_day:
                date_cond3 = (day_filter2.day == d)
                day_filter3 = day_filter2[date_cond3]

                consum_ = np.sum(day_filter3['그리드 소비(kWh)']) # 일일 그리드 소비 -> 수전 전력량
                export_ = np.sum(day_filter3['수출 된 에너지(kWh)']) # 일일 수출 된 에너지 -> 잉여 전력량
                time_total = np.sum(day_filter3['전력 소비량(kWh)']) # 일일 전력 소비량

                # 값 대입
                data_time['year'].append(y)
                data_time['month'].append(m)
                data_time['day'].append(d)
                data_time['그리드 소비(kWh)'].append(consum_)
                data_time['수출 된 에너지(kWh)'].append(export_)
                data_time['전력 소비량(kWh)'].append(time_total)
    data_frame_time = pd.DataFrame(data_time)

    df_root = os.path.join(root, 'data_day_f2')
    if not os.path.isdir(df_root):
        os.makedirs(df_root)

    xlsx_name = df_root + '/' + f'{user}_day.xlsx'
    data_frame_time.to_excel(xlsx_name, sheet_name='day', index=False)
    print(f'{user} 태양광 미사용 가구 Dataset : 1일 단위 변경 종료')
    return


# 실행 함수
def func_try():
    user_name = get_name_root_not()
    for i in range(len(user_name)):
        make_dataset_day_not(user_name[i])
    return

# 실행부
if __name__ == '__main__':
    tmp = func_try()
    print(tmp)

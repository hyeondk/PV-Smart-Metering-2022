### Code : Final Dataset 구성
### Writer : Donghyeon Kim
### Date : 2022.09.04.

## 세팅 ##
# 라이브러리 설정
import os
import pandas as pd
import numpy as np
import openpyxl

# 사용자 함수 호출
# get_project_root : 파일의 상위-상위 경로 호출
# get_name_root : 모든 사용자 이름 호출
# get_name_root_use : 태양광 사용자 이름 호출
# get_name_root_not : 태양광 미사용자 이름 호출
from pack_utils import get_project_root, get_name_root, get_name_root_use, get_name_root_not, kw_dict, kw_value_dict
from pack_utils import get_name_root_use2, get_name_root_not2
from pack_utils import get_name_root_use3

## 태양광 설치 가구 ##
# 1. 1시간 단위 dataset
def make_final_hour_use(user):
    print(f'{user} 태양광 사용 가구 : 1시간 단위 Dataset 생성 시작')

    # User Data
    root = get_project_root()
    folder_root = os.path.join(root, 'data_hour_f')
    file_name = folder_root + '\\' + f'{user}_hour.xlsx'
    df_user = pd.read_excel(file_name)

    # Name Index
    all_name = get_name_root()
    idx = all_name.index(user) + 1

    # kW Dictionary
    kw_type = kw_dict(user)
    kw_value_type = kw_value_dict(user)

    # 결과 Dictionary
    data_time = {}
    data_time['가구번호'] = []
    data_time['연도'] = []
    data_time['월'] = []
    data_time['일'] = []
    data_time['시간'] = []
    data_time['설비용량(kW)'] = []
    data_time['발전량(kWh)'] = []
    data_time['전력소비량(kWh)'] = []
    data_time['수전전력량(kWh)'] = []
    data_time['잉여전력량(kWh)'] = []
    data_time['잉여전력량/발전량'] = []
    data_time['자가소비율'] = []
    data_time['자가공급률'] = []

    u_year = df_user.year.unique().tolist()

    for y in u_year:
        date_cond1 = (df_user.year == y)
        day_filter1 = df_user[date_cond1]
        u_month = day_filter1.month.unique().tolist()

        for m in u_month:
            date_cond2 = (day_filter1.month == m)
            day_filter2 = day_filter1[date_cond2]
            u_day = day_filter2.day.unique().tolist()

            for d in u_day:
                date_cond3 = (day_filter2.day == d)
                day_filter3 = day_filter2[date_cond3]
                u_hour = day_filter3.hour.unique().tolist()

                for h in u_hour:
                    date_cond4 = (day_filter3.hour == h)
                    day_filter4 = day_filter3[date_cond4]

                    # 가구번호
                    data_time['가구번호'].append(idx)

                    # 연도
                    data_time['연도'].append(y)

                    # 월
                    data_time['월'].append(m)

                    # 일
                    data_time['일'].append(d)

                    # 시간
                    data_time['시간'].append(h)

                    # 설비용량(kW)
                    data_time['설비용량(kW)'].append(kw_type)

                    # 발전량(kWh)
                    yield_ = np.sum(day_filter4['에너지 수율(kWh)'])
                    data_time['발전량(kWh)'].append(yield_)

                    # 전력소비량
                    total_self = np.sum(day_filter4['전력 소비량(kWh)'])
                    data_time['전력소비량(kWh)'].append(total_self)

                    # 수전전력량
                    consum_ = np.sum(day_filter4['그리드 소비(kWh)'])
                    data_time['수전전력량(kWh)'].append(consum_)

                    # 잉여전력량(kWh)
                    export_ = np.sum(day_filter4['수출 된 에너지(kWh)'])
                    data_time['잉여전력량(kWh)'].append(export_)

                    # 잉여전력량/발전량
                    export_yield = export_ / yield_
                    data_time['잉여전력량/발전량'].append(export_yield)

                    # 자가소비율
                    scr = (yield_ - export_) / yield_
                    data_time['자가소비율'].append(scr)

                    # 자가공급률
                    ssr = (yield_ - export_) / total_self
                    data_time['자가공급률'].append(ssr)
    data_frame_time = pd.DataFrame(data_time)

    df_root = os.path.join(root, 'data_final_hour_use')
    if not os.path.isdir(df_root):
        os.makedirs(df_root)

    xlsx_name = df_root + '\\' + f'{user}_final_hour.xlsx'
    data_frame_time.to_excel(xlsx_name, sheet_name='hour', index=False)
    print(f'{user} 태양광 사용 가구 : 1시간 단위 Dataset 생성 종료')
    return

# 2. 1일 단위 dataset
def make_final_day_use(user):
    print(f'{user} 태양광 사용 가구 : 1일 단위 Dataset 생성 시작')

    # User Data
    root = get_project_root()
    folder_root = os.path.join(root, 'data_day_f')
    file_name = folder_root + '\\' + f'{user}_day.xlsx'
    df_user = pd.read_excel(file_name)

    # Name Index
    all_name = get_name_root()
    idx = all_name.index(user) + 1

    # kW Dictionary
    kw_type = kw_dict(user)
    kw_value_type = kw_value_dict(user)

    # 결과 Dictionary
    data_time = {}
    data_time['가구번호'] = []
    data_time['연도'] = []
    data_time['월'] = []
    data_time['일'] = []
    data_time['설비용량(kW)'] = []
    data_time['발전량(kWh)'] = []
    data_time['발전시간'] = []
    data_time['이용률'] = []
    data_time['전력소비량(kWh)'] = []
    data_time['수전전력량(kWh)'] = []
    data_time['잉여전력량(kWh)'] = []
    data_time['잉여전력량/발전량'] = []
    data_time['자가소비율'] = []
    data_time['자가공급률'] = []

    u_year = df_user.year.unique().tolist()

    for y in u_year:
        date_cond1 = (df_user.year == y)
        day_filter1 = df_user[date_cond1]
        u_month = day_filter1.month.unique().tolist()

        for m in u_month:
            date_cond2 = (day_filter1.month == m)
            day_filter2 = day_filter1[date_cond2]
            u_day = day_filter2.day.unique().tolist()

            for d in u_day:
                date_cond3 = (day_filter2.day == d)
                day_filter3 = day_filter2[date_cond3]

                # Dictionary appending #
                # 가구번호
                data_time['가구번호'].append(idx)

                # 연도
                data_time['연도'].append(y)

                # 월
                data_time['월'].append(m)

                # 일
                data_time['일'].append(d)

                # 설비용량(kW)
                data_time['설비용량(kW)'].append(kw_type)

                # 발전량(kWh)
                yield_ = np.sum(day_filter3['에너지 수율(kWh)'])
                data_time['발전량(kWh)'].append(yield_)

                # 발전시간
                yield_time = yield_ / kw_value_type
                data_time['발전시간'].append(yield_time)

                # 이용률
                use_rate = (yield_time / 24) * 100
                data_time['이용률'].append(use_rate)

                # 전력소비량
                total_self = np.sum(day_filter3['전력 소비량(kWh)'])
                data_time['전력소비량(kWh)'].append(total_self)

                # 수전전력량
                consum_ = np.sum(day_filter3['그리드 소비(kWh)'])
                data_time['수전전력량(kWh)'].append(consum_)

                # 잉여전력량(kWh)
                export_ = np.sum(day_filter3['수출 된 에너지(kWh)'])
                data_time['잉여전력량(kWh)'].append(export_)

                # 잉여전력량/발전량
                export_yield = export_ / yield_
                data_time['잉여전력량/발전량'].append(export_yield)

                # 자가소비율
                scr = (yield_ - export_) / yield_
                data_time['자가소비율'].append(scr)

                # 자가공급률
                ssr = (yield_ - export_) / total_self
                data_time['자가공급률'].append(ssr)
    data_frame_time = pd.DataFrame(data_time)

    df_root = os.path.join(root, 'data_final_day_use')
    if not os.path.isdir(df_root):
        os.makedirs(df_root)

    xlsx_name = df_root + '\\' + f'{user}_final_day.xlsx'
    data_frame_time.to_excel(xlsx_name, sheet_name='day', index=False)
    print(f'{user} 태양광 사용 가구 : 1일 단위 Dataset 생성 종료')
    return

##################################################################
## 태양광 미설치 가구 ##
# 1. 1시간 단위 dataset
def make_final_hour_not(user):
    print(f'{user} 태양광 미사용 가구 : 1시간 단위 Dataset 생성 시작')

    # User Data
    root = get_project_root()
    folder_root = os.path.join(root, 'data_hour_f2')
    file_name = folder_root + '\\' + f'{user}_hour.xlsx'
    df_user = pd.read_excel(file_name)

    # Name Index
    all_name = get_name_root()
    idx = all_name.index(user) + 1

    # 결과 Dictionary
    data_time = {}
    data_time['가구번호'] = []
    data_time['연도'] = []
    data_time['월'] = []
    data_time['일'] = []
    data_time['시간'] = []
    data_time['전력소비량(kWh)'] = []
    data_time['수전전력량(kWh)'] = []

    u_year = df_user.year.unique().tolist()

    for y in u_year:
        date_cond1 = (df_user.year == y)
        day_filter1 = df_user[date_cond1]
        u_month = day_filter1.month.unique().tolist()

        for m in u_month:
            date_cond2 = (day_filter1.month == m)
            day_filter2 = day_filter1[date_cond2]
            u_day = day_filter2.day.unique().tolist()

            for d in u_day:
                date_cond3 = (day_filter2.day == d)
                day_filter3 = day_filter2[date_cond3]
                u_hour = day_filter3.hour.unique().tolist()

                for h in u_hour:
                    date_cond4 = (day_filter3.hour == h)
                    day_filter4 = day_filter3[date_cond4]

                    # 가구번호
                    data_time['가구번호'].append(idx)

                    # 연도
                    data_time['연도'].append(y)

                    # 월
                    data_time['월'].append(m)

                    # 일
                    data_time['일'].append(d)

                    # 시간
                    data_time['시간'].append(h)

                    # 전력소비량
                    total_self = np.sum(day_filter4['전력 소비량(kWh)'])
                    data_time['전력소비량(kWh)'].append(total_self)

                    # 수전전력량
                    consum_ = np.sum(day_filter4['그리드 소비(kWh)'])
                    data_time['수전전력량(kWh)'].append(consum_)
    data_frame_time = pd.DataFrame(data_time)

    df_root = os.path.join(root, 'data_final_hour_not')
    if not os.path.isdir(df_root):
        os.makedirs(df_root)

    xlsx_name = df_root + '\\' + f'{user}_final_hour.xlsx'
    data_frame_time.to_excel(xlsx_name, sheet_name='hour', index=False)
    print(f'{user} 태양광 미사용 가구 : 1시간 단위 Dataset 생성 종료')
    return

# 2. 1일 단위 dataset
def make_final_day_not(user):
    print(f'{user} 태양광 미사용 가구 : 1일 단위 Dataset 생성 시작')

    # User Data
    root = get_project_root()
    folder_root = os.path.join(root, 'data_day_f2')
    file_name = folder_root + '\\' + f'{user}_day.xlsx'
    df_user = pd.read_excel(file_name)

    # Name Index
    all_name = get_name_root()
    idx = all_name.index(user) + 1

    # 결과 Dictionary
    data_time = {}
    data_time['가구번호'] = []
    data_time['연도'] = []
    data_time['월'] = []
    data_time['일'] = []
    data_time['전력소비량(kWh)'] = []
    data_time['수전전력량(kWh)'] = []

    u_year = df_user.year.unique().tolist()

    for y in u_year:
        date_cond1 = (df_user.year == y)
        day_filter1 = df_user[date_cond1]
        u_month = day_filter1.month.unique().tolist()

        for m in u_month:
            date_cond2 = (day_filter1.month == m)
            day_filter2 = day_filter1[date_cond2]
            u_day = day_filter2.day.unique().tolist()

            for d in u_day:
                date_cond3 = (day_filter2.day == d)
                day_filter3 = day_filter2[date_cond3]

                # Dictionary appending #
                # 가구번호
                data_time['가구번호'].append(idx)

                # 연도
                data_time['연도'].append(y)

                # 월
                data_time['월'].append(m)

                # 일
                data_time['일'].append(d)

                # 전력소비량
                total_self = np.sum(day_filter3['전력 소비량(kWh)'])
                data_time['전력소비량(kWh)'].append(total_self)

                # 수전전력량
                consum_ = np.sum(day_filter3['그리드 소비(kWh)'])
                data_time['수전전력량(kWh)'].append(consum_)
    data_frame_time = pd.DataFrame(data_time)

    df_root = os.path.join(root, 'data_final_day_not')
    if not os.path.isdir(df_root):
        os.makedirs(df_root)

    xlsx_name = df_root + '\\' + f'{user}_final_day.xlsx'
    data_frame_time.to_excel(xlsx_name, sheet_name='day', index=False)
    print(f'{user} 태양광 미사용 가구 : 1일 단위 Dataset 생성 종료')
    return

##################################################################
## User Data + Weather Data ##
# 1. 태양광 사용 가구
def col_concat_use(user):
    print(f'{user} 태양광 사용 가구 : User Data + Weather Data 시작')

    # Root
    root = get_project_root()

    # User Data
    folder_root = os.path.join(root, 'data_final_hour_use')
    file_name = folder_root + '\\' + f'{user}_final_hour.xlsx'
    df_user = pd.read_excel(file_name)

    # Name Index
    all_name = get_name_root()
    idx = all_name.index(user) + 1

    # Weather Data
    weather_folder_root = os.path.join(root, 'data_weather')
    csv_name = weather_folder_root + '\\' + 'keei_ldaps.csv'
    df_weather = pd.read_csv(csv_name, encoding='cp949')

    # Weather Data - year, month, day, hour
    df_weather['dt'] = pd.to_datetime(df_weather['dt'], format='%Y/%m/%d %H:%M:%S')
    df_weather['연도'] = df_weather['dt'].dt.year
    df_weather['월'] = df_weather['dt'].dt.month
    df_weather['일'] = df_weather['dt'].dt.day
    df_weather['시간'] = df_weather['dt'].dt.hour

    # Weather Data - User Filtering
    df_weather_filter = df_weather[df_weather.owner == user]

    # Weather Data - User + Variable Filtering
    df_weather_filter2 = df_weather_filter[['temperature', 'uws_10m', 'vws_10m', 'ghi', 'precipitation',
                                            'relative_humidity_1p5m', 'specific_humidity_1p5m', 'id_hh', 'id_hs',
                                            '연도', '월', '일', '시간']]

    # Data Copy
    df_weather_final = df_weather_filter2.copy()

    # Weather Data - Add 'ym' variable
    df_weather_final.loc[:, 'ym'] = df_weather_filter2['연도'].astype(str) + '/' + df_weather_filter2['월'].astype(str)

    # Weather Data - variable 'temperature'
    # Convert Kelvin into Celsius
    df_weather_final['temperature'] = df_weather_filter2['temperature'] - 273.15

    # Merging Data
    result = pd.merge(df_user, df_weather_final, how='left', on=['연도', '월', '일', '시간'])

    df_root = os.path.join(root, 'data_merge_wt_f')
    if not os.path.isdir(df_root):
        os.makedirs(df_root)

    xlsx_name = df_root + '\\' + f'{user}_final_merge_wt.xlsx'
    result.to_excel(xlsx_name, sheet_name='hour', index=False)
    print(f'{user} 태양광 사용 가구 : User Data + Weather Data 종료')
    return

# 2. 태양광 미사용 가구
def col_concat_not(user):
    print(f'{user} 태양광 미사용 가구 : User Data + Weather Data 시작')

    # Root
    root = get_project_root()

    # User Data
    folder_root = os.path.join(root, 'data_final_hour_not')
    file_name = folder_root + '\\' + f'{user}_final_hour.xlsx'
    df_user = pd.read_excel(file_name)

    # Name Index
    all_name = get_name_root()
    idx = all_name.index(user) + 1

    # Weather Data
    weather_folder_root = os.path.join(root, 'data_weather')
    csv_name = weather_folder_root + '\\' + 'keei_ldaps.csv'
    df_weather = pd.read_csv(csv_name, encoding='cp949')

    # Weather Data - year, month, day, hour
    df_weather['dt'] = pd.to_datetime(df_weather['dt'], format='%Y/%m/%d %H:%M:%S')
    df_weather['연도'] = df_weather['dt'].dt.year
    df_weather['월'] = df_weather['dt'].dt.month
    df_weather['일'] = df_weather['dt'].dt.day
    df_weather['시간'] = df_weather['dt'].dt.hour

    # Weather Data - User Filtering
    df_weather_filter = df_weather[df_weather.owner == user]

    # Weather Data - User + Variable Filtering
    df_weather_filter2 = df_weather_filter[['temperature', 'uws_10m', 'vws_10m', 'ghi', 'precipitation',
                                            'relative_humidity_1p5m', 'specific_humidity_1p5m', 'id_hh', 'id_hs',
                                            '연도', '월', '일', '시간']]

    # Data Copy
    df_weather_final = df_weather_filter2.copy()

    # Weather Data - Add 'ym' variable
    df_weather_final.loc[:, 'ym'] = df_weather_filter2['연도'].astype(str) + '/' + df_weather_filter2['월'].astype(str)

    # Weather Data - variable 'temperature'
    # Convert Kelvin into Celsius
    df_weather_final['temperature'] = df_weather_filter2['temperature'] - 273.15

    # Merging Data
    result = pd.merge(df_user, df_weather_final, how='left', on=['연도', '월', '일', '시간'])

    df_root = os.path.join(root, 'data_merge_wt_f')
    if not os.path.isdir(df_root):
        os.makedirs(df_root)

    xlsx_name = df_root + '\\' + f'{user}_final_merge_wt.xlsx'
    result.to_excel(xlsx_name, sheet_name='hour', index=False)
    print(f'{user} 태양광 미사용 가구 : User Data + Weather Data 종료')
    return

##################################################################
## User Data Concatenation - by hour ##
def concat_data():
    root = get_project_root()
    df_result = pd.DataFrame()

    print('All Dataset : Concatenation 시작')
    folder_root = os.path.join(root, 'data_merge_wt_f')

    solar_use = get_name_root_use()
    solar_not = get_name_root_not()

    for i in range(len(solar_use)):
        print(f'{solar_use[i]} Dataset 호출')
        dir_file = os.path.join(folder_root, f'{solar_use[i]}_final_merge_wt.xlsx')
        df_user = pd.read_excel(dir_file)

        df_user.loc[:, 'type'] = 'use'
        df_user.loc[:, 'owner'] = solar_use[i]

        df_result = pd.concat([df_result, df_user])

    for j in range(len(solar_not)):
        print(f'{solar_not[j]} Dataset 호출')
        dir_file2 = os.path.join(folder_root, f'{solar_not[j]}_final_merge_wt.xlsx')
        df_user2 = pd.read_excel(dir_file2)

        df_user2.loc[:, 'type'] = 'not'
        df_user2.loc[:, 'owner'] = solar_not[j]

        df_result = pd.concat([df_result, df_user2])

    # 결과물 저장
    df_root = os.path.join(root, 'data_merge_wt_f')
    if not os.path.exists(df_root):
        os.makedirs(df_root)

    xlsx_name = df_root + '\\' + 'all_concat_hour.xlsx'
    df_result.to_excel(xlsx_name, sheet_name='concat', index=False)
    print('All Dataset : Concatenation 종료')
    return

## User Data Concatenation - by day ##
def concat_data2():
    root = get_project_root()
    df_result = pd.DataFrame()

    print('All Dataset : Concatenation 시작')
    folder_root = os.path.join(root, 'data_final_day_use')
    folder_root2 = os.path.join(root, 'data_final_day_not')

    solar_use = get_name_root_use()
    solar_not = get_name_root_not()

    for i in range(len(solar_use)):
        print(f'{solar_use[i]} Dataset 호출')
        dir_file = os.path.join(folder_root, f'{solar_use[i]}_final_day.xlsx')
        df_user = pd.read_excel(dir_file)

        df_user.loc[:, 'type'] = 'use'
        df_user.loc[:, 'owner'] = solar_use[i]
        df_user.loc[:, 'ym'] = df_user['연도'].astype(str) + '/' + df_user['월'].astype(str)

        df_result = pd.concat([df_result, df_user])

    for j in range(len(solar_not)):
        print(f'{solar_not[j]} Dataset 호출')
        dir_file2 = os.path.join(folder_root2, f'{solar_not[j]}_final_day.xlsx')
        df_user2 = pd.read_excel(dir_file2)

        df_user2.loc[:, 'type'] = 'not'
        df_user2.loc[:, 'owner'] = solar_not[j]
        df_user2.loc[:, 'ym'] = df_user2['연도'].astype(str) + '/' + df_user2['월'].astype(str)

        df_result = pd.concat([df_result, df_user2])

    # 결과물 저장
    df_root = os.path.join(root, 'data_merge_wt_f')
    if not os.path.exists(df_root):
        os.makedirs(df_root)

    xlsx_name = df_root + '\\' + 'all_concat_day.xlsx'
    df_result.to_excel(xlsx_name, sheet_name='concat', index=False)
    print('All Dataset : Concatenation 종료')
    return

## 단독 주택 한정 ##
def filter_data_hour():
    print('1시간 단위 데이터 : 필터링 시작')
    root = get_project_root()
    df_result = pd.DataFrame()

    folder_root = os.path.join(root, 'data_merge_wt_f')
    solar_use = get_name_root_use2()
    solar_not = get_name_root_not2()

    data_hour = os.path.join(folder_root, 'all_concat_hour.xlsx')
    file_hour = pd.read_excel(data_hour)

    for i in range(len(solar_use)):
        df_user = file_hour[file_hour.owner == solar_use[i]]
        df_result = pd.concat([df_result, df_user])

    for j in range(len(solar_not)):
        df_user2 = file_hour[file_hour.owner == solar_not[j]]
        df_result = pd.concat([df_result, df_user2])

    # 결과물 저장
    xlsx_name = folder_root + '\\' + 'all_concat_hour2.xlsx'
    df_result.to_excel(xlsx_name, sheet_name='concat', index=False)
    print('1시간 단위 데이터 : 필터링 종료')
    return

def filter_data_day():
    print('1일 단위 데이터 : 필터링 시작')
    root = get_project_root()
    df_result = pd.DataFrame()

    folder_root = os.path.join(root, 'data_merge_wt_f')
    solar_use = get_name_root_use2()
    solar_not = get_name_root_not2()

    data_day = os.path.join(folder_root, 'all_concat_day.xlsx')
    file_day = pd.read_excel(data_day)

    for i in range(len(solar_use)):
        df_user = file_day[file_day.owner == solar_use[i]]
        df_result = pd.concat([df_result, df_user])

    for j in range(len(solar_not)):
        df_user2 = file_day[file_day.owner == solar_not[j]]
        df_result = pd.concat([df_result, df_user2])

    # 결과물 저장
    xlsx_name = folder_root + '\\' + 'all_concat_day2.xlsx'
    df_result.to_excel(xlsx_name, sheet_name='concat', index=False)
    print('1일 단위 데이터 : 필터링 종료')
    return

## 태양광 : 3kW 한정 ##
def filter_data_hour2():
    print('1시간 단위 데이터 : 필터링 시작')
    root = get_project_root()
    df_result = pd.DataFrame()

    folder_root = os.path.join(root, 'data_merge_wt_f')
    solar_use = get_name_root_use3()
    solar_not = get_name_root_not2()

    data_hour = os.path.join(folder_root, 'all_concat_hour2.xlsx')
    file_hour = pd.read_excel(data_hour)

    for i in range(len(solar_use)):
        df_user = file_hour[file_hour.owner == solar_use[i]]
        df_result = pd.concat([df_result, df_user])

    for j in range(len(solar_not)):
        df_user2 = file_hour[file_hour.owner == solar_not[j]]
        df_result = pd.concat([df_result, df_user2])

    # 결과물 저장
    xlsx_name = folder_root + '\\' + 'all_concat_hour_3kw.xlsx'
    df_result.to_excel(xlsx_name, sheet_name='concat', index=False)
    print('1시간 단위 데이터 : 필터링 종료')
    return

def filter_data_day2():
    print('1일 단위 데이터 : 필터링 시작')
    root = get_project_root()
    df_result = pd.DataFrame()

    folder_root = os.path.join(root, 'data_merge_wt_f')
    solar_use = get_name_root_use3()
    solar_not = get_name_root_not2()

    data_day = os.path.join(folder_root, 'all_concat_day2.xlsx')
    file_day = pd.read_excel(data_day)

    for i in range(len(solar_use)):
        df_user = file_day[file_day.owner == solar_use[i]]
        df_result = pd.concat([df_result, df_user])

    for j in range(len(solar_not)):
        df_user2 = file_day[file_day.owner == solar_not[j]]
        df_result = pd.concat([df_result, df_user2])

    # 결과물 저장
    xlsx_name = folder_root + '\\' + 'all_concat_day_3kw.xlsx'
    df_result.to_excel(xlsx_name, sheet_name='concat', index=False)
    print('1일 단위 데이터 : 필터링 종료')
    return

# 실행부
if __name__ == '__main__':
    tmp = filter_data_hour2()
    print(tmp)

    tmp2 = filter_data_day2()
    print(tmp2)

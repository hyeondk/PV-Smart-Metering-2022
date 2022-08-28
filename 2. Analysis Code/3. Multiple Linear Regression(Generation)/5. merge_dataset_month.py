### Code : Merge Dataset by month
### Writer : Donghyeon Kim
### Date : 2022.08.15

# 0. 라이브러리 설정
from pathlib import Path
import os
import pandas as pd
import numpy as np
import openpyxl

# 1. 파일의 상위-상위 경로 설정
def get_project_root() -> Path:
    return Path(__file__).parent.parent

# 2. 사용자 이름 획득
def get_name_root():
    root = get_project_root()
    folder_root = os.path.join(root, 'data') # 루트 + 'data' 경로 설정
    user_name = os.listdir(folder_root) # 경로 안에 있는 사용자명
    return user_name

# 3. Weather Data 호출 및 정제
def get_weather_data():
    # 데이터 호출
    root = get_project_root()
    weather_folder_root = os.path.join(root, 'data_weather')
    csv_name = weather_folder_root + '\\' + 'keei_ldaps.csv'
    df_weather = pd.read_csv(csv_name, encoding='cp949')

    # 데이터 정제
    df_weather['dt'] = pd.to_datetime(df_weather['dt'], format='%Y/%m/%d %H:%M:%S')
    df_weather['year'] = df_weather['dt'].dt.year
    df_weather['month'] = df_weather['dt'].dt.month
    df_weather['day'] = df_weather['dt'].dt.day
    df_weather['hour'] = df_weather['dt'].dt.hour
    return df_weather

# 4. Visibility Data 호출 및 정제
def get_visibility_data():
    # 데이터 호출
    root = get_project_root()
    asos_folder_root = os.path.join(root, 'data_ASOS')
    csv_file_name = ['OBS_ASOS_TIM_1.csv', 'OBS_ASOS_TIM_2.csv']

    for i in range(len(csv_file_name)):
        csv_name = asos_folder_root + '\\' + csv_file_name[i]
        if i == 0:
            df_asos = pd.read_csv(csv_name, encoding='cp949')
        else:
            temp = pd.read_csv(csv_name, encoding='cp949')
            df_asos = pd.concat([df_asos, temp])

    # 데이터 정제
    df_asos['일시'] = pd.to_datetime(df_asos['일시'], format='%Y/%m/%d %H:%M:%S')
    df_asos['year'] = df_asos['일시'].dt.year
    df_asos['month'] = df_asos['일시'].dt.month
    df_asos['day'] = df_asos['일시'].dt.day
    df_asos['hour'] = df_asos['일시'].dt.hour
    return df_asos

# 5. Weather Data와 사용자 데이터 Merge
def weather_user_merge(user):
    # 사용자 데이터 호출
    root = get_project_root()
    user_folder_root = os.path.join(root, 'data_revised_month')
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_revised_month.xlsx'
    df_user = pd.read_excel(xlsx_name)

    # Weather Data 호출
    df_weather = get_weather_data()

    # 사용자에 해당하는 Weather Data 필터링
    df_weather_user = df_weather[df_weather.owner == user]

    # Weather Data : 시간 -> 일 단위 변경을 위한 Dictionary
    df_weather_f = {}
    df_weather_f['temperature'] = []
    df_weather_f['uws_10m'] = []
    df_weather_f['vws_10m'] = []
    df_weather_f['ghi'] = []
    df_weather_f['precipitation'] = []
    df_weather_f['relative_humidity_1p5m'] = []
    df_weather_f['specific_humidity_1p5m'] = []
    df_weather_f['owner'] = []
    df_weather_f['id_hh'] = []
    df_weather_f['id_hs'] = []
    df_weather_f['place'] = []
    df_weather_f['kW_type'] = []
    df_weather_f['year'] = []
    df_weather_f['month'] = []

    # 날짜 필터링 이후 Weather Data 단위 변경
    u_year = df_weather_user.year.unique().tolist()

    for y in u_year:
        date_cond1 = (df_weather_user.year == y)
        day_filter1 = df_weather_user[date_cond1]
        u_month = day_filter1.month.unique().tolist()

        for m in u_month:
            date_cond2 = (day_filter1.month == m)
            day_filter2 = day_filter1[date_cond2]
            day_filter_m1 = pd.DataFrame()
            day_filter_m2 = pd.DataFrame()
            day_filter_m3 = pd.DataFrame()
            day_filter_m4 = pd.DataFrame()

            if m in [1, 2, 11, 12]:
                cond_hour = range(8, 18+1)  # 일사량 : 8 ~ 18시

                for h in cond_hour:
                    temp1 = day_filter2.loc[day_filter2.hour == h, :]
                    day_filter_m1 = pd.concat([day_filter_m1, temp1])

                # 1) temperature
                v1 = np.mean(day_filter_m1.temperature)
                df_weather_f['temperature'].append(v1)

                # 2) uws_10m
                v2 = np.mean(day_filter_m1.uws_10m)
                df_weather_f['uws_10m'].append(v2)

                # 3) vws_10m
                v3 = np.mean(day_filter_m1.vws_10m)
                df_weather_f['vws_10m'].append(v3)

                # 4) GHI
                v4 = np.sum(day_filter_m1.ghi)
                df_weather_f['ghi'].append(v4)

                # 5) precipitation
                v5 = np.sum(day_filter_m1.precipitation)
                df_weather_f['precipitation'].append(v5)

                # 6) relative_humidity_1p5m
                v6 = np.mean(day_filter_m1.relative_humidity_1p5m)
                df_weather_f['relative_humidity_1p5m'].append(v6)

                # 7) specific_humidity_1p5m
                v7 = np.mean(day_filter_m1.specific_humidity_1p5m)
                df_weather_f['specific_humidity_1p5m'].append(v7)

                # 8) owner
                v8 = user
                df_weather_f['owner'].append(v8)

                # 9) ID_HH
                v9 = df_weather_user.id_hh.unique().tolist()[0]
                df_weather_f['id_hh'].append(v9)

                # 10) ID_HS
                v10 = df_weather_user.id_hs.unique().tolist()[0]
                df_weather_f['id_hs'].append(v10)

                # 11) place
                v11 = df_weather_user.place.unique().tolist()[0]
                df_weather_f['place'].append(v11)

                # 12) kW_type
                v12 = df_weather_user.kW_type.unique().tolist()[0]
                df_weather_f['kW_type'].append(v12)

                # 13) year, month
                df_weather_f['year'].append(y)
                df_weather_f['month'].append(m)
            elif m == 3:
                cond_hour = range(8, 19+1)  # 일사량 : 8 ~ 19시

                for h in cond_hour:
                    temp2 = day_filter2.loc[day_filter2.hour == h, :]
                    day_filter_m2 = pd.concat([day_filter_m2, temp2])

                # 1) temperature
                v1 = np.mean(day_filter_m2.temperature)
                df_weather_f['temperature'].append(v1)

                # 2) uws_10m
                v2 = np.mean(day_filter_m2.uws_10m)
                df_weather_f['uws_10m'].append(v2)

                # 3) vws_10m
                v3 = np.mean(day_filter_m2.vws_10m)
                df_weather_f['vws_10m'].append(v3)

                # 4) GHI
                v4 = np.sum(day_filter_m2.ghi)
                df_weather_f['ghi'].append(v4)

                # 5) precipitation
                v5 = np.sum(day_filter_m2.precipitation)
                df_weather_f['precipitation'].append(v5)

                # 6) relative_humidity_1p5m
                v6 = np.mean(day_filter_m2.relative_humidity_1p5m)
                df_weather_f['relative_humidity_1p5m'].append(v6)

                # 7) specific_humidity_1p5m
                v7 = np.mean(day_filter_m2.specific_humidity_1p5m)
                df_weather_f['specific_humidity_1p5m'].append(v7)

                # 8) owner
                v8 = user
                df_weather_f['owner'].append(v8)

                # 9) ID_HH
                v9 = df_weather_user.id_hh.unique().tolist()[0]
                df_weather_f['id_hh'].append(v9)

                # 10) ID_HS
                v10 = df_weather_user.id_hs.unique().tolist()[0]
                df_weather_f['id_hs'].append(v10)

                # 11) place
                v11 = df_weather_user.place.unique().tolist()[0]
                df_weather_f['place'].append(v11)

                # 12) kW_type
                v12 = df_weather_user.kW_type.unique().tolist()[0]
                df_weather_f['kW_type'].append(v12)

                # 13) year, month
                df_weather_f['year'].append(y)
                df_weather_f['month'].append(m)
            elif m in [4, 9, 10]:
                cond_hour = range(7, 19+1)  # 일사량 : 7 ~ 19시

                for h in cond_hour:
                    temp3 = day_filter2.loc[day_filter2.hour == h, :]
                    day_filter_m3 = pd.concat([day_filter_m3, temp3])

                # 1) temperature
                v1 = np.mean(day_filter_m3.temperature)
                df_weather_f['temperature'].append(v1)

                # 2) uws_10m
                v2 = np.mean(day_filter_m3.uws_10m)
                df_weather_f['uws_10m'].append(v2)

                # 3) vws_10m
                v3 = np.mean(day_filter_m3.vws_10m)
                df_weather_f['vws_10m'].append(v3)

                # 4) GHI
                v4 = np.sum(day_filter_m3.ghi)
                df_weather_f['ghi'].append(v4)

                # 5) precipitation
                v5 = np.sum(day_filter_m3.precipitation)
                df_weather_f['precipitation'].append(v5)

                # 6) relative_humidity_1p5m
                v6 = np.mean(day_filter_m3.relative_humidity_1p5m)
                df_weather_f['relative_humidity_1p5m'].append(v6)

                # 7) specific_humidity_1p5m
                v7 = np.mean(day_filter_m3.specific_humidity_1p5m)
                df_weather_f['specific_humidity_1p5m'].append(v7)

                # 8) owner
                v8 = user
                df_weather_f['owner'].append(v8)

                # 9) ID_HH
                v9 = df_weather_user.id_hh.unique().tolist()[0]
                df_weather_f['id_hh'].append(v9)

                # 10) ID_HS
                v10 = df_weather_user.id_hs.unique().tolist()[0]
                df_weather_f['id_hs'].append(v10)

                # 11) place
                v11 = df_weather_user.place.unique().tolist()[0]
                df_weather_f['place'].append(v11)

                # 12) kW_type
                v12 = df_weather_user.kW_type.unique().tolist()[0]
                df_weather_f['kW_type'].append(v12)

                # 13) year, month
                df_weather_f['year'].append(y)
                df_weather_f['month'].append(m)
            elif m in [5, 6, 7, 8]:
                cond_hour = range(6, 20 + 1)  # 일사량 : 6 ~ 20시

                for h in cond_hour:
                    temp4 = day_filter2.loc[day_filter2.hour == h, :]
                    day_filter_m4 = pd.concat([day_filter_m4, temp4])

                # 1) temperature
                v1 = np.mean(day_filter_m4.temperature)
                df_weather_f['temperature'].append(v1)

                # 2) uws_10m
                v2 = np.mean(day_filter_m4.uws_10m)
                df_weather_f['uws_10m'].append(v2)

                # 3) vws_10m
                v3 = np.mean(day_filter_m4.vws_10m)
                df_weather_f['vws_10m'].append(v3)

                # 4) GHI
                v4 = np.sum(day_filter_m4.ghi)
                df_weather_f['ghi'].append(v4)

                # 5) precipitation
                v5 = np.sum(day_filter_m4.precipitation)
                df_weather_f['precipitation'].append(v5)

                # 6) relative_humidity_1p5m
                v6 = np.mean(day_filter_m4.relative_humidity_1p5m)
                df_weather_f['relative_humidity_1p5m'].append(v6)

                # 7) specific_humidity_1p5m
                v7 = np.mean(day_filter_m4.specific_humidity_1p5m)
                df_weather_f['specific_humidity_1p5m'].append(v7)

                # 8) owner
                v8 = user
                df_weather_f['owner'].append(v8)

                # 9) ID_HH
                v9 = df_weather_user.id_hh.unique().tolist()[0]
                df_weather_f['id_hh'].append(v9)

                # 10) ID_HS
                v10 = df_weather_user.id_hs.unique().tolist()[0]
                df_weather_f['id_hs'].append(v10)

                # 11) place
                v11 = df_weather_user.place.unique().tolist()[0]
                df_weather_f['place'].append(v11)

                # 12) kW_type
                v12 = df_weather_user.kW_type.unique().tolist()[0]
                df_weather_f['kW_type'].append(v12)

                # 13) year, month
                df_weather_f['year'].append(y)
                df_weather_f['month'].append(m)

    df_weather_f = pd.DataFrame(df_weather_f)

    # Data Merge
    df_weather_use = pd.merge(df_weather_f, df_user, how='left', on=['year', 'month'])

    # Column 이름 변경
    df_weather_use.rename(columns={'그리드 소비(kWh)': 'grid_kWh',
                                   '수출 된 에너지(kWh)': 'export_kWh',
                                   '에너지 수율(kWh)': 'yield_kWh'}, inplace=True)

    # 'year/month' Column 추가
    df_weather_use.loc[:, 'ym'] = df_weather_use.year.astype(str) + '/' + df_weather_use.month.astype(str)

    # 2021.03 ~ 2022.04 기간에 해당하는 데이터만 필터링
    d_year = [2021, 2022]
    d_month_21 = range(3, 12 + 1)
    d_month_22 = range(1, 4 + 1)

    for y in d_year:
        count = 0
        if y == 2021:
            for ma in d_month_21:
                temp = df_weather_use[(df_weather_use.year == y) & (df_weather_use.month == ma)]
                if count == 0:
                    result = temp
                    count += 1
                else:
                    result = pd.concat([result, temp])
        elif y == 2022:
            for mb in d_month_22:
                temp = df_weather_use[(df_weather_use.year == y) & (df_weather_use.month == mb)]
                result = pd.concat([result, temp])

    df_weather_use = result

    # 온도(temperature) 변환
    # 기존 형태 : Kelvin(켈빈 온도) -> 변경하고자 하는 형태 : 섭씨 온도
    # 섭씨 온도와 켈빈 온도 관계식 : 0(C) + 273.15(K) = 273.15(K)
    df_weather_use.temperature = df_weather_use.temperature - 273.15

    # Index ignore
    df_weather_use = df_weather_use.reset_index(drop=True)

    # 강수량(precipitation) > 0이면 그룹 = 1, 아니면 그룹 = 0으로 부여
    prec_group = []
    for i in range(len(df_weather_use)):
        if df_weather_use.precipitation[i] == 0:
            prec_group.append('no rain')
        else:
            prec_group.append('rain')

    df_weather_use.loc[:, 'status'] = prec_group
    return df_weather_use

# 6. (Weather Data + 사용자 데이터)와 Visibility Data Merge
# Regression Analysis를 위한 사전 작업
def weather_user_visibility_merge(user):
    print(f'{user} 데이터 : Weather Data와 Visibility Data Merge 시작')
    # 루트 설정
    root = get_project_root()

    # 결합된 데이터 호출(Weather Data + 사용자 데이터)
    df_weather_use = weather_user_merge(user)
    weather_place = df_weather_use.place.unique().tolist()[0]

    # Visibility Data 호출
    df_asos = get_visibility_data()

    # Visibility Data Column 변경
    df_asos.rename(columns={'시정(10m)': 'visibility', '지점명': 'place'}, inplace=True)

    # Visibility Data Column 필터링
    df_asos_filter = df_asos.drop(['지점', '일시'], axis=1)

    # User Data에 해당하는 장소로 필터링
    df_asos_filter = df_asos_filter[df_asos_filter.place == weather_place]

    # Weather Data : 시간 -> 일 단위 변경을 위한 Dictionary
    df_asos_filter2 = {}
    df_asos_filter2['year'] = []
    df_asos_filter2['month'] = []
    df_asos_filter2['place'] = []
    df_asos_filter2['visibility'] = []

    # 날짜 필터링 이후 Weather Data 단위 변경
    u_year = df_asos_filter.year.unique().tolist()

    for y in u_year:
        date_cond1 = (df_asos_filter.year == y)
        day_filter1 = df_asos_filter[date_cond1]
        u_month = day_filter1.month.unique().tolist()

        for m in u_month:
            date_cond2 = (day_filter1.month == m)
            day_filter2 = day_filter1[date_cond2]
            day_filter_m1 = pd.DataFrame()
            day_filter_m2 = pd.DataFrame()
            day_filter_m3 = pd.DataFrame()
            day_filter_m4 = pd.DataFrame()

            if m in [1, 2, 11, 12]:
                cond_hour = range(8, 18+1)

                for h in cond_hour:
                    temp1 = day_filter2.loc[day_filter2.hour == h, :]
                    day_filter_m1 = pd.concat([day_filter_m1, temp1])

                # 1) year, month
                df_asos_filter2['year'].append(y)
                df_asos_filter2['month'].append(m)

                # 2) place
                v1 = weather_place
                df_asos_filter2['place'].append(v1)

                # 3) visibility
                v2 = np.mean(day_filter_m1.visibility)
                df_asos_filter2['visibility'].append(v2)
            elif m == 3:
                cond_hour = range(8, 19+1)

                for h in cond_hour:
                    temp2 = day_filter2.loc[day_filter2.hour == h, :]
                    day_filter_m2 = pd.concat([day_filter_m2, temp2])

                # 1) year, month
                df_asos_filter2['year'].append(y)
                df_asos_filter2['month'].append(m)

                # 2) place
                v1 = weather_place
                df_asos_filter2['place'].append(v1)

                # 3) visibility
                v2 = np.mean(day_filter_m2.visibility)
                df_asos_filter2['visibility'].append(v2)
            elif m in [4, 9, 10]:
                cond_hour = range(7, 19+1)

                for h in cond_hour:
                    temp3 = day_filter2.loc[day_filter2.hour == h, :]
                    day_filter_m3 = pd.concat([day_filter_m3, temp3])

                # 1) year, month
                df_asos_filter2['year'].append(y)
                df_asos_filter2['month'].append(m)

                # 2) place
                v1 = weather_place
                df_asos_filter2['place'].append(v1)

                # 3) visibility
                v2 = np.mean(day_filter_m3.visibility)
                df_asos_filter2['visibility'].append(v2)
            elif m in [5, 6, 7, 8]:
                cond_hour = range(6, 20+1)

                for h in cond_hour:
                    temp4 = day_filter2.loc[day_filter2.hour == h, :]
                    day_filter_m4 = pd.concat([day_filter_m4, temp4])

                # 1) year, month
                df_asos_filter2['year'].append(y)
                df_asos_filter2['month'].append(m)

                # 2) place
                v1 = weather_place
                df_asos_filter2['place'].append(v1)

                # 3) visibility
                v2 = np.mean(day_filter_m4.visibility)
                df_asos_filter2['visibility'].append(v2)

    df_asos_f = pd.DataFrame(df_asos_filter2)

    # 두 데이터 Merge
    reg_data = pd.merge(df_weather_use, df_asos_f, how='left', on=['year', 'month', 'place'])

    # Merged Data를 DF로 설정
    reg_data = pd.DataFrame(reg_data)

    result_root = os.path.join(root, 'data_merge_month')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    xlsx_name = result_root + '/' + f'{user}_dataset_merge_month.xlsx'
    reg_data.to_excel(xlsx_name, sheet_name='merge', index=False)
    print(f'{user} 데이터 : Weather Data와 Visibility Data Merge 종료')
    return

# 7. 모든 사용자에 대해 Merge 진행하는 함수
def func_try():
    user_name = get_name_root()
    for i in range(len(user_name)):
        weather_user_visibility_merge(user_name[i])
    return

# 실행부
if __name__ == '__main__':
    tmp = func_try()
    print(tmp)

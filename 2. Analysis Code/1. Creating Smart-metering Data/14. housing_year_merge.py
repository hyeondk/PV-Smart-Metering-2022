### Code : 스마트미터링 자료 가구별 1년치 데이터 생성(추가 작업) / weather data + 가구에너지패널조사 data merge
### Writer : Donghyeon Kim
### Date : 2022.07.05

# 0. 라이브러리 실행
from pathlib import Path
import os
import pandas as pd
import numpy as np
import openpyxl

# 1. 파일의 상위-상위 경로 설정
def get_project_root() -> Path:
    return Path(__file__).parent.parent

# 2. data 폴더 안에 사용자 이름 획득
def get_name_root():
    root = get_project_root()
    folder_root = os.path.join(root, 'data') # 루트 + 'data' 경로 설정
    user_name = os.listdir(folder_root) # 경로 안에 있는 사용자명
    return user_name

# 3. weather data 연단위 평균값 계산
def weather_data_year_mean():
    root = get_project_root()
    user_name = get_name_root()
    weather_folder_root = os.path.join(root, 'data_weather')
    csv_name = weather_folder_root + '\\' + 'keei_ldaps.csv'
    wdf = pd.read_csv(csv_name, encoding='cp949')

    wdf['dt'] = pd.to_datetime(wdf['dt'], format='%Y/%m/%d %H:%M:%S')
    wdf['year'] = wdf['dt'].dt.year
    wdf['month'] = wdf['dt'].dt.month
    wdf['day'] = wdf['dt'].dt.day
    wdf['hour'] = wdf['dt'].dt.hour

    wdf_filter = wdf[['temperature', 'uws_10m', 'vws_10m', 'ghi', 'precipitation', 'relative_humidity_1p5m', 'specific_humidity_1p5m',
                      'id_hh', 'id_hs', 'year', 'month']]

    print('가구 ID 별로 weather data 연평균 계산 시작')
    for i in range(len(user_name)):

        # 결과 Dictionary 생성
        weather_time = {}
        weather_time['id_hh'] = [] # 가구 ID
        weather_time['id_hs'] = [] # 주택 ID
        weather_time['temperature'] = [] # 온도
        weather_time['uws_10m'] = [] # x축 풍속(10m 고도)
        weather_time['vws_10m'] = [] # y축 풍속(10m 고도)
        weather_time['ghi'] = [] # 일사량
        weather_time['precipitation'] = [] # 강수량
        weather_time['relative_humidity_1p5m'] = [] # 상대습도(1.5m 고도)
        weather_time['specific_humidity_1p5m'] = [] # 절대습도(1.5m 고도)

        # 날씨 관련 변수 리스트 생성
        temp = 0
        uws = 0
        vws = 0
        ghi = 0
        prec = 0
        rel_humid = 0
        spec_humid = 0

        # 데이터 길이 변수 설정(21, 22년도 한함)
        weather_length = 0

        cond = (wdf_filter.id_hh == wdf_filter.id_hh.unique().tolist()[i])
        wdf_final = wdf_filter[cond]

        u_year = wdf_final.year.unique() # 연도 필터링

        for y in u_year:
            date_cond = (wdf_final.year == y)
            day_filter = wdf_final[date_cond]
            if y == 2020:
                continue

            if y == 2021:
                u_month_21 = [5, 6, 7, 8, 9, 10, 11, 12]
                for m in u_month_21:
                    date_cond21 = (day_filter.month == m)
                    day_filter21 = day_filter[date_cond21]

                    temp += day_filter21.temperature.sum()
                    uws += day_filter21.uws_10m.sum()
                    vws += day_filter21.vws_10m.sum()
                    ghi += day_filter21.ghi.sum()
                    prec += day_filter21.precipitation.sum()
                    rel_humid += day_filter21.relative_humidity_1p5m.sum()
                    spec_humid += day_filter21.specific_humidity_1p5m.sum()

                    weather_length += len(day_filter21)

            if y == 2022:
                u_month_22 = [1, 2, 3, 4]
                for mm in u_month_22:
                    date_cond22 = (day_filter.month == mm)
                    day_filter22 = day_filter[date_cond22]

                    temp += day_filter22.temperature.sum()
                    uws += day_filter22.uws_10m.sum()
                    vws += day_filter22.vws_10m.sum()
                    ghi += day_filter22.ghi.sum()
                    prec += day_filter22.precipitation.sum()
                    rel_humid += day_filter22.relative_humidity_1p5m.sum()
                    spec_humid += day_filter22.specific_humidity_1p5m.sum()

                    weather_length += len(day_filter22)

        temp_value = temp/weather_length
        uws_value = uws/weather_length
        vws_value = vws/weather_length
        ghi_value = ghi/weather_length
        prec_value = prec/weather_length
        rel_humid_value = rel_humid/weather_length
        spec_humid_value = spec_humid/weather_length

        idhh = wdf_final.id_hh.unique().tolist()[0]  # 가구 ID
        idhs = wdf_final.id_hs.unique().tolist()[0]  # 주택 ID

        # 값 대입
        weather_time['id_hh'].append(idhh)
        weather_time['id_hs'].append(idhs)
        weather_time['temperature'].append(temp_value)
        weather_time['uws_10m'].append(uws_value)
        weather_time['vws_10m'].append(vws_value)
        weather_time['ghi'].append(ghi_value)
        weather_time['precipitation'].append(prec_value)
        weather_time['relative_humidity_1p5m'].append(rel_humid_value)
        weather_time['specific_humidity_1p5m'].append(spec_humid_value)

        data_frame_weather = pd.DataFrame(weather_time)

        # 최종 데이터프레임 : final_result(df 스택 방식)
        if i == 0:
            weather_result = data_frame_weather
        else:
            weather_result = pd.concat([weather_result, data_frame_weather])

    print('가구 ID 별로 weather data 연평균 계산 완료')
    print('weather data by year가 return 값으로 출력됩니다.')

    return weather_result

# 4.'{user_name}_dataset_housing_year.xlsx' 파일 하나씩 불러서 weather data와 merge
def dataset_weather_merge():
    root = get_project_root()
    folder_root = os.path.join(root, 'result_by_user') # 파일 1(스마트미터링 자료) 호출을 위한 루트
    user_name = get_name_root()

    # 파일 2 : weather data
    # 모든 사용자에 대한 정보를 다 가지고 있으므로, 1번만 호출하면 됨.
    df2 = weather_data_year_mean()

    for i in range(len(user_name)):
        print(f'{user_name[i]} dataset과 weather data와의 merge 시작')

        # 파일 1 : 스마트미터링 자료_시간 단위
        xlsx_name = folder_root + '\\' + f'{user_name[i]}_dataset_housing_year.xlsx'
        df1 = pd.read_excel(xlsx_name, sheet_name='year')

        # merge 실행
        result = pd.merge(df1, df2, how='left', on=['id_hh', 'id_hs'])

        # 최종 데이터프레임 : final_result(df 스택 방식)
        if i == 0:
            final_result = result
        else:
            final_result = pd.concat([final_result, result])

        print(f'{user_name[i]} dataset과 weather data와의 merge 완료')

    print('merge 최종 완료')
    print('merged dataset이 return 값으로 출력됩니다.')

    return final_result

# 4. 앞서 만든 merged dataset과 가구에너지패널조사 data와 merge
def final_merge():
    root = get_project_root()

    # 파일 1 + 파일 2 : weather data가 포함된 스마트미터링 자료
    mid_result = dataset_weather_merge()

    # 파일 3 : 가구에너지패널조사 data 중 '주택 및 가구특성, 에너지소비량'
    house_folder_root = os.path.join(root, 'data_HEPS')
    csv_name1 = house_folder_root + '\\' + 'HEPS2019_micro_eng.csv'
    df3 = pd.read_csv(csv_name1, encoding='cp949', low_memory=False)

    df3_filter = df3[['id_hh', 'id_hs', 's09_add_21', 's09_10001', 's09_10003', 's09_10004', 's09_10005', 's09_10006', 's09_10007', 's09_10008',
                      's09_10009', 's09_10010', 's09_10012', 's09_10013', 's09_10014', 's09_10015', 's09_10016', 's09_10017', 's09_10018', 's09_10019',
                      's09_10020', 'g_s09_10021', 's09_20001', 's09_20002', 's09_20003', 's09_20004', 's09_20005', 's09_20006', 's09_20007', 's09_20008',
                      's09_20009', 's09_20010', 's09_20011', 's09_20012', 's09_20013', 's09_20014', 's09_20015', 's09_20016', 's09_20017', 's09_20018',
                      's09_20019', 's09_20021', 's09_20022', 's09_20023', 's09_20024', 's09_20025', 's09_20026', 's09_20027', 's09_20028', 's09_20029',
                      's09_20030', 's09_20031', 's09_20032', 's09_20033', 's09_20034', 's09_20035', 's09_20036', 's09_20037', 's09_20038', 's09_20039',
                      's09_20040', 's09_20041', 's09_20042', 's09_20043', 's09_20044', 's09_20045', 's09_20046', 's09_20048', 's09_20050', 's09_20051',
                      's09_20053', 's09_20055', 's09_20056', 's09_20057', 's09_20058', 's09_20059', 's09_20060', 's09_20061', 's09_20062', 's09_20063',
                      's09_20064', 's09_20065', 's09_20066', 's09_20067', 's09_20068', 's09_20069', 's09_20070', 's09_20071', 's09_20072', 's09_20074',
                      's09_20076', 's09_20077', 's09_20078', 's09_20079', 's09_20080', 's09_20081', 's09_20082', 's09_20083', 's09_20084', 's09_20086',
                      's09_20087', 's09_20088', 's09_20089', 's09_20090', 's09_20091', 's09_20092', 's09_20093', 's09_20094', 's09_20095', 's09_20096',
                      's09_20097', 's09_20098', 's09_20099', 's09_20100', 's09_20101', 's09_20102', 's09_20103', 's09_20104', 's09_20105', 's09_20106',
                      's09_20107', 's09_20108', 's09_20109', 's09_20111', 's09_20112', 's09_20113', 's09_20114', 's09_20115', 's09_20116', 's09_20117',
                      's09_20118', 's09_20119', 's09_20120', 's09_20121', 's09_20122', 's09_20123', 's09_20124', 's09_20125', 's09_20126', 's09_20127',
                      's09_20128', 's09_20129', 's09_20130', 's09_20131', 's09_20132', 's09_20133', 's09_20134', 's09_20135', 's09_20136', 's09_20137',
                      's09_20138', 's09_20139', 's09_20140', 's09_20141', 's09_20142', 's09_20143', 's09_20144', 's09_20145', 's09_20146', 's09_20147',
                      's09_20149', 's09_20150', 's09_20151', 's09_20152', 's09_20153', 's09_20154', 's09_20155', 's09_20156', 's09_20157', 's09_20158',
                      's09_20159', 's09_20160', 's09_20161', 's09_20162', 's09_20163', 's09_20164', 's09_20165', 's09_20166', 's09_20167', 's09_20168',
                      's09_20169', 's09_20170', 's09_20171', 's09_20172', 's09_20173', 's09_20174', 's09_20175', 's09_20176', 's09_20177', 's09_20178',
                      's09_20179', 's09_20180', 's09_20181', 's09_20182', 's09_20183', 's09_20185', 's09_20186', 's09_20187', 's09_20188', 's09_20189',
                      's09_20190', 's09_20191', 's09_20192', 's09_20193', 's09_20194', 's09_20195', 's09_20196', 's09_20197', 's09_20198', 's09_20199',
                      's09_20200', 's09_20201', 's09_20202', 's09_20203', 's09_20204', 's09_20205', 's09_20206', 's09_20207', 's09_20208', 's09_20209',
                      's09_20210', 's09_20211', 's09_20212', 's09_20213', 's09_20214', 's09_20215', 's09_20216', 's09_20217', 's09_20218', 's09_20219',
                      's09_20221', 's09_20222', 's09_20223', 's09_20224', 's09_20225', 's09_20226', 's09_20227', 's09_20228', 's09_20229', 's09_20230',
                      's09_20231', 's09_20232', 's09_20233', 's09_20234', 's09_20235', 's09_20236', 's09_20237', 's09_20238', 's09_20239', 's09_20240',
                      's09_20241', 's09_20242', 's09_20243', 's09_20245', 's09_20246', 's09_20247', 's09_20248', 's09_20249', 's09_20250', 's09_20251',
                      's09_20252', 's09_20253', 'r3_s09_301113', 'r2_s09_302313', 'r3_s09_304113', 'm_r4_s09_305002', 's09_prop_1813', 's09_60001', 's09_60020', 's09_60031',
                      's09_60032', 's09_60033', 's09_60034', 's09_60035', 's09_60036', 's09_60039', 's09_60050', 's09_60051', 's09_60052', 's09_60053',
                      's09_60054', 's09_60057', 's09_60087', 's09_80001', 's09_80002', 's09_80003', 's09_80004', 's09_80005', 's09_80044', 's09_80045',
                      'r1_s09_80046', 'r1_s09_80047']]

    print('-----------------------------------------')
    print('앞서 실행한 merged dataset과 합치는 과정입니다.')
    print('-----------------------------------------')
    print('merged dataset과 주택 및 가구특성_가구에너지패널조사 data와의 merge 시작')

    process1 = pd.merge(mid_result, df3_filter, how='left', on=['id_hh', 'id_hs'])

    print('주택 및 가구특성_가구에너지패널조사 data와의 merge 완료')

    # 파일 4 : 가구에너지패널조사 data 중 '가전기기'
    csv_name2 = house_folder_root + '\\' + 'HEPS2019_micro_app.csv'
    df4 = pd.read_csv(csv_name2, encoding='cp949', low_memory=False)

    df4_filter = df4[['id_hs', 'id_hh', 'app1_numb', 'app1_a_numb', 'app1_b_numb', 'app1_c_numb_n', 'app2_numb', 'app2_a_numb', 'app2_b_numb', 'app2_c_numb',
                      'app3_numb', 'app3_a_numb', 'app3_b_numb', 'app3_c_numb', 'app3_c_numb_1', 'app3_c_numb_2', 'app3_d_numb', 'app4_numb_n', 'app5_numb_n',
                      'app5_a_numb', 'app5_b_numb', 'app5_c_numb', 'app6_numb', 'app7_numb', 'app7_a_numb', 'app7_b_numb', 'app7_c_numb', 'app8_numb',
                      'app8_a_numb', 'app8_b_numb', 'app8_c_numb', 'app9_numb', 'app9_a_numb', 'app9_b_numb', 'app9_c_numb', 'app10_numb',
                      'app1101_1003', 'app1102_1003', 'app1103_1003', 'app1104_1003', 'app1105_1003', 'app1106_1003', 'app1107_1003', 'app1108_1003',
                      'app1109_1003', 'app1110_1003', 'app1111_1003', 'app1112_1003', 'app1113_1003', 'app1114_1003', 'app1115_1003', 'app1116_1003',
                      'app1117_1003', 'app1118_1003', 'app1119_1003', 'app1120_1003', 'app1121_1003', 'app1122_1003', 'app1123_1003', 'app1124_1003',
                      'app1125_1003', 'app1126_1003', 'app1127_1003', 'app1128_1003', 'app1129_1003_r', 'app1130_1003', 'app1131_1003', 'app1132_1003',
                      'app1133_1003', 'app1134_1003', 'fluo_total', 'LED_total', 'incan_total']]

    print('merged dataset과 가전기기_가구에너지패널조사 data와의 merge 시작')

    process2 = pd.merge(process1, df4_filter, how='left', on=['id_hh', 'id_hs'])

    print('가전기기_가구에너지패널조사 data와의 merge 완료')

    # 파일 5 : 가구에너지패널조사 data 중 '자가용'
    csv_name3 = house_folder_root + '\\' + 'HEPS2019_micro_vh.csv'
    df5 = pd.read_csv(csv_name3, encoding='cp949', low_memory=False)

    df5_filter = df5[['id_hs', 'id_hh', 'car_numb', 'car_a_numb', 'car_b_numb', 'car_c_numb', 'car_a1013', 'car_b1013']]

    print('merged dataset과 자가용_가구에너지패널조사 data와의 merge 시작')

    hour_final = pd.merge(process2, df5_filter, how='left', on=['id_hh', 'id_hs'])

    print('자가용_가구에너지패널조사 data와의 merge 완료')

    print('모든 data에 대해 merge 완료')
    print('-----------------------------------------')

    # 변수 'temperature' 값이 켈빈온도(K)이므로 섭씨온도(C)로 변경
    # 공식 : "0(C) + 273.15 = 273.15(K)"
    hour_final['temperature'] = hour_final['temperature'] - 273.15

    df_root = os.path.join(root, 'result_reg1')
    if not os.path.isdir(df_root):
        os.makedirs(df_root)

    print('최종 데이터프레임 csv 파일 저장 시작')

    final_csv_name = df_root + '/' + 'year_merge_data2.csv'
    hour_final.to_csv(final_csv_name, mode='w')

    print('최종 데이터프레임 csv 파일 저장 완료')
    return

# 실행부(Pycharm : Ctrl + Shift + F10)
if __name__ == '__main__':
    tmp = final_merge()
    print(tmp)

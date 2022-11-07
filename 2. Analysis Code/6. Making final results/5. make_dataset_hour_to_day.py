### Code : 1시간 단위 데이터 -> 1일 단위 데이터 변환
### Writer : Donghyeon Kim
### Date : 2022.11.07

# 0. 라이브러리 실행
from pathlib import Path
import os
import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime, date
from pack_utils import get_name_use_final, get_name_not_final

# 1. 파일의 상위-상위 경로 설정
def get_project_root() -> Path:
    return Path(__file__).parent.parent

# 2. 태양광 사용 가구 : 1시간 단위 -> 1일 단위 데이터 변경
def hour_to_day_use():
    root = get_project_root() # 루트
    user_name = get_name_use_final()

    folder_root = os.path.join(root, 'data_final') # 폴더 경로

    print('----------------------------------')
    print('분석할 xlsx 파일 로딩 시작')

    dir_file = os.path.join(folder_root, 'final_data_hour.xlsx') # 파일 경로
    rawdata = pd.read_excel(dir_file)

    print('분석할 xlsx 파일 로드 완료')
    print('----------------------------------')

    # 결과 Dictionary 생성
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
    data_time['type'] = []
    data_time['owner'] = []
    data_time['ym'] = []

    for i in range(len(user_name)):
        print(f'{user_name[i]} 태양광 사용 가구 : 1시간 단위 dataset 생성 시작')
        rawdata_filter = rawdata[rawdata.owner == user_name[i]]

        # Dictionary에 값 대입을 위한 필요한 변수 값 도출
        # 1) 연도 변수 설정
        u_year = rawdata_filter['연도'].unique()

        # 2) 연도 - 월 - 일 필터링 후 일일 단위 데이터 수집
        for y in u_year:
            date_cond1 = (rawdata_filter['연도'] == y)
            day_filter1 = rawdata_filter[date_cond1]
            u_month = day_filter1['월'].unique()

            for m in u_month:
                date_cond2 = (day_filter1['월'] == m)
                day_filter2 = day_filter1[date_cond2]
                u_day = day_filter2['일'].unique()

                for d in u_day:
                    date_cond3 = (day_filter2['일'] == d)
                    day_filter3 = day_filter2[date_cond3]

                    consum_ = sum(day_filter3['수전전력량(kWh)']) # 수전 전력량(그리드 소비량)
                    yield_ = sum(day_filter3['발전량(kWh)']) # 발전량(전력 생산량)
                    export_ = sum(day_filter3['잉여전력량(kWh)']) # 잉여 전력량(수출된 에너지)
                    time_total = sum(day_filter3['전력소비량(kWh)']) # 전력 소비량

                    housing_num = rawdata_filter['가구번호'].unique().tolist()[0] # 가구번호
                    capacity = rawdata_filter['설비용량(kW)'].unique().tolist()[0] # 설비용량

                    # 값 대입
                    data_time['가구번호'].append(housing_num)
                    data_time['연도'].append(y)
                    data_time['월'].append(m)
                    data_time['일'].append(d)
                    data_time['설비용량(kW)'].append(capacity)
                    data_time['발전량(kWh)'].append(yield_)
                    data_time['발전시간'].append(np.nan) # Excel 별도 작업
                    data_time['이용률'].append(np.nan) # Excel 별도 작업
                    data_time['전력소비량(kWh)'].append(time_total)
                    data_time['수전전력량(kWh)'].append(consum_)
                    data_time['잉여전력량(kWh)'].append(export_)
                    data_time['잉여전력량/발전량'].append(np.nan) # Excel 별도 작업
                    data_time['자가소비율'].append(np.nan) # Excel 별도 작업
                    data_time['자가공급률'].append(np.nan) # Excel 별도 작업
                    data_time['type'].append('use')
                    data_time['owner'].append(user_name[i])
                    data_time['ym'].append(str(y) + '/' + str(m))

        data_frame_time = pd.DataFrame(data_time)
        print(f'{user_name[i]} 태양광 사용 가구 : 1시간 단위 dataset 생성 종료')

    print("함수 종료")
    return data_frame_time

# 3. 태양광 미사용 가구 : 1시간 단위 -> 1일 단위 데이터 변경
def hour_to_day_not():
    root = get_project_root() # 루트
    user_name = get_name_not_final()

    folder_root = os.path.join(root, 'data_final') # 폴더 경로

    print('----------------------------------')
    print('분석할 xlsx 파일 로딩 시작')

    dir_file = os.path.join(folder_root, 'final_data_hour.xlsx') # 파일 경로
    rawdata = pd.read_excel(dir_file)

    print('분석할 xlsx 파일 로드 완료')
    print('----------------------------------')

    # 결과 Dictionary 생성
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
    data_time['type'] = []
    data_time['owner'] = []
    data_time['ym'] = []

    for i in range(len(user_name)):
        print(f'{user_name[i]} 태양광 미사용 가구 : 1시간 단위 dataset 생성 시작')
        rawdata_filter = rawdata[rawdata.owner == user_name[i]]

        # Dictionary에 값 대입을 위한 필요한 변수 값 도출
        # 1) 연도 변수 설정
        u_year = rawdata_filter['연도'].unique()

        # 2) 연도 - 월 - 일 필터링 후 일일 단위 데이터 수집
        for y in u_year:
            date_cond1 = (rawdata_filter['연도'] == y)
            day_filter1 = rawdata_filter[date_cond1]
            u_month = day_filter1['월'].unique()

            for m in u_month:
                date_cond2 = (day_filter1['월'] == m)
                day_filter2 = day_filter1[date_cond2]
                u_day = day_filter2['일'].unique()

                for d in u_day:
                    date_cond3 = (day_filter2['일'] == d)
                    day_filter3 = day_filter2[date_cond3]

                    consum_ = sum(day_filter3['수전전력량(kWh)']) # 수전 전력량(그리드 소비량)
                    # yield_ = sum(day_filter3['발전량(kWh)']) # 발전량(전력 생산량)
                    # export_ = sum(day_filter3['잉여전력량(kWh)']) # 잉여 전력량(수출된 에너지)
                    time_total = sum(day_filter3['전력소비량(kWh)']) # 전력 소비량

                    housing_num = rawdata_filter['가구번호'].unique().tolist()[0] # 가구번호
                    # capacity = rawdata_filter['설비용량(kW)'].unique().tolist()[0] # 설비용량

                    # 값 대입
                    data_time['가구번호'].append(housing_num)
                    data_time['연도'].append(y)
                    data_time['월'].append(m)
                    data_time['일'].append(d)
                    data_time['설비용량(kW)'].append(np.nan) # 데이터 없음
                    data_time['발전량(kWh)'].append(np.nan) # 데이터 없음
                    data_time['발전시간'].append(np.nan) # 데이터 없음
                    data_time['이용률'].append(np.nan) # 데이터 없음
                    data_time['전력소비량(kWh)'].append(time_total)
                    data_time['수전전력량(kWh)'].append(consum_)
                    data_time['잉여전력량(kWh)'].append(np.nan) # 데이터 없음
                    data_time['잉여전력량/발전량'].append(np.nan) # 데이터 없음
                    data_time['자가소비율'].append(np.nan) # 데이터 없음
                    data_time['자가공급률'].append(np.nan) # 데이터 없음
                    data_time['type'].append('not')
                    data_time['owner'].append(user_name[i])
                    data_time['ym'].append(str(y) + '/' + str(m))

        data_frame_time = pd.DataFrame(data_time)
        print(f'{user_name[i]} 태양광 미사용 가구 : 1시간 단위 dataset 생성 종료')

    print("함수 종료")
    return data_frame_time

# 4. Dataset Concatenation
def data_concat():
    # 루트 설정
    root = get_project_root()

    # 폴더 루트 설정
    folder_root = os.path.join(root, 'data_final')

    print('Dataset Concatenation 시작')

    # Dataset
    day_use = hour_to_day_use()
    day_not = hour_to_day_not()

    # Concatenation
    result = pd.concat([day_use, day_not])

    # Save result
    final_xlsx_name = os.path.join(folder_root, 'final_data_day.xlsx')
    result.to_excel(final_xlsx_name, sheet_name='day', index=False)

    print('Dataset Concatenation 종료')
    return

# 실행부(Pycharm : Ctrl + Shift + F10)
if __name__ == '__main__':
    tmp = data_concat()
    print(tmp)

### Code : Merge Dataset by hour
### Writer : Donghyeon Kim
### Date : 2022.08.15

# 방법론 #
# 이미 (사용자명)_dataset_merge.xlsx를 통해 모든 데이터를 merge하였음.
# 해당 데이터에서 각 월별로 일사량 존재 시간대만 필터링하고자 함.

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

# 3. Merged Data : 일사량 존재 시간대 필터링
def merge_hour(user):
    print(f'{user} 데이터 : 일사량 존재 시간 필터링 시작')

    # 사용자 데이터 호출
    root = get_project_root()
    user_folder_root = os.path.join(root, 'data_merge')
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_merge.xlsx'
    df_user = pd.read_excel(xlsx_name)

    # 필터링 결과
    day_filter_m = pd.DataFrame()

    # 날짜별로 필터링
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

                if m in [1, 2, 11, 12]:
                    cond_hour = range(8, 18+1) # 일사량 : 8 ~ 18시

                    for h in cond_hour:
                        temp = day_filter3.loc[day_filter3.hour == h, :]
                        day_filter_m = pd.concat([day_filter_m, temp])
                elif m == 3:
                    cond_hour = range(8, 19+1) # 일사량 : 8 ~ 19시

                    for h in cond_hour:
                        temp = day_filter3.loc[day_filter3.hour == h, :]
                        day_filter_m = pd.concat([day_filter_m, temp])
                elif m in [4, 9, 10]:
                    cond_hour = range(7, 19+1)  # 일사량 : 7 ~ 19시

                    for h in cond_hour:
                        temp = day_filter3.loc[day_filter3.hour == h, :]
                        day_filter_m = pd.concat([day_filter_m, temp])
                elif m in [5, 6, 7, 8]:
                    cond_hour = range(6, 20+1) # 일사량 : 6 ~ 20시

                    for h in cond_hour:
                        temp = day_filter3.loc[day_filter3.hour == h, :]
                        day_filter_m = pd.concat([day_filter_m, temp])

    final_data = pd.DataFrame(day_filter_m)

    result_root = os.path.join(root, 'data_merge_hour')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    xlsx_name = result_root + '/' + f'{user}_dataset_merge_hour.xlsx'
    final_data.to_excel(xlsx_name, sheet_name='merge', index=False)
    print(f'{user} 데이터 : 일사량 존재 시간 필터링 종료')
    return

# 4. 모든 사용자에 대해 Merge 진행하는 함수
def func_try():
    user_name = get_name_root()
    for i in range(len(user_name)):
        merge_hour(user_name[i])
    return

# 실행부
if __name__ == '__main__':
    tmp = func_try()
    print(tmp)

### Code : Merge Data_Number of Household members and Income
### Writer : Donghyeon Kim
### Date : 2022.08.24

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

# 3. HEPS data 호출
# 가구원 수, 소득(세후) 확인
def get_heps_data():
    root = get_project_root()
    folder_root = os.path.join(root, 'data_HEPS')
    file_name = folder_root + '\\' + 'HEPS2019_micro_eng.csv'
    df = pd.read_csv(file_name, encoding='cp949', low_memory=False)

    df_filter = df[['id_hh', 'id_hs', 's09_80001', 'r1_s09_80047']]

    return df_filter

# 4. Merged Monthly Data 호출
def get_month_data(user):
    root = get_project_root()
    user_folder_root = os.path.join(root, 'data_merge_month')
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_merge_month.xlsx'

    df_user = pd.read_excel(xlsx_name)
    df_user_filter = df_user.iloc[0, :] # Series
    df_user_filter = pd.DataFrame(df_user_filter).transpose() # Convert to DataFrame
    df_user_final = df_user_filter[['owner', 'id_hh', 'id_hs']]
    return df_user_final

# 5. Data Merge by household
def set_merge(user):
    print(f'{user} 데이터 : Merge 시작')
    df_user = get_month_data(user)
    df_heps = get_heps_data()

    df_result = pd.merge(df_user, df_heps, how='left', on=['id_hh', 'id_hs'])
    df_result.rename(columns={"s09_80001": 'member_num'}, inplace=True) # 컬럼 이름 변경
    df_result.rename(columns={"r1_s09_80047": 'income'}, inplace=True) # 컬럼 이름 변경

    print(f'{user} 데이터 : Merge 종료')
    return df_result

# 6. 모든 사용자에 대해 실행
def func_try():
    root = get_project_root()
    user_name = get_name_root()
    final_result = pd.DataFrame()

    folder_root = os.path.join(root, 'data_HEPS')
    if not os.path.isdir(folder_root):
        os.makedirs(folder_root)

    for i in range(len(user_name)):
        temp = set_merge(user_name[i])
        final_result = pd.concat([final_result, temp])

    final_xlsx_name = folder_root + '\\' + 'Household Info.xlsx'
    final_result.to_excel(final_xlsx_name, sheet_name='info', index=False)
    return

# 실행부
if __name__ == '__main__':
    tmp = func_try()
    print(tmp)

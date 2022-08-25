### Code : Merged Data 수정 - '전력 소비량' 변수 추가
### Writer : Donghyeon Kim
### Date : 2022.08.26

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

# 3. '전력 소비량' 변수 추가
def revise_data(user):
    print(f'{user} 데이터 : 전력 소비량 변수 추가 시작')
    root = get_project_root()
    folder_root = os.path.join(root, 'data_merge')
    xlsx_name = folder_root + '\\' + f'{user}_dataset_merge.xlsx'
    df_user = pd.read_excel(xlsx_name)

    if "yield_kWh" in df_user.columns:
        df_user.loc[:, 'consum_kWh'] = df_user.grid_kWh + df_user.yield_kWh - df_user.export_kWh
    else:
        df_user.loc[:, 'consum_kWh'] = df_user.grid_kWh

    folder_root = os.path.join(root, 'data_merge2')
    if not os.path.isdir(folder_root):
        os.makedirs(folder_root)

    final_xlsx_name = folder_root + '\\' + f'{user}_dataset_merge2.xlsx'
    df_user.to_excel(final_xlsx_name, sheet_name='merge2', index=False)
    print(f'{user} 데이터 : 전력 소비량 변수 추가 종료')
    return

# 4. 모든 사용자에 대해 실행
def func_try():
    user_name = get_name_root()
    for i in range(len(user_name)):
        revise_data(user_name[i])
    return

# 실행부
if __name__ == '__main__':
    tmp = func_try()
    print(tmp)

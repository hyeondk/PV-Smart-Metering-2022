### Code : Concatenate Data by day(태양광 사용가구 한정)
### Writer : Donghyeon Kim
### Date : 2022.08.27

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

# 3. Day Data(일사량 존재시간 한정) 호출
def get_day_data(user):
    root = get_project_root()
    folder_root = os.path.join(root, 'data_merge_day_ghi')
    xlsx_name = folder_root + '\\' + f'{user}_dataset_merge_day.xlsx'
    df_user = pd.read_excel(xlsx_name)
    return df_user

# 4. Day Data Concatenation
def concat_data():
    # 루트 설정
    root = get_project_root()

    # 전체 가구 이름
    user_name = get_name_root()

    # 태양광 사용 가구 이름
    solar_user = ['고OO', '김OO', '김OO', '민OO', '박OO', '변OO', '서OO', '송OO', '오OO', '오OO',
                  '유OO', '윤OO', '이OO', '이OO', '이OO', '임OO', '임OO', '조OO', '최OO', '최OO']

    # 결과 저장을 위한 빈 데이터프레임
    result = pd.DataFrame()

    # Day Data 호출
    for i in range(len(solar_user)):
        print(f'{solar_user[i]} 데이터 : Concatenation 시작')
        df_user = get_day_data(solar_user[i])

        # 태양광 사용 가구 Index
        idx = user_name.index(solar_user[i]) + 1

        # 데이터프레임에 Index 열 추가
        df_user.loc[:, 'idx'] = idx

        result = pd.concat([result, df_user])
        print(f'{solar_user[i]} 데이터 : Concatenation 종료')

    folder_root = os.path.join(root, 'data_merge_day_ghi')
    final_xlsx_name = folder_root + '\\' + 'solar_use_day.xlsx'
    result.to_excel(final_xlsx_name, sheet_name='concat', index=False)
    return

# 실행부
if __name__ == '__main__':
    tmp = concat_data()
    print(tmp)

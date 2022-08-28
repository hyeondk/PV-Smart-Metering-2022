### Code : Merged Data에 NA 채우기(태양광 발전량, Solar Power Generation)
### Writer : Donghyeon Kim
### Date : 2022.08.26

# 0. 라이브러리 설정
from pathlib import Path
import os
import pandas as pd
import numpy as np
import openpyxl
import scipy.stats
from sklearn import linear_model

# 1. 파일의 상위-상위 경로 설정
def get_project_root() -> Path:
    return Path(__file__).parent.parent

# 2. 사용자 이름 획득
def get_name_root():
    root = get_project_root()
    folder_root = os.path.join(root, 'data') # 루트 + 'data' 경로 설정
    user_name = os.listdir(folder_root) # 경로 안에 있는 사용자명
    return user_name

# 3. Merged Data(일사량 한정) 호출
def get_merged_data_ghi(user):
    root = get_project_root()
    folder_root = os.path.join(root, 'data_merge_hour_ghi')
    xlsx_name = folder_root + '\\' + f'{user}_dataset_merge_hour.xlsx'
    df_user = pd.read_excel(xlsx_name)
    return df_user

# 4. Merged Data(일사량 한정) - 태양광 생산량(발전량) NA 채우기
def fillna_on_data(user):
    print(f'{user} 데이터 : 태양광 생산량(발전량) NA 채우기 시작')
    # 루트 설정
    root = get_project_root()

    # 사용자 필터링
    if user in ['박OO', '오OO', '유OO', '윤OO', '이OO', '이OO', '임OO']:
        pass
    else:
        # 데이터 호출
        df_user = get_merged_data_ghi(user)

        # 날짜 필터링 : 2021/3 ~ 2022/4
        date_list = df_user.ym.unique().tolist()

        # 3kW 표준화
        df_kw_type = df_user.kW_type.unique().tolist()[0]

        if df_kw_type == '300W':
            df_user.yield_kWh = df_user.yield_kWh * 10
        elif df_kw_type == '6kW':
            df_user.yield_kWh = df_user.yield_kWh / 2
        elif df_kw_type == '18kW':
            df_user.yield_kWh = df_user.yield_kWh / 6

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            # Remove NAs
            x = df_user_filter.dropna(axis=0)[['ghi', 'temperature', 'visibility']]
            y = df_user_filter.dropna(axis=0)[['yield_kWh']]

            # Modeling #
            # Initialization
            lin_reg = linear_model.LinearRegression(fit_intercept=True)

            # Fitting
            lin_reg_model = lin_reg.fit(x, y)

            # Predicted
            y_predict = lin_reg_model.predict(df_user.loc[:, ['ghi', 'temperature', 'visibility']])

            # Fill NA
            if df_kw_type == '300W':
                y_predict = y_predict / 10
            elif df_kw_type == '6kW':
                y_predict = y_predict * 2
            elif df_kw_type == '18kW':
                y_predict = y_predict * 6

            # Convert to positive values(not negative)
            y_predict = abs(y_predict)

            df_user['yield_kWh'].fillna(pd.Series(y_predict.flatten()), inplace=True)

        folder_root = os.path.join(root, 'data_fillna_yield')
        if not os.path.isdir(folder_root):
            os.makedirs(folder_root)

        final_xlsx_name = folder_root + '\\' + f'{user}_dataset_result.xlsx'
        df_user.to_excel(final_xlsx_name, sheet_name='result', index=False)

    print(f'{user} 데이터 : 태양광 생산량(발전량) NA 채우기 종료')
    return

# 5. 모든 태양광 가구에 대해 실행
def func_try():
    solar_user = ['고OO', '김OO', '김OO', '민OO', '변OO', '서OO', '송OO', '오OO',
                  '이OO', '임OO', '조OO', '최OO', '최OO']
    for i in range(len(solar_user)):
        fillna_on_data(solar_user[i])
    return

# 실행부
if __name__ == '__main__':
    tmp = func_try()
    print(tmp)

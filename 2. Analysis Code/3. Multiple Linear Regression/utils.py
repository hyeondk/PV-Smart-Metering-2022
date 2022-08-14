### Code : Function Utils_Ver2
### Writer : Donghyeon Kim
### Date : 2022.08.12

# 0. 라이브러리 설정
from pathlib import Path
import os
import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

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
    user_folder_root = os.path.join(root, 'data_revised_hour')
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_revised_hour.xlsx'
    df_user = pd.read_excel(xlsx_name)
    df_user_filter = df_user.drop(['date'], axis=1)

    # Weather Data 호출
    df_weather = get_weather_data()

    # 사용자에 해당하는 Weather Data 필터링
    df_weather_use = df_weather[df_weather.owner == user]

    # Data Merge
    df_weather_use = pd.merge(df_weather_use, df_user_filter, how='left', on=['year', 'month', 'day', 'hour'])

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

    # Visibility Data 호출
    df_asos = get_visibility_data()

    # Visibility Data Column 변경
    df_asos.rename(columns={'시정(10m)': 'visibility', '지점명': 'place'}, inplace=True)

    # Visibility Data Column 필터링
    df_asos_filter = df_asos.drop(['지점', '일시'], axis=1)

    # 두 데이터 Merge
    reg_data = pd.merge(df_weather_use, df_asos_filter, how='left', on=['year', 'month', 'day', 'hour', 'place'])

    # Timezone 제거
    reg_data['dt'] = reg_data['dt'].dt.tz_localize(None)

    # Merged Data를 DF로 설정
    reg_data = pd.DataFrame(reg_data)

    result_root = os.path.join(root, 'data_merge')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    xlsx_name = result_root + '/' + f'{user}_dataset_merge.xlsx'
    reg_data.to_excel(xlsx_name, sheet_name='merge', index=False)
    print(f'{user} 데이터 : Weather Data와 Visibility Data Merge 종료')
    return

# 7. Multiple Linear Regession : 시간 단위
def mlr_hour(user):
    print(f'{user} 데이터 : 시간 단위 Multiple Linear Regression 시작')
    # 루트 설정
    root = get_project_root()

    # Merged Data 호출
    user_folder_root = os.path.join(root, 'data_merge')
    user_name = get_name_root()
    idx = user_name.index(user) + 1
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_merge.xlsx'
    df_user = pd.read_excel(xlsx_name)

    # 날짜 필터링 : 2021/4 ~ 2022/3
    date_list = df_user.ym.unique().tolist()
    date_list = date_list[1:-1] # 2021/3, 2022/4 제외

    # 3kW 표준화
    df_kw_type = df_user.kW_type.unique().tolist()[0]

    if df_kw_type == '300W':
        df_user.yield_kWh = df_user.yield_kWh * 10
    elif df_kw_type == '6kW':
        df_user.yield_kWh = df_user.yield_kWh / 2
    elif df_kw_type == '18kW':
        df_user.yield_kWh = df_user.yield_kWh / 6

    if idx not in [16, 31, 33, 35, 43, 45]:
        # SLR Result Dictionary
        result = {}
        result['date'] = []  # Date
        result['obs_no'] = []  # data length by year/month
        result['b0'] = []  # b0(constant)
        result['b1'] = []  # b1(coefficient)
        result['b2'] = []  # b2(coefficient)
        result['b3'] = []  # b3(coefficient)
        result['formula'] = []  # formula
        result['r2'] = []  # R-Square
        result['mse'] = []  # MSE(Mean Squared Error)

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]
            df_user_filter = df_user_filter.drop('export_kWh', axis=1)

            # Date
            result['date'].append(date_list[i])

            # Data Length
            data_len = len(df_user_filter)
            result['obs_no'].append(data_len)

            # Remove NAs
            x = df_user_filter.dropna(axis=0)[['ghi', 'temperature', 'visibility']]
            y = df_user_filter.dropna(axis=0)[['yield_kWh']]

            # Modeling #
            # Initialization
            lin_reg = linear_model.LinearRegression(fit_intercept=True)

            # Fitting
            lin_reg_model = lin_reg.fit(x, y)

            # Predicted
            y_predict = lin_reg_model.predict(x)

            # Constant
            b0 = round(lin_reg_model.intercept_[0], 5)
            result['b0'].append(b0)

            # Coefficient
            b1 = round(lin_reg_model.coef_.tolist()[0][0], 5)
            result['b1'].append(b1)

            b2 = round(lin_reg_model.coef_.tolist()[0][1], 5)
            result['b2'].append(b2)

            b3 = round(lin_reg_model.coef_.tolist()[0][2], 5)
            result['b3'].append(b3)

            # Formula
            formula = f'y = {b0} + {b1}*ghi + {b2}*temp + {b3}*vis'
            result['formula'].append(formula)

            # R-Square
            r2 = round(lin_reg_model.score(x, y), 3)
            result['r2'].append(r2)

            # MSE
            mse = mean_squared_error(y, y_predict)
            result['mse'].append(mse)
    elif idx == 35:
        # SLR Result Dictionary
        result = {}
        result['date'] = [] # Date
        result['obs_no'] = [] # data length by year/month
        result['b0'] = [] # b0(constant)
        result['b1'] = [] # b1(coefficient)
        result['b2'] = [] # b2(coefficient)
        result['b3'] = [] # b3(coefficient)
        result['formula'] = [] # formula
        result['r2'] = [] # R-Square
        result['mse'] = [] # MSE(Mean Squared Error)

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]
            df_user_filter = df_user_filter.drop('export_kWh', axis=1)

            # Date
            result['date'].append(date_list[i])

            # Data Length
            data_len = len(df_user_filter)
            result['obs_no'].append(data_len)

            # Remove NAs
            x = df_user_filter.dropna(axis=0)[['ghi', 'temperature', 'visibility']]
            y = df_user_filter.dropna(axis=0)[['yield_kWh']]

            # Modeling #
            # Initialization
            lin_reg = linear_model.LinearRegression(fit_intercept=True)

            # Fitting
            lin_reg_model = lin_reg.fit(x, y)

            # Predicted
            y_predict = lin_reg_model.predict(x)

            # Constant
            b0 = round(lin_reg_model.intercept_[0], 5)
            result['b0'].append(b0)

            # Coefficient
            b1 = round(lin_reg_model.coef_.tolist()[0][0], 5)
            result['b1'].append(b1)

            b2 = round(lin_reg_model.coef_.tolist()[0][1], 5)
            result['b2'].append(b2)

            b3 = round(lin_reg_model.coef_.tolist()[0][2], 5)
            result['b3'].append(b3)

            # Formula
            formula = f'y = {b0} + {b1}*ghi + {b2}*temp + {b3}*vis'
            result['formula'].append(formula)

            # R-Square
            r2 = round(lin_reg_model.score(x, y), 3)
            result['r2'].append(r2)

            # MSE
            mse = mean_squared_error(y, y_predict)
            result['mse'].append(mse)

        final_result = pd.DataFrame(result)
    elif idx == 45:
        # SLR Result Dictionary
        result = {}
        result['date'] = [] # Date
        result['obs_no'] = [] # data length by year/month
        result['b0'] = [] # b0(constant)
        result['b1'] = [] # b1(coefficient)
        result['b2'] = [] # b2(coefficient)
        result['b3'] = [] # b3(coefficient)
        result['formula'] = [] # formula
        result['r2'] = [] # R-Square
        result['mse'] = [] # MSE(Mean Squared Error)

        for i in range(len(date_list)):
            if i in [0, 1, 2, 3, 4, 5]:
                # Year/Month Filtering
                df_user_filter = df_user[df_user.ym == date_list[i]]

                # Date
                result['date'].append(date_list[i])

                # Data Length
                data_len = len(df_user_filter)
                result['obs_no'].append(data_len)

                # Remove NAs
                # x = df_user_filter.dropna(axis=0)[['ghi', 'temperature', 'visibility']]
                # y = df_user_filter.dropna(axis=0)[['yield_kWh']]

                # Modeling #
                # Initialization
                # lin_reg = linear_model.LinearRegression(fit_intercept=True)

                # Fitting
                # lin_reg_model = lin_reg.fit(x, y)

                # Predicted
                # y_predict = lin_reg_model.predict(x)

                # Constant
                b0 = np.nan
                result['b0'].append(b0)

                # Coefficient
                b1 = np.nan
                result['b1'].append(b1)

                b2 = np.nan
                result['b2'].append(b2)

                b3 = np.nan
                result['b3'].append(b3)

                # Formula
                formula = np.nan
                result['formula'].append(formula)

                # R-Square
                r2 = np.nan
                result['r2'].append(r2)

                # MSE
                mse = np.nan
                result['mse'].append(mse)
            else:
                # Year/Month Filtering
                df_user_filter = df_user[df_user.ym == date_list[i]]

                # Date
                result['date'].append(date_list[i])

                # Data Length
                data_len = len(df_user_filter)
                result['obs_no'].append(data_len)

                # Remove NAs
                x = df_user_filter.dropna(axis=0)[['ghi', 'temperature', 'visibility']]
                y = df_user_filter.dropna(axis=0)[['yield_kWh']]

                # Modeling #
                # Initialization
                lin_reg = linear_model.LinearRegression(fit_intercept=True)

                # Fitting
                lin_reg_model = lin_reg.fit(x, y)

                # Predicted
                y_predict = lin_reg_model.predict(x)

                # Constant
                b0 = round(lin_reg_model.intercept_[0], 5)
                result['b0'].append(b0)

                # Coefficient
                b1 = round(lin_reg_model.coef_.tolist()[0][0], 5)
                result['b1'].append(b1)

                b2 = round(lin_reg_model.coef_.tolist()[0][1], 5)
                result['b2'].append(b2)

                b3 = round(lin_reg_model.coef_.tolist()[0][2], 5)
                result['b3'].append(b3)

                # Formula
                formula = f'y = {b0} + {b1}*ghi + {b2}*temp + {b3}*vis'
                result['formula'].append(formula)

                # R-Square
                r2 = round(lin_reg_model.score(x, y), 3)
                result['r2'].append(r2)

                # MSE
                mse = mean_squared_error(y, y_predict)
                result['mse'].append(mse)

        final_result = pd.DataFrame(result)
    elif idx in [16, 31, 33, 43]:
        # SLR Result Dictionary
        result = {}
        result['date'] = [] # Date
        result['obs_no'] = [] # data length by year/month
        result['b0'] = [] # b0(constant)
        result['b1'] = [] # b1(coefficient)
        result['b2'] = [] # b2(coefficient)
        result['b3'] = [] # b3(coefficient)
        result['formula'] = [] # formula
        result['r2'] = [] # R-Square
        result['mse'] = [] # MSE(Mean Squared Error)

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            # Date
            result['date'].append(date_list[i])

            # Data Length
            data_len = len(df_user_filter)
            result['obs_no'].append(data_len)

            # Remove NAs
            # x = df_user_filter.dropna(axis=0)[['ghi', 'temperature', 'visibility']]
            # y = df_user_filter.dropna(axis=0)[['yield_kWh']]

            # Modeling #
            # Initialization
            # lin_reg = linear_model.LinearRegression(fit_intercept=True)

            # Fitting
            # lin_reg_model = lin_reg.fit(x, y)

            # Predicted
            # y_predict = lin_reg_model.predict(x)

            # Constant
            b0 = np.nan
            result['b0'].append(b0)

            # Coefficient
            b1 = np.nan
            result['b1'].append(b1)

            b2 = np.nan
            result['b2'].append(b2)

            b3 = np.nan
            result['b3'].append(b3)

            # Formula
            formula = np.nan
            result['formula'].append(formula)

            # R-Square
            r2 = np.nan
            result['r2'].append(r2)

            # MSE
            mse = np.nan
            result['mse'].append(mse)

        final_result = pd.DataFrame(result)

    print(f'{user} 데이터 : 시간 단위 Multiple Linear Regression 종료')
    return final_result

# 8. MLR 시간 단위 Residuals Plot
def residual_hour(user):
    print(f'{user} 데이터 : MLR 시간 단위 Residuals Plot 시작')
    # 루트 설정
    root = get_project_root()

    # Merged Data 호출
    user_folder_root = os.path.join(root, 'data_merge')
    user_name = get_name_root()
    idx = user_name.index(user) + 1
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_merge.xlsx'
    df_user = pd.read_excel(xlsx_name)

    # 날짜 필터링 : 2021/4 ~ 2022/3
    date_list = df_user.ym.unique().tolist()
    date_list = date_list[1:-1]  # 2021/3, 2022/4 제외

    # 3kW 표준화
    df_kw_type = df_user.kW_type.unique().tolist()[0]

    if df_kw_type == '300W':
        df_user.yield_kWh = df_user.yield_kWh * 10
    elif df_kw_type == '6kW':
        df_user.yield_kWh = df_user.yield_kWh / 2
    elif df_kw_type == '18kW':
        df_user.yield_kWh = df_user.yield_kWh / 6

    if idx not in [16, 31, 33, 35, 43, 45]:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Residuals Plot by hour for household No.{idx} site', y=0.92, fontsize=22, fontweight='bold')

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
            y_predict = lin_reg_model.predict(x)

            # Residuals
            residuals = y - y_predict

            # Residuals Plot Data
            y_predict = y_predict.flatten()
            residuals = residuals.values.flatten().tolist()

            # Residuals Plot
            if i in [1, 2, 4, 5, 7, 8]:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot(y_predict, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                axp.set(xlabel=None)
                axp.set(ylabel=None)
                axp.axes.xaxis.set_ticklabels([])
                axp.axes.yaxis.set_ticklabels([])
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
            elif i in [0, 3, 6]:  # x축 label만 제거
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot(y_predict, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                axp.set(xlabel=None)
                plt.ylabel("Residuals", fontsize=18)
                axp.axes.xaxis.set_ticklabels([])
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
            elif i in [10, 11]:  # y축 label만 제거
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot(y_predict, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Predicted", fontsize=18)
                axp.set(ylabel=None)
                axp.axes.yaxis.set_ticklabels([])
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
            else:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot(y_predict, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Predicted", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
    elif idx == 35:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Residuals Plot by hour for household No.{idx} site', y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]
            df_user_filter = df_user_filter.drop('export_kWh', axis=1)

            # Remove NAs
            x = df_user_filter.dropna(axis=0)[['ghi', 'temperature', 'visibility']]
            y = df_user_filter.dropna(axis=0)[['yield_kWh']]

            # Modeling #
            # Initialization
            lin_reg = linear_model.LinearRegression(fit_intercept=True)

            # Fitting
            lin_reg_model = lin_reg.fit(x, y)

            # Predicted
            y_predict = lin_reg_model.predict(x)

            # Residuals
            residuals = y - y_predict

            # Residuals Plot Data
            y_predict = y_predict.flatten()
            residuals = residuals.values.flatten().tolist()

            # Residuals Plot
            if i in [1, 2, 4, 5, 7, 8]:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot(y_predict, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                axp.set(xlabel=None)
                axp.set(ylabel=None)
                axp.axes.xaxis.set_ticklabels([])
                axp.axes.yaxis.set_ticklabels([])
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
            elif i in [0, 3, 6]:  # x축 label만 제거
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot(y_predict, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                axp.set(xlabel=None)
                plt.ylabel("Residuals", fontsize=18)
                axp.axes.xaxis.set_ticklabels([])
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
            elif i in [10, 11]:  # y축 label만 제거
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot(y_predict, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Predicted", fontsize=18)
                axp.set(ylabel=None)
                axp.axes.yaxis.set_ticklabels([])
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
            else:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot(y_predict, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Predicted", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
    elif idx == 45:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Residuals Plot by hour for household No.{idx} site', y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            if i in [0, 1, 2, 3, 4, 5]:
                if i in [0, 3]:
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-0.5, 3.0)
                    plt.ylim(-2.5, 2.5)
                    axp = sns.scatterplot()
                    plt.title(f'{date_list[i]}', fontsize=20)
                    axp.set(xlabel=None)
                    plt.ylabel("Residuals", fontsize=18)
                    axp.axes.xaxis.set_ticklabels([])
                    handles, labels = axp.get_legend_handles_labels()
                    list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                    list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                    labels = [x[1] for x in list_labels_handles]
                    handles = [x[0] for x in list_labels_handles]
                    ax.legend(handles, labels)
                else:
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-0.5, 3.0)
                    plt.ylim(-2.5, 2.5)
                    axp = sns.scatterplot()
                    plt.title(f'{date_list[i]}', fontsize=20)
                    axp.set(xlabel=None)
                    axp.set(ylabel=None)
                    axp.axes.xaxis.set_ticklabels([])
                    axp.axes.yaxis.set_ticklabels([])
                    handles, labels = axp.get_legend_handles_labels()
                    list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                    list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                    labels = [x[1] for x in list_labels_handles]
                    handles = [x[0] for x in list_labels_handles]
                    ax.legend(handles, labels)
            else:
                # Remove NAs
                x = df_user_filter.dropna(axis=0)[['ghi', 'temperature', 'visibility']]
                y = df_user_filter.dropna(axis=0)[['yield_kWh']]

                # Modeling #
                # Initialization
                lin_reg = linear_model.LinearRegression(fit_intercept=True)

                # Fitting
                lin_reg_model = lin_reg.fit(x, y)

                # Predicted
                y_predict = lin_reg_model.predict(x)

                # Residuals
                residuals = y - y_predict

                # Residuals Plot Data
                y_predict = y_predict.flatten()
                residuals = residuals.values.flatten().tolist()

                # Residuals Plot
                if i in [1, 2, 4, 5, 7, 8]:
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-0.5, 3.0)
                    plt.ylim(-2.5, 2.5)
                    axp = sns.scatterplot(y_predict, residuals, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    axp.set(xlabel=None)
                    axp.set(ylabel=None)
                    axp.axes.xaxis.set_ticklabels([])
                    axp.axes.yaxis.set_ticklabels([])
                    handles, labels = axp.get_legend_handles_labels()
                    list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                    list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                    labels = [x[1] for x in list_labels_handles]
                    handles = [x[0] for x in list_labels_handles]
                    ax.legend(handles, labels)
                elif i in [0, 3, 6]:  # x축 label만 제거
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-0.5, 3.0)
                    plt.ylim(-2.5, 2.5)
                    axp = sns.scatterplot(y_predict, residuals, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    axp.set(xlabel=None)
                    plt.ylabel("Residuals", fontsize=18)
                    axp.axes.xaxis.set_ticklabels([])
                    handles, labels = axp.get_legend_handles_labels()
                    list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                    list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                    labels = [x[1] for x in list_labels_handles]
                    handles = [x[0] for x in list_labels_handles]
                    ax.legend(handles, labels)
                elif i in [10, 11]:  # y축 label만 제거
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-0.5, 3.0)
                    plt.ylim(-2.5, 2.5)
                    axp = sns.scatterplot(y_predict, residuals, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    plt.xlabel("Predicted", fontsize=18)
                    axp.set(ylabel=None)
                    axp.axes.yaxis.set_ticklabels([])
                    handles, labels = axp.get_legend_handles_labels()
                    list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                    list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                    labels = [x[1] for x in list_labels_handles]
                    handles = [x[0] for x in list_labels_handles]
                    ax.legend(handles, labels)
                else:
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-0.5, 3.0)
                    plt.ylim(-2.5, 2.5)
                    axp = sns.scatterplot(y_predict, residuals, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    plt.xlabel("Predicted", fontsize=18)
                    plt.ylabel("Residuals", fontsize=18)
                    handles, labels = axp.get_legend_handles_labels()
                    list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                    list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                    labels = [x[1] for x in list_labels_handles]
                    handles = [x[0] for x in list_labels_handles]
                    ax.legend(handles, labels)
    elif idx in [16, 31, 33, 43]:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Residuals Plot by hour for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            # Residuals Plot
            if i in [1, 2, 4, 5, 7, 8]:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                axp.set(xlabel=None)
                axp.set(ylabel=None)
                axp.axes.xaxis.set_ticklabels([])
                axp.axes.yaxis.set_ticklabels([])
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
            elif i in [0, 3, 6]:  # x축 label만 제거
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                axp.set(xlabel=None)
                plt.ylabel("Residuals", fontsize=18)
                axp.axes.xaxis.set_ticklabels([])
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
            elif i in [10, 11]:  # y축 label만 제거
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Predicted", fontsize=18)
                axp.set(ylabel=None)
                axp.axes.yaxis.set_ticklabels([])
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
            else:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-2.5, 2.5)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Predicted", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)

    result_root = os.path.join(root, 'result_plot_use')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    result_user_root = os.path.join(result_root, f'{user}')
    if not os.path.isdir(result_user_root):
        os.makedirs(result_user_root)

    fig_name = result_user_root + '/' + f'{user}_mlr_hour_residual.png'
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print(f'{user} 데이터 : MLR 시간 단위 Residuals Plot 종료')
    return
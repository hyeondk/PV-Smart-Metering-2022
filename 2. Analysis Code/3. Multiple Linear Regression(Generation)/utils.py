### Code : Function Utils
### Writer : Donghyeon Kim
### Date : 2022.08.15

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

#####################################################

# 3. Multiple Linear Regession : 시간 단위
def mlr_hour(user):
    print(f'{user} 데이터 : 시간 단위 Multiple Linear Regression 시작')
    # 루트 설정
    root = get_project_root()

    # Merged Data 호출
    user_folder_root = os.path.join(root, 'data_merge_hour')
    user_name = get_name_root()
    idx = user_name.index(user) + 1
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_merge_hour.xlsx'
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

    # 전역 변수
    global final_result

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

# 4. MLR 시간 단위 Residuals Plot
def residual_hour(user):
    print(f'{user} 데이터 : MLR 시간 단위 Residuals Plot 시작')
    # 루트 설정
    root = get_project_root()

    # Merged Data 호출
    user_folder_root = os.path.join(root, 'data_merge_hour')
    user_name = get_name_root()
    idx = user_name.index(user) + 1
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_merge_hour.xlsx'
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
                plt.ylim(-3.5, 3.5)
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
                plt.ylim(-3.5, 3.5)
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
                plt.ylim(-3.5, 3.5)
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
                plt.ylim(-3.5, 3.5)
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
                plt.ylim(-3.5, 3.5)
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
                plt.ylim(-3.5, 3.5)
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
                plt.ylim(-3.5, 3.5)
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
                plt.ylim(-3.5, 3.5)
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
                    plt.ylim(-3.5, 3.5)
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
                    plt.ylim(-3.5, 3.5)
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
                    plt.ylim(-3.5, 3.5)
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
                    plt.ylim(-3.5, 3.5)
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
                    plt.ylim(-3.5, 3.5)
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
                    plt.ylim(-3.5, 3.5)
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
        plt.suptitle(f'Residuals Plot by hour for household No.{idx} site', y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            # Residuals Plot
            if i in [1, 2, 4, 5, 7, 8]:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-0.5, 3.0)
                plt.ylim(-3.5, 3.5)
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
                plt.ylim(-3.5, 3.5)
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
                plt.ylim(-3.5, 3.5)
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
                plt.ylim(-3.5, 3.5)
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

#####################################################

# 5. Multiple Linear Regession : 일 단위
def mlr_day(user):
    print(f'{user} 데이터 : 일 단위 Multiple Linear Regression 시작')
    # 루트 설정
    root = get_project_root()

    # Merged Data 호출
    user_folder_root = os.path.join(root, 'data_merge_day')
    user_name = get_name_root()
    idx = user_name.index(user) + 1
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_merge_day.xlsx'
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

    # 전역 변수
    global final_result

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
    print(f'{user} 데이터 : 일 단위 Multiple Linear Regression 종료')
    return final_result

# 6. MLR 일 단위 Residuals Plot
def residual_day(user):
    print(f'{user} 데이터 : MLR 일 단위 Residuals Plot 시작')
    # 루트 설정
    root = get_project_root()

    # Merged Data 호출
    user_folder_root = os.path.join(root, 'data_merge_day')
    user_name = get_name_root()
    idx = user_name.index(user) + 1
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_merge_day.xlsx'
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
        plt.suptitle(f'Residuals Plot by day for household No.{idx} site', y=0.92, fontsize=22, fontweight='bold')

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
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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
        plt.suptitle(f'Residuals Plot by day for household No.{idx} site', y=0.92, fontsize=22, fontweight='bold')

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
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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
        plt.suptitle(f'Residuals Plot by day for household No.{idx} site', y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            if i in [0, 1, 2, 3, 4, 5]:
                if i in [0, 3]:
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-2.5, 22.5)
                    plt.ylim(-12.5, 12.5)
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
                    plt.xlim(-2.5, 22.5)
                    plt.ylim(-12.5, 12.5)
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
                    plt.xlim(-2.5, 22.5)
                    plt.ylim(-12.5, 12.5)
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
                    plt.xlim(-2.5, 22.5)
                    plt.ylim(-12.5, 12.5)
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
                    plt.xlim(-2.5, 22.5)
                    plt.ylim(-12.5, 12.5)
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
                    plt.xlim(-2.5, 22.5)
                    plt.ylim(-12.5, 12.5)
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
        plt.suptitle(f'Residuals Plot by day for household No.{idx} site', y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            # Residuals Plot
            if i in [1, 2, 4, 5, 7, 8]:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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
                plt.xlim(-2.5, 22.5)
                plt.ylim(-12.5, 12.5)
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

    fig_name = result_user_root + '/' + f'{user}_mlr_day_residual.png'
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print(f'{user} 데이터 : MLR 일 단위 Residuals Plot 종료')
    return

#####################################################

# 7. Multiple Linear Regession : 월 단위
def mlr_month(user):
    print(f'{user} 데이터 : 월 단위 Multiple Linear Regression 시작')
    # 루트 설정
    root = get_project_root()

    # Merged Data 호출
    user_folder_root = os.path.join(root, 'data_merge_month')
    user_name = get_name_root()
    idx = user_name.index(user) + 1
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_merge_month.xlsx'
    df_user = pd.read_excel(xlsx_name)

    # 3kW 표준화
    df_kw_type = df_user.kW_type.unique().tolist()[0]

    if df_kw_type == '300W':
        df_user.yield_kWh = df_user.yield_kWh * 10
    elif df_kw_type == '6kW':
        df_user.yield_kWh = df_user.yield_kWh / 2
    elif df_kw_type == '18kW':
        df_user.yield_kWh = df_user.yield_kWh / 6

    # 전역 변수
    global final_result

    if idx not in [16, 31, 33, 35, 43]:
        # SLR Result Dictionary
        result = {}
        result['obs_no'] = []  # data length by year/month
        result['b0'] = []  # b0(constant)
        result['b1'] = []  # b1(coefficient)
        result['b2'] = []  # b2(coefficient)
        result['b3'] = []  # b3(coefficient)
        result['formula'] = []  # formula
        result['r2'] = []  # R-Square
        result['mse'] = []  # MSE(Mean Squared Error)

        # Data
        df_user_filter = df_user

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
    elif idx == 35:
        # SLR Result Dictionary
        result = {}
        result['obs_no'] = [] # data length by year/month
        result['b0'] = [] # b0(constant)
        result['b1'] = [] # b1(coefficient)
        result['b2'] = [] # b2(coefficient)
        result['b3'] = [] # b3(coefficient)
        result['formula'] = [] # formula
        result['r2'] = [] # R-Square
        result['mse'] = [] # MSE(Mean Squared Error)

        # Data
        df_user_filter = df_user.drop('export_kWh', axis=1)

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
        result['obs_no'] = [] # data length by year/month
        result['b0'] = [] # b0(constant)
        result['b1'] = [] # b1(coefficient)
        result['b2'] = [] # b2(coefficient)
        result['b3'] = [] # b3(coefficient)
        result['formula'] = [] # formula
        result['r2'] = [] # R-Square
        result['mse'] = [] # MSE(Mean Squared Error)

        # Data
        df_user_filter = df_user

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
    print(f'{user} 데이터 : 일 단위 Multiple Linear Regression 종료')
    return final_result

# 8. MLR 월 단위 Residuals Plot
def residual_month(user):
    print(f'{user} 데이터 : MLR 월 단위 Residuals Plot 시작')
    # 루트 설정
    root = get_project_root()

    # Merged Data 호출
    user_folder_root = os.path.join(root, 'data_merge_month')
    user_name = get_name_root()
    idx = user_name.index(user) + 1
    xlsx_name = user_folder_root + '\\' + f'{user}_dataset_merge_month.xlsx'
    df_user = pd.read_excel(xlsx_name)

    # 3kW 표준화
    df_kw_type = df_user.kW_type.unique().tolist()[0]

    if df_kw_type == '300W':
        df_user.yield_kWh = df_user.yield_kWh * 10
    elif df_kw_type == '6kW':
        df_user.yield_kWh = df_user.yield_kWh / 2
    elif df_kw_type == '18kW':
        df_user.yield_kWh = df_user.yield_kWh / 6

    if idx not in [16, 31, 33, 35, 43]:
        # Data
        df_user_filter = df_user

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
        sns.set(rc={'figure.figsize': (14, 9)})
        sns.scatterplot(y_predict, residuals)
        plt.title(f'Residuals Plot by month for household No.{idx} site', fontsize=22, fontweight='bold')
        plt.xlabel("Predicted", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-50.0, 550.0)
        plt.ylim(-75.0, 75.0)
    elif idx == 35:
        # Data
        df_user_filter = df_user.drop('export_kWh', axis=1)

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
        sns.set(rc={'figure.figsize': (14, 9)})
        sns.scatterplot(y_predict, residuals)
        plt.title(f'Residuals Plot by month for household No.{idx} site', fontsize=22, fontweight='bold')
        plt.xlabel("Predicted", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-50.0, 550.0)
        plt.ylim(-75.0, 75.0)
    elif idx in [16, 31, 33, 43]:
        # Data
        df_user_filter = df_user

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

        # Residuals
        # residuals = y - y_predict

        # Residuals Plot Data
        # y_predict = y_predict.flatten()
        # residuals = residuals.values.flatten().tolist()

        # Residuals Plot
        sns.set(rc={'figure.figsize': (14, 9)})
        sns.scatterplot()
        plt.title(f'Residuals Plot by month for household No.{idx} site', fontsize=22, fontweight='bold')
        plt.xlabel("Predicted", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-50.0, 550.0)
        plt.ylim(-75.0, 75.0)

    result_root = os.path.join(root, 'result_plot_use')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    result_user_root = os.path.join(result_root, f'{user}')
    if not os.path.isdir(result_user_root):
        os.makedirs(result_user_root)

    fig_name = result_user_root + '/' + f'{user}_mlr_month_residual.png'
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print(f'{user} 데이터 : MLR 월 단위 Residuals Plot 종료')
    return
### Code : Utils by day
### Writer : Donghyeon Kim
### Date : 2022.08.19

# Multiple Linear Regression #
# 단위 : 1일
# 1) Predicted vs Actual
# 2) X1(GHI) vs Residuals
# 3) X2(Temperature) vs Residuals
# 4) X3(Visibility) vs Residuals

# 0. 라이브러리 설정
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

#####################################################

# 3. Scatter Plot : Predicted vs Actual
def pred_act_plot_day(user):
    print(f'{user} 데이터 : MLR 일 단위 Predicted vs Actual Scatter Plot 시작')
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
        plt.suptitle(f'Scatter Plot between predicted and actual by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

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

            # Actual y
            y_actual = y.values.flatten().tolist()

            # Residuals Plot
            if i in [1, 2, 4, 5, 7, 8]:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-2.5, 22.5)
                plt.ylim(-2.5, 32.0)
                axp = sns.scatterplot(y_predict, y_actual, ax=ax)
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
                plt.ylim(-2.5, 32.0)
                axp = sns.scatterplot(y_predict, y_actual, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                axp.set(xlabel=None)
                plt.ylabel("Actual", fontsize=18)
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
                plt.ylim(-2.5, 32.0)
                axp = sns.scatterplot(y_predict, y_actual, ax=ax)
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
                plt.ylim(-2.5, 32.0)
                axp = sns.scatterplot(y_predict, y_actual, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Predicted", fontsize=18)
                plt.ylabel("Actual", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
    elif idx == 35:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between predicted and actual by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

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

            # Actual y
            y_actual = y.values.flatten().tolist()

            # Residuals Plot
            if i in [1, 2, 4, 5, 7, 8]:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-2.5, 22.5)
                plt.ylim(-2.5, 32.0)
                axp = sns.scatterplot(y_predict, y_actual, ax=ax)
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
                plt.ylim(-2.5, 32.0)
                axp = sns.scatterplot(y_predict, y_actual, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                axp.set(xlabel=None)
                plt.ylabel("Actual", fontsize=18)
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
                plt.ylim(-2.5, 32.0)
                axp = sns.scatterplot(y_predict, y_actual, ax=ax)
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
                plt.ylim(-2.5, 32.0)
                axp = sns.scatterplot(y_predict, y_actual, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Predicted", fontsize=18)
                plt.ylabel("Actual", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
    elif idx == 45:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between predicted and actual by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            if i in [0, 1, 2, 3, 4, 5]:
                if i in [0, 3]:
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-2.5, 22.5)
                    plt.ylim(-2.5, 32.0)
                    axp = sns.scatterplot()
                    plt.title(f'{date_list[i]}', fontsize=20)
                    axp.set(xlabel=None)
                    plt.ylabel("Actual", fontsize=18)
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
                    plt.ylim(-2.5, 32.0)
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

                # Actual y
                y_actual = y.values.flatten().tolist()

                # Residuals Plot
                if i in [1, 2, 4, 5, 7, 8]:
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-2.5, 22.5)
                    plt.ylim(-2.5, 32.0)
                    axp = sns.scatterplot(y_predict, y_actual, ax=ax)
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
                    plt.ylim(-2.5, 32.0)
                    axp = sns.scatterplot(y_predict, y_actual, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    axp.set(xlabel=None)
                    plt.ylabel("Actual", fontsize=18)
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
                    plt.ylim(-2.5, 32.0)
                    axp = sns.scatterplot(y_predict, y_actual, ax=ax)
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
                    plt.ylim(-2.5, 32.0)
                    axp = sns.scatterplot(y_predict, y_actual, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    plt.xlabel("Predicted", fontsize=18)
                    plt.ylabel("Actual", fontsize=18)
                    handles, labels = axp.get_legend_handles_labels()
                    list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                    list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                    labels = [x[1] for x in list_labels_handles]
                    handles = [x[0] for x in list_labels_handles]
                    ax.legend(handles, labels)
    elif idx in [16, 31, 33, 43]:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between predicted and actual by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            # Residuals Plot
            if i in [1, 2, 4, 5, 7, 8]:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-2.5, 22.5)
                plt.ylim(-2.5, 32.0)
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
                plt.ylim(-2.5, 32.0)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                axp.set(xlabel=None)
                plt.ylabel("Actual", fontsize=18)
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
                plt.ylim(-2.5, 32.0)
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
                plt.ylim(-2.5, 32.0)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Predicted", fontsize=18)
                plt.ylabel("Actual", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)

    result_root = os.path.join(root, 'result_plot_use2')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    result_user_root = os.path.join(result_root, f'{user}')
    if not os.path.isdir(result_user_root):
        os.makedirs(result_user_root)

    fig_name = result_user_root + '/' + f'{user}_mlr_day_predicted_actual.png'
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print(f'{user} 데이터 : MLR 일 단위 Predicted vs Actual Scatter Plot 종료')
    return

#####################################################

# 4. Scatter Plot : X1 vs Residual
def x1_resid_plot_day(user):
    print(f'{user} 데이터 : MLR 일 단위 X1 vs Residuals Scatter Plot 시작')
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
        plt.suptitle(f'Scatter Plot between GHI and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

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
                plt.xlim(-100.0, 9500.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.ghi, residuals, ax=ax)
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
                plt.xlim(-100.0, 9500.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.ghi, residuals, ax=ax)
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
                plt.xlim(-100.0, 9500.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.ghi, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("GHI(W/m^2)", fontsize=18)
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
                plt.xlim(-100.0, 9500.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.ghi, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("GHI(W/m^2)", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
    elif idx == 35:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between GHI and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

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
                plt.xlim(-100.0, 9500.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.ghi, residuals, ax=ax)
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
                plt.xlim(-100.0, 9500.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.ghi, residuals, ax=ax)
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
                plt.xlim(-100.0, 9500.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.ghi, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("GHI(W/m^2)", fontsize=18)
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
                plt.xlim(-100.0, 9500.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.ghi, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("GHI(W/m^2)", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
    elif idx == 45:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between GHI and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            if i in [0, 1, 2, 3, 4, 5]:
                if i in [0, 3]:
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-100.0, 9500.0)
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
                    plt.xlim(-100.0, 9500.0)
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
                    plt.xlim(-100.0, 9500.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.ghi, residuals, ax=ax)
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
                    plt.xlim(-100.0, 9500.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.ghi, residuals, ax=ax)
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
                    plt.xlim(-100.0, 9500.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.ghi, residuals, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    plt.xlabel("GHI(W/m^2)", fontsize=18)
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
                    plt.xlim(-100.0, 9500.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.ghi, residuals, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    plt.xlabel("GHI(W/m^2)", fontsize=18)
                    plt.ylabel("Residuals", fontsize=18)
                    handles, labels = axp.get_legend_handles_labels()
                    list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                    list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                    labels = [x[1] for x in list_labels_handles]
                    handles = [x[0] for x in list_labels_handles]
                    ax.legend(handles, labels)
    elif idx in [16, 31, 33, 43]:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between GHI and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            # Residuals Plot
            if i in [1, 2, 4, 5, 7, 8]:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-100.0, 9500.0)
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
                plt.xlim(-100.0, 9500.0)
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
                plt.xlim(-100.0, 9500.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("GHI(W/m^2)", fontsize=18)
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
                plt.xlim(-100.0, 9500.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("GHI(W/m^2)", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)

    result_root = os.path.join(root, 'result_plot_use2')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    result_user_root = os.path.join(result_root, f'{user}')
    if not os.path.isdir(result_user_root):
        os.makedirs(result_user_root)

    fig_name = result_user_root + '/' + f'{user}_mlr_day_ghi_residual.png'
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print(f'{user} 데이터 : MLR 일 단위 X1 vs Residuals Scatter Plot 종료')
    return

#####################################################

# 5. Scatter Plot : X2 vs Residual
def x2_resid_plot_day(user):
    print(f'{user} 데이터 : MLR 일 단위 X2 vs Residuals Scatter Plot 시작')
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
        plt.suptitle(f'Scatter Plot between Temperature and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

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
                plt.xlim(-12.5, 45.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.temperature, residuals, ax=ax)
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
                plt.xlim(-12.5, 45.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.temperature, residuals, ax=ax)
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
                plt.xlim(-12.5, 45.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.temperature, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Temperature(C)", fontsize=18)
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
                plt.xlim(-12.5, 45.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.temperature, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Temperature(C)", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
    elif idx == 35:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between Temperature and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

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
                plt.xlim(-12.5, 45.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.temperature, residuals, ax=ax)
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
                plt.xlim(-12.5, 45.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.temperature, residuals, ax=ax)
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
                plt.xlim(-12.5, 45.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.temperature, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Temperature(C)", fontsize=18)
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
                plt.xlim(-12.5, 45.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.temperature, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Temperature(C)", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
    elif idx == 45:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between Temperature and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            if i in [0, 1, 2, 3, 4, 5]:
                if i in [0, 3]:
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-12.5, 45.0)
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
                    plt.xlim(-12.5, 45.0)
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
                    plt.xlim(-12.5, 45.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.temperature, residuals, ax=ax)
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
                    plt.xlim(-12.5, 45.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.temperature, residuals, ax=ax)
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
                    plt.xlim(-12.5, 45.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.temperature, residuals, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    plt.xlabel("Temperature(C)", fontsize=18)
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
                    plt.xlim(-12.5, 45.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.temperature, residuals, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    plt.xlabel("Temperature(C)", fontsize=18)
                    plt.ylabel("Residuals", fontsize=18)
                    handles, labels = axp.get_legend_handles_labels()
                    list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                    list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                    labels = [x[1] for x in list_labels_handles]
                    handles = [x[0] for x in list_labels_handles]
                    ax.legend(handles, labels)
    elif idx in [16, 31, 33, 43]:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between Temperature and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            # Residuals Plot
            if i in [1, 2, 4, 5, 7, 8]:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-12.5, 45.0)
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
                plt.xlim(-12.5, 45.0)
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
                plt.xlim(-12.5, 45.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Temperature(C)", fontsize=18)
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
                plt.xlim(-12.5, 45.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Temperature(C)", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)

    result_root = os.path.join(root, 'result_plot_use2')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    result_user_root = os.path.join(result_root, f'{user}')
    if not os.path.isdir(result_user_root):
        os.makedirs(result_user_root)

    fig_name = result_user_root + '/' + f'{user}_mlr_day_temp_residual.png'
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print(f'{user} 데이터 : MLR 시간 단위 X2 vs Residuals Scatter Plot 종료')
    return

#####################################################

# 6. Scatter Plot : X3 vs Residual
def x3_resid_plot_day(user):
    print(f'{user} 데이터 : MLR 일 단위 X3 vs Residuals Scatter Plot 시작')
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
        plt.suptitle(f'Scatter Plot between Visibility and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

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
                plt.xlim(-100.0, 5100.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.visibility, residuals, ax=ax)
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
                plt.xlim(-100.0, 5100.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.visibility, residuals, ax=ax)
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
                plt.xlim(-100.0, 5100.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.visibility, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Visibility(10m)", fontsize=18)
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
                plt.xlim(-100.0, 5100.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.visibility, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Visibility(10m)", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
    elif idx == 35:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between Visibility and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

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
                plt.xlim(-100.0, 5100.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.visibility, residuals, ax=ax)
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
                plt.xlim(-100.0, 5100.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.visibility, residuals, ax=ax)
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
                plt.xlim(-100.0, 5100.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.visibility, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Visibility(10m)", fontsize=18)
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
                plt.xlim(-100.0, 5100.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot(x.visibility, residuals, ax=ax)
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Visibility(10m)", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)
    elif idx == 45:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between Visibility and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            if i in [0, 1, 2, 3, 4, 5]:
                if i in [0, 3]:
                    ax = plt.subplot(4, 3, i + 1)
                    plt.subplots_adjust(wspace=0.1, hspace=0.2)
                    plt.xlim(-100.0, 5100.0)
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
                    plt.xlim(-100.0, 5100.0)
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
                    plt.xlim(-100.0, 5100.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.visibility, residuals, ax=ax)
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
                    plt.xlim(-100.0, 5100.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.visibility, residuals, ax=ax)
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
                    plt.xlim(-100.0, 5100.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.visibility, residuals, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    plt.xlabel("Visibility(10m)", fontsize=18)
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
                    plt.xlim(-100.0, 5100.0)
                    plt.ylim(-12.5, 12.5)
                    axp = sns.scatterplot(x.visibility, residuals, ax=ax)
                    plt.title(f'{date_list[i]}', fontsize=20)
                    plt.xlabel("Visibility(10m)", fontsize=18)
                    plt.ylabel("Residuals", fontsize=18)
                    handles, labels = axp.get_legend_handles_labels()
                    list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                    list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                    labels = [x[1] for x in list_labels_handles]
                    handles = [x[0] for x in list_labels_handles]
                    ax.legend(handles, labels)
    elif idx in [16, 31, 33, 43]:
        sns.set(rc={'figure.figsize': (20, 24)})
        plt.suptitle(f'Scatter Plot between Visibility and residuals by day for household No.{idx} site',
                     y=0.92, fontsize=22, fontweight='bold')

        for i in range(len(date_list)):
            # Year/Month Filtering
            df_user_filter = df_user[df_user.ym == date_list[i]]

            # Residuals Plot
            if i in [1, 2, 4, 5, 7, 8]:
                ax = plt.subplot(4, 3, i + 1)
                plt.subplots_adjust(wspace=0.1, hspace=0.2)
                plt.xlim(-100.0, 5100.0)
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
                plt.xlim(-100.0, 5100.0)
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
                plt.xlim(-100.0, 5100.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Visibility(10m)", fontsize=18)
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
                plt.xlim(-100.0, 5100.0)
                plt.ylim(-12.5, 12.5)
                axp = sns.scatterplot()
                plt.title(f'{date_list[i]}', fontsize=20)
                plt.xlabel("Visibility(10m)", fontsize=18)
                plt.ylabel("Residuals", fontsize=18)
                handles, labels = axp.get_legend_handles_labels()
                list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
                list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
                labels = [x[1] for x in list_labels_handles]
                handles = [x[0] for x in list_labels_handles]
                ax.legend(handles, labels)

    result_root = os.path.join(root, 'result_plot_use2')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    result_user_root = os.path.join(result_root, f'{user}')
    if not os.path.isdir(result_user_root):
        os.makedirs(result_user_root)

    fig_name = result_user_root + '/' + f'{user}_mlr_day_vis_residual.png'
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print(f'{user} 데이터 : MLR 시간 단위 X3 vs Residuals Scatter Plot 종료')
    return
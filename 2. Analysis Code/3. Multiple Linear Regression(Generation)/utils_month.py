### Code : Utils by month
### Writer : Donghyeon Kim
### Date : 2022.08.19

# Multiple Linear Regression #
# 단위 : 1달(1개월)
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
def pred_act_plot_month(user):
    print(f'{user} 데이터 : MLR 월 단위 Predicted vs Actual Scatter Plot 시작')
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

        # Actual y
        y_actual = y.values.flatten().tolist()

        # Residuals Plot
        sns.set(rc={'figure.figsize': (14, 9)})
        sns.scatterplot(y_predict, y_actual)
        plt.title(f'Scatter Plot between predicted and actual by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("Predicted", fontsize=20)
        plt.ylabel("Actual", fontsize=20)
        plt.xlim(-50.0, 550.0)
        plt.ylim(-50.0, 630.0)
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

        # Actual y
        y_actual = y.values.flatten().tolist()

        # Residuals Plot
        sns.set(rc={'figure.figsize': (14, 9)})
        sns.scatterplot(y_predict, y_actual)
        plt.title(f'Scatter Plot between predicted and actual by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("Predicted", fontsize=20)
        plt.ylabel("Actual", fontsize=20)
        plt.xlim(-50.0, 550.0)
        plt.ylim(-50.0, 630.0)
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

        # Actual y
        # y_actual = y.values.flatten().tolist()

        # Residuals Plot
        sns.set(rc={'figure.figsize': (14, 9)})
        sns.scatterplot()
        plt.title(f'Scatter Plot between predicted and actual by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("Predicted", fontsize=20)
        plt.ylabel("Actual", fontsize=20)
        plt.xlim(-50.0, 550.0)
        plt.ylim(-50.0, 630.0)

    result_root = os.path.join(root, 'result_plot_use2')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    result_user_root = os.path.join(result_root, f'{user}')
    if not os.path.isdir(result_user_root):
        os.makedirs(result_user_root)

    fig_name = result_user_root + '/' + f'{user}_mlr_month_predicted_actual.png'
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print(f'{user} 데이터 : MLR 월 단위 Predicted vs Actual Scatter Plot 종료')
    return

#####################################################

# 4. Scatter Plot : X1 vs Residual
def x1_resid_plot_month(user):
    print(f'{user} 데이터 : MLR 월 단위 X1 vs Residuals Scatter Plot 시작')
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
        sns.scatterplot(x.ghi, residuals)
        plt.title(f'Scatter Plot between GHI and residuals by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("GHI(W/m^2)", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-10000.0, 165000.0)
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
        sns.scatterplot(x.ghi, residuals)
        plt.title(f'Scatter Plot between GHI and residuals by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("GHI(W/m^2)", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-10000.0, 165000.0)
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
        plt.title(f'Scatter Plot between GHI and residuals by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("GHI(W/m^2)", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-10000.0, 165000.0)
        plt.ylim(-75.0, 75.0)

    result_root = os.path.join(root, 'result_plot_use2')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    result_user_root = os.path.join(result_root, f'{user}')
    if not os.path.isdir(result_user_root):
        os.makedirs(result_user_root)

    fig_name = result_user_root + '/' + f'{user}_mlr_month_ghi_residual.png'
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print(f'{user} 데이터 : MLR 월 단위 X1 vs Residuals Scatter Plot 종료')
    return

#####################################################

# 5. Scatter Plot : X2 vs Residual
def x2_resid_plot_month(user):
    print(f'{user} 데이터 : MLR 월 단위 X2 vs Residuals Scatter Plot 시작')
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
        sns.scatterplot(x.temperature, residuals)
        plt.title(f'Scatter Plot between Temperature and residuals by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("Temperature(C)", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-12.5, 45.0)
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
        sns.scatterplot(x.temperature, residuals)
        plt.title(f'Scatter Plot between Temperature and residuals by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("Temperature(C)", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-12.5, 45.0)
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
        plt.title(f'Scatter Plot between Temperature and residuals by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("Temperature(C)", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-12.5, 45.0)
        plt.ylim(-75.0, 75.0)

    result_root = os.path.join(root, 'result_plot_use2')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    result_user_root = os.path.join(result_root, f'{user}')
    if not os.path.isdir(result_user_root):
        os.makedirs(result_user_root)

    fig_name = result_user_root + '/' + f'{user}_mlr_month_temp_residual.png'
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print(f'{user} 데이터 : MLR 월 단위 X2 vs Residuals Scatter Plot 종료')
    return

#####################################################

# 6. Scatter Plot : X3 vs Residual
def x3_resid_plot_month(user):
    print(f'{user} 데이터 : MLR 월 단위 X3 vs Residuals Scatter Plot 시작')
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
        sns.scatterplot(x.visibility, residuals)
        plt.title(f'Scatter Plot between Visibility and residuals by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("Visibility(10m)", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-100.0, 4500.0)
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
        sns.scatterplot(x.visibility, residuals)
        plt.title(f'Scatter Plot between Visibility and residuals by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("Visibility(10m)", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-100.0, 4500.0)
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
        plt.title(f'Scatter Plot between Visibility and residuals by month for household No.{idx} site',
                  fontsize=22, fontweight='bold')
        plt.xlabel("Visibility(10m)", fontsize=20)
        plt.ylabel("Residuals", fontsize=20)
        plt.xlim(-100.0, 4500.0)
        plt.ylim(-75.0, 75.0)

    result_root = os.path.join(root, 'result_plot_use2')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    result_user_root = os.path.join(result_root, f'{user}')
    if not os.path.isdir(result_user_root):
        os.makedirs(result_user_root)

    fig_name = result_user_root + '/' + f'{user}_mlr_month_vis_residual.png'
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print(f'{user} 데이터 : MLR 월 단위 X3 vs Residuals Scatter Plot 종료')
    return
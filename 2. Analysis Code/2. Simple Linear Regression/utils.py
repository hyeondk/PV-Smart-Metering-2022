### Code : Function Utils
### Writer : Donghyeon Kim
### Date : 2022.08.02

# 0. 라이브러리 설정
from pathlib import Path
import os
import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns

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
    df_user_filter = df_user.drop(['date'], axis = 1)

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
            prec_group.append('no')
        else:
            prec_group.append('yes')

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

# 7. Scatter Plot : 시간(Hour) - 발전량(Solar Power Generation)
def hour_generation_plot(user):
    print(f'{user} 데이터 : 시간(Hour)-발전량(Solar Power Generation) Scatter Plot 시작')
    # 루트 설정
    root = get_project_root()

    # 결합된 데이터 호출
    df_weather_use = weather_user_merge(user)

    # 사용자 index 추출
    user_name = get_name_root()
    idx = user_name.index(user) + 1

    # 날짜 : 2021/4 ~ 2022/3
    date_list = df_weather_use.ym.unique().tolist()
    date_list = date_list[1:-1]  # 2021/3, 2022/4 제외

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f'Scatter Plot between solar power generation and hour for household No.{idx} site',
                 y=0.92, fontsize=18, fontweight='bold')

    color_dict = dict({'no': 'dodgerblue', 'yes': 'red'})

    for i in range(len(date_list)):
        df_weather_use_f = df_weather_use[df_weather_use.ym == date_list[i]]

        if i not in [9, 10, 11]:
            ax = plt.subplot(4, 3, i + 1)
            plt.subplots_adjust(hspace=0.25)
            plt.xlim(-1.0, 24.0)
            plt.ylim(-0.2, 3.0)
            axp = sns.scatterplot(x='hour', y='yield_kWh', hue='status', data=df_weather_use_f, ax=ax,
                                  palette=color_dict)
            plt.title(f'{date_list[i]}', fontsize=15)
            plt.xlabel("Hour", fontsize=13)
            plt.ylabel("Generation(kWh)", fontsize=13)
            plt.tick_params('x', labelbottom=False)
            handles, labels = axp.get_legend_handles_labels()
            list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
            list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
            labels = [x[1] for x in list_labels_handles]
            handles = [x[0] for x in list_labels_handles]
            ax.legend(handles, labels)
        else:
            ax = plt.subplot(4, 3, i + 1)
            plt.subplots_adjust(hspace=0.25)
            plt.xlim(-1.0, 24.0)
            plt.ylim(-0.2, 3.0)
            axp = sns.scatterplot(x='hour', y='yield_kWh', hue='status', data=df_weather_use_f, ax=ax,
                                  palette=color_dict)
            plt.title(f'{date_list[i]}', fontsize=15)
            plt.xlabel("Hour", fontsize=13)
            plt.ylabel("Generation(kWh)", fontsize=13)
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

    fig_name = result_user_root + '/' + f'{user}_generation_hour.png'
    plt.savefig(fig_name, dpi=300)
    print(f'{user} 데이터 : 시간(Hour)-발전량(Solar Power Generation) Scatter Plot 종료')
    return

# 8. Scatter Plot : 일사량(GHI) - 발전량(Solar Power Generation)
def ghi_generation_plot(user):
    print(f'{user} 데이터 : 일사량(GHI)-발전량(Solar Power Generation) Scatter Plot 시작')
    # 루트 설정
    root = get_project_root()

    # 결합된 데이터 호출
    df_weather_use = weather_user_merge(user)

    # 사용자 index 추출
    user_name = get_name_root()
    idx = user_name.index(user) + 1

    # 날짜 : 2021/4 ~ 2022/3
    date_list = df_weather_use.ym.unique().tolist()
    date_list = date_list[1:-1]  # 2021/3, 2022/4 제외

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f'Scatter Plot between solar power generation and GHI for household No.{idx} site',
                 y=0.92, fontsize=18, fontweight='bold')

    color_dict = dict({'no': 'dodgerblue', 'yes': 'red'})

    for i in range(len(date_list)):
        df_weather_use_f = df_weather_use[df_weather_use.ym == date_list[i]]

        if i not in [9, 10, 11]:
            ax = plt.subplot(4, 3, i + 1)
            plt.subplots_adjust(hspace=0.25)
            plt.xlim(-20.0, 900.0)
            plt.ylim(-0.2, 3.0)
            axp = sns.scatterplot(x='ghi', y='yield_kWh', hue='status', data=df_weather_use_f, ax=ax,
                                  palette=color_dict)
            plt.title(f'{date_list[i]}', fontsize=15)
            plt.xlabel("GHI(W/m^2)", fontsize=13)
            plt.ylabel("Generation(kWh)", fontsize=13)
            plt.tick_params('x', labelbottom=False)
            handles, labels = axp.get_legend_handles_labels()
            list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
            list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
            labels = [x[1] for x in list_labels_handles]
            handles = [x[0] for x in list_labels_handles]
            ax.legend(handles, labels)
        else:
            ax = plt.subplot(4, 3, i + 1)
            plt.subplots_adjust(hspace=0.25)
            plt.xlim(-20.0, 900.0)
            plt.ylim(-0.2, 3.0)
            axp = sns.scatterplot(x='ghi', y='yield_kWh', hue='status', data=df_weather_use_f, ax=ax,
                                  palette=color_dict)
            plt.title(f'{date_list[i]}', fontsize=15)
            plt.xlabel("GHI(W/m^2)", fontsize=13)
            plt.ylabel("Generation(kWh)", fontsize=13)
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

    fig_name = result_user_root + '/' + f'{user}_generation_ghi.png'
    plt.savefig(fig_name, dpi=300)
    print(f'{user} 데이터 : 일사량(GHI)-발전량(Solar Power Generation) Scatter Plot 종료')
    return


# 9. Scatter Plot : 시간(Hour) - 에너지 소비(Energy Consumption)
def hour_consumption_plot(user):
    print(f'{user} 데이터 : 시간(Hour)-에너지 소비(Energy Consumption) Scatter Plot 시작')
    # 루트 설정
    root = get_project_root()

    # 결합된 데이터 호출
    df_weather_use = weather_user_merge(user)

    # 사용자 index 추출
    user_name = get_name_root()
    idx = user_name.index(user) + 1

    # 날짜 : 2021/4 ~ 2022/3
    date_list = df_weather_use.ym.unique().tolist()
    date_list = date_list[1:-1]  # 2021/3, 2022/4 제외

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f'Scatter Plot between energy consumption and hour for household No.{idx} site',
                 y=0.92, fontsize=18, fontweight='bold')

    color_dict = dict({'no': 'dodgerblue', 'yes': 'red'})

    for i in range(len(date_list)):
        df_weather_use_f = df_weather_use[df_weather_use.ym == date_list[i]]

        if i not in [9, 10, 11]:
            ax = plt.subplot(4, 3, i + 1)
            plt.subplots_adjust(hspace=0.25)
            plt.xlim(-1.0, 24.0)
            plt.ylim(-0.2, 3.0)
            axp = sns.scatterplot(x='hour', y='grid_kWh', hue='status', data=df_weather_use_f, ax=ax,
                                  palette=color_dict)
            plt.title(f'{date_list[i]}', fontsize=15)
            plt.xlabel("Hour", fontsize=13)
            plt.ylabel("Consumption(kWh)", fontsize=13)
            plt.tick_params('x', labelbottom=False)
            handles, labels = axp.get_legend_handles_labels()
            list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
            list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
            labels = [x[1] for x in list_labels_handles]
            handles = [x[0] for x in list_labels_handles]
            ax.legend(handles, labels)
        else:
            ax = plt.subplot(4, 3, i + 1)
            plt.subplots_adjust(hspace=0.25)
            plt.xlim(-1.0, 24.0)
            plt.ylim(-0.2, 3.0)
            axp = sns.scatterplot(x='hour', y='grid_kWh', hue='status', data=df_weather_use_f, ax=ax,
                                  palette=color_dict)
            plt.title(f'{date_list[i]}', fontsize=15)
            plt.xlabel("Hour", fontsize=13)
            plt.ylabel("Consumption(kWh)", fontsize=13)
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

    fig_name = result_user_root + '/' + f'{user}_consumption_hour.png'
    plt.savefig(fig_name, dpi=300)
    print(f'{user} 데이터 : 시간(Hour)-에너지 소비(Energy Consumption) Scatter Plot 종료')
    return

# 10. Scatter Plot : 일사량(GHI) - 에너지 소비(Energy Consumption)
def ghi_consumption_plot(user):
    print(f'{user} 데이터 : 일사량(GHI)-에너지 소비(Energy Consumption) Scatter Plot 시작')
    # 루트 설정
    root = get_project_root()

    # 결합된 데이터 호출
    df_weather_use = weather_user_merge(user)

    # 사용자 index 추출
    user_name = get_name_root()
    idx = user_name.index(user) + 1

    # 날짜 : 2021/4 ~ 2022/3
    date_list = df_weather_use.ym.unique().tolist()
    date_list = date_list[1:-1]  # 2021/3, 2022/4 제외

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f'Scatter Plot between energy consumption and GHI for household No.{idx} site',
                 y=0.92, fontsize=18, fontweight='bold')

    color_dict = dict({'no': 'dodgerblue', 'yes': 'red'})

    for i in range(len(date_list)):
        df_weather_use_f = df_weather_use[df_weather_use.ym == date_list[i]]

        if i not in [9, 10, 11]:
            ax = plt.subplot(4, 3, i + 1)
            plt.subplots_adjust(hspace=0.25)
            plt.xlim(-20.0, 900.0)
            plt.ylim(-0.2, 3.0)
            axp = sns.scatterplot(x='ghi', y='grid_kWh', hue='status', data=df_weather_use_f, ax=ax, palette=color_dict)
            plt.title(f'{date_list[i]}', fontsize=15)
            plt.xlabel("GHI(W/m^2)", fontsize=13)
            plt.ylabel("Consumption(kWh)", fontsize=13)
            plt.tick_params('x', labelbottom=False)
            handles, labels = axp.get_legend_handles_labels()
            list_labels_handles = [(h, v) for h, v in zip(handles, labels)]
            list_labels_handles = sorted(list_labels_handles, key=lambda x: x[1])
            labels = [x[1] for x in list_labels_handles]
            handles = [x[0] for x in list_labels_handles]
            ax.legend(handles, labels)
        else:
            ax = plt.subplot(4, 3, i + 1)
            plt.subplots_adjust(hspace=0.25)
            plt.xlim(-20.0, 900.0)
            plt.ylim(-0.2, 3.0)
            axp = sns.scatterplot(x='ghi', y='grid_kWh', hue='status', data=df_weather_use_f, ax=ax, palette=color_dict)
            plt.title(f'{date_list[i]}', fontsize=15)
            plt.xlabel("GHI(W/m^2)", fontsize=13)
            plt.ylabel("Consumption(kWh)", fontsize=13)
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

    fig_name = result_user_root + '/' + f'{user}_consumption_ghi.png'
    plt.savefig(fig_name, dpi=300)
    print(f'{user} 데이터 : 일사량(GHI)-에너지 소비(Energy Consumption) Scatter Plot 종료')
    return

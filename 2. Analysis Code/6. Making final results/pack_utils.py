### Code : Function Utils
### Writer : Donghyeon Kim
### Update : 2022.11.06

# 0. 라이브러리 설정
from pathlib import Path
import os

# 1. 파일의 상위-상위 경로 설정
def get_project_root() -> Path:
    return Path(__file__).parent.parent

# 2. 전체 사용자 이름 획득
def get_name_root():
    root = get_project_root()
    folder_root = os.path.join(root, 'data') # 루트 + 'data' 경로 설정
    user_name = os.listdir(folder_root) # 경로 안에 있는 사용자명
    return user_name

# 3. 태양광 사용자 이름 획득(NA 많은 사용자 제외)
def get_name_root_use():
    name_list = ['김OO', '서OO', '송OO', '오OO', '이OO',
                 '이OO', '임OO', '조OO', '최OO', '최OO']
    return name_list

# 3-2. 태양광 사용자 이름(단독주택 한정)
def get_name_root_use2():
    name_list = ['김OO', '서OO', '송OO', '오OO', '이OO',
                 '이OO', '임OO', '조OO', '최OO', '최OO']
    return name_list

# 3-3. 태양광 사용자 이름(3kW 한정)
def get_name_root_use3():
    name_list = ['김OO', '서OO', '송OO', '오OO',
                 '이OO', '임OO', '조OO', '최OO']
    return name_list

# 4. 태양광 미사용자 이름 획득
def get_name_root_not():
    name_list = ['강OO', '고OO', '고OO', '구OO', '김OO', '김OO', '김OO', '명OO', '문OO', '박OO',
                 '박OO', '박OO', '박OO', '백OO', '손OO', '양OO', '양OO', '양OO', '윤OO', '이OO',
                 '이OO', '이OO', '이OO', '최OO', '최OO']
    return name_list

# 4-2. 태양광 미사용자 이름(단독주택 한정)
def get_name_root_not2():
    name_list = ['고OO', '고OO', '구OO', '김OO', '김OO', '김OO', '문OO', '박OO', '박OO',
                 '손OO', '양OO', '양OO', '양OO', '윤OO', '이OO', '이OO', '이OO', '최OO']
    return name_list

# 5. 설비용량(kW) Dictionary
def kw_dict(user):
    use_dict = {'김OO': '3kW',
                '서OO': '3kW',
                '송OO': '3kW',
                '오OO': '3kW',
                '이OO': '300W',
                '이OO': '3kW',
                '임OO': '3kW',
                '조OO': '3kW',
                '최OO': '3kW',
                '최OO': '18kW'}
    item = use_dict.get(user)
    return item

# 6. 설비용량(kW) 숫자 Dictionary
def kw_value_dict(user):
    use_dict = {'김OO': 3,
                '서OO': 3,
                '송OO': 3,
                '오OO': 3,
                '이OO': 0.3,
                '이OO': 3,
                '임OO': 3,
                '조OO': 3,
                '최OO': 3,
                '최OO': 18}
    item = use_dict.get(user)
    return item

###########################################

# 사용자 최종 정리 #

# 1. 태양광 사용자 최종 이름
# 모두 단독주택이라는 공통점이 있음.
def get_name_use_final():
    name_list = ['김OO', '서OO', '송OO', '오OO', '이OO',
                 '이OO', '임OO', '조OO', '최OO', '최OO', '윤OO']
    return name_list

# 2. 태양광 사용자 최종 이름 중 3kW
def get_name_use_final_3kw():
    name_list = ['김OO', '서OO', '송OO', '오OO', '이OO',
                 '임OO', '조OO', '최OO', '윤OO']
    return name_list

# 3. 태양광 미사용자 최종 이름
# kW 구분은 없으나, 주택 형태는 다양함.
def get_name_not_final():
    name_list = ['강OO', '고OO', '고OO', '구OO', '김OO', '김OO', '김OO', '명OO', '문OO', '박OO',
                 '박OO', '박OO', '박OO', '백OO', '손OO', '양OO', '양OO', '양OO', '윤OO', '이OO',
                 '이OO', '이OO', '이OO', '최OO', '최OO']
    return name_list

# 4. 태양광 미사용자 최종 이름 중 단독주택
# 4-2. 태양광 미사용자 이름(단독주택 한정)
def get_name_not_final_detach():
    name_list = ['고OO', '고OO', '구OO', '김OO', '김OO', '김OO', '문OO', '박OO', '박OO',
                 '손OO', '양OO', '양OO', '양OO', '윤OO', '이OO', '이OO', '이OO', '최OO']
    return name_list

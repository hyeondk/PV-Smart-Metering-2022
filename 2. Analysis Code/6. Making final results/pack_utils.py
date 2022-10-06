### Code : Function Utils
### Writer : Donghyeon Kim

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
    name_list = ['김풍옥', '서구자', '송순단', '오미옥', '이미자',
                 '이재철(김정겸)', '임명란', '조완숙', '최영해', '최차복']
    return name_list

# 3-2. 태양광 사용자 이름(단독주택 한정)
def get_name_root_use2():
    name_list = ['김풍옥', '서구자', '송순단', '오미옥', '이미자',
                 '이재철(김정겸)', '임명란', '조완숙', '최영해', '최차복']
    return name_list

# 3-3. 태양광 사용자 이름(3kW 한정)
def get_name_root_use3():
    name_list = ['김풍옥', '서구자', '송순단', '오미옥',
                 '이재철(김정겸)', '임명란', '조완숙', '최영해']
    return name_list

# 4. 태양광 미사용자 이름 획득
def get_name_root_not():
    name_list = ['강혜지', '고병욱', '고영준', '구규승', '김기용', '김소영', '김옥희', '명매희', '문선미', '박경희',
                 '박은영', '박재균', '박희정', '백현미', '손창숙', '양명자', '양정열', '양희상', '윤봉희', '이봉선',
                 '이수진', '이영자', '이정애', '최영남', '최은영']
    return name_list

# 4-2. 태양광 미사용자 이름(단독주택 한정)
def get_name_root_not2():
    name_list = ['고병욱', '고영준', '구규승', '김기용', '김소영', '김옥희', '문선미', '박재균', '박희정',
                 '손창숙', '양명자', '양정열', '양희상', '윤봉희', '이봉선', '이영자', '이정애', '최영남']
    return name_list

# 5. 설비용량(kW) Dictionary
def kw_dict(user):
    use_dict = {'김풍옥': '3kW',
                '서구자': '3kW',
                '송순단': '3kW',
                '오미옥': '3kW',
                '이미자': '300W',
                '이재철(김정겸)': '3kW',
                '임명란': '3kW',
                '조완숙': '3kW',
                '최영해': '3kW',
                '최차복': '18kW'}
    item = use_dict.get(user)
    return item

# 6. 설비용량(kW) 숫자 Dictionary
def kw_value_dict(user):
    use_dict = {'김풍옥': 3,
                '서구자': 3,
                '송순단': 3,
                '오미옥': 3,
                '이미자': 0.3,
                '이재철(김정겸)': 3,
                '임명란': 3,
                '조완숙': 3,
                '최영해': 3,
                '최차복': 18}
    item = use_dict.get(user)
    return item

import json
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_df(filename):
    ##  날짜 / 플레이어 이름 / 승패 / flair(0 또는 1) / degree / score / point
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataframes = []

    for key, json_data in tqdm(data.items(), desc='Reading JSON', unit=' keys'):
        # JSON 데이터를 데이터프레임으로 변환
        df = pd.DataFrame(json_data['players'])

        # Unix 시간을 datetime 객체로 변환
        date = json_data['date']

        # 필요한 열만 선택
        df = df[['name', 'team', 'flair', 'score', 'points', 'degree', 'auth']]
        df['date'] = datetime.datetime.utcfromtimestamp(date)

        # team 열의 값을 Red 또는 Blue로 변환
        df['team'] = df['team'].apply(lambda x: 'Red' if x == 1 else 'Blue')

        # 승패 정보를 계산하여 추가
        winning_team = 'Red' if json_data['teams'][0]['score'] > json_data['teams'][1]['score'] else 'Blue'
        df['win'] = (df['team'] == winning_team).astype(int)

        # flair 열 변환
        df['flair'] = df['flair'].apply(lambda x: 0 if x == 0 else 1)

        # df['auth'] = df['auth'].astype(int)
        df = df[df['auth']]

        dataframes.append(df)

    # dataframes 리스트에 있는 모든 데이터프레임을 수직으로 연결하여 하나의 데이터프레임으로 만듭니다.
    combined_df = pd.concat(dataframes, ignore_index=True)

    ## 결과 데이터프레임 출력
    print(combined_df)

    ## count same name
    print(combined_df['name'].value_counts())
    return combined_df


def filter_df(df, activation_period=7, target_period=7, target_column='played_next_7_days'):
    df_copy = df.copy()

    # 이탈 계산 ------------------------------------------------------------------------------------------
    # 각 유저의 처음 접속 날짜 찾기
    df_copy['first_login_date'] = df_copy.groupby('name')['date'].transform('min')
    df_copy[target_column] = 0

    # 최초 로그인 + activation_period + target_period 이후 데이터 제거
    activation_period = datetime.timedelta(days=activation_period)
    target_period = datetime.timedelta(days=target_period)

    df_copy = df_copy[df_copy['date'] < df_copy['first_login_date'] + activation_period + target_period]

    # target_period 기간 중 접속 여부
    start_date = df_copy['first_login_date'] + activation_period
    end_date = start_date + target_period
    condition = (df_copy['date'] > start_date) & (df_copy['date'] < end_date)
    df_copy[target_column] = condition.astype(int)

    # 승률 계산 ------------------------------------------------------------------------------------------
    # 연속된 승/패 횟수 계산
    df_copy['start_of_streak'] = df_copy.groupby('name')['win'].diff().ne(0)
    df_copy['streak_id'] = df_copy.groupby('name')['win'].cumsum()
    df_copy['streak'] = df_copy.groupby(['name', 'streak_id']).cumcount() + 1
    df_copy['streak_id'] = df_copy.groupby('name')['start_of_streak'].cumsum()
    df_copy['streak_counter'] = df_copy.groupby(['name', 'streak_id']).cumcount() + 1

    # 연승, 연패
    df_copy['winning_streak'] = df_copy['streak_counter'] * df_copy['win']
    df_copy['losing_streak'] = df_copy['streak_counter'] * (1 - df_copy['win'])

    df_copy['auth'] = df_copy['auth'].astype(int)
    df_copy['win_count'] = df_copy['win']
    df_copy['lose_count'] = 1 - df_copy['win']

    # 필요한 칼럼 선택 ------------------------------------------------------------------------------------
    selected_columns = ['name', 'score', 'points', 'degree', 'win', 'win_count', 'lose_count', 'winning_streak',
                        'losing_streak', target_column]

    # 결과 데이터프레임 출력
    result_df = df_copy[selected_columns]

    # flair, score, points, degree 열은  max 또는 mean 계산
    result_df = result_df.groupby('name').agg(
        {'score': 'mean', 'points': 'mean', 'degree': 'mean', 'win': 'mean', 'win_count': 'sum', 'lose_count': 'sum',
         'winning_streak': 'max',
         'losing_streak': 'max',
         target_column: 'max'}).reset_index()
    result_df.drop('name', axis=1, inplace=True)

    # 결과 데이터프레임 출력
    return result_df


def data_split(df, t_col, test_size, random_state=123456):
    X = df.drop(t_col, axis='columns')
    y = df[[t_col]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X, y, X_train, X_test, y_train, y_test

import json
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_df(filename):
    ##  날짜 / 플레이어 이름 / 승패 / flair(0 또는 1) / degree / score / point
    with open(filename, 'r') as f:
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

        df = df[df['auth']]

        dataframes.append(df)

    # dataframes 리스트에 있는 모든 데이터프레임을 수직으로 연결하여 하나의 데이터프레임으로 만듭니다.
    combined_df = pd.concat(dataframes, ignore_index=True)

    ## 결과 데이터프레임 출력
    print(combined_df)

    ## count same name
    print(combined_df['name'].value_counts())
    return combined_df;


def filter(df, activation_period=7, target_period=7, target_column='played_next_7_days'):
    # 각 유저의 처음 로그인 날짜 찾기
    first_login_dates = df.groupby('name')['date'].min().reset_index()

    # 처음 로그인 날짜를 기준으로 시작 날짜 계산
    first_login_dates['start_date'] = first_login_dates['date']
    first_login_dates['end_date'] = first_login_dates['start_date'] + datetime.timedelta(days=activation_period)

    # 다음 7일 동안 활동한 유저 필터링
    filtered_users = df[df['name'].isin(first_login_dates['name'].tolist())]
    filtered_users = filtered_users[filtered_users['date'] >= first_login_dates['start_date'].min()]
    filtered_users = filtered_users[filtered_users['date'] <= first_login_dates['end_date'].max()]

    # 각 유저가 다음 7일 동안에 플레이한 여부를 나타내는 칼럼 추가 (0 또는 1)
    filtered_users[target_column] = 0  # 기본값으로 0 설정

    # 다음 7일 동안에 활동한 유저 목록
    active_users_within_next_7_days = df[
        (df['date'] >= first_login_dates['end_date'].min()) &
        (df['date'] < first_login_dates['end_date'].max() + datetime.timedelta(days=target_period))
        ]['name'].unique()

    # 다음 7일 동안에 활동한 유저인 경우 target_column 칼럼 값을 1로 설정
    filtered_users.loc[filtered_users['name'].isin(active_users_within_next_7_days), target_column] = 1

    ## 결과 데이터프레임 출력
    print(filtered_users.keys())

    ## 필요한 칼럼 선택 및 처리

    # 필요한 칼럼 선택
    selected_columns = ['name', 'score', 'points', 'degree', 'flair', target_column]

    # 결과 데이터프레임 출력
    result_df = filtered_users[selected_columns]

    # flair, score, points, degree 열은 그룹별로 max 또는 mean 계산
    result_df = result_df.groupby('name').agg(
        {'flair': 'max', 'score': 'mean', 'points': 'mean', 'degree': 'mean',
         target_column: 'max'}).reset_index()
    result_df.drop('name', axis=1, inplace=True)

    # 결과 데이터프레임 출력
    print(result_df)
    return result_df


def data_split(df, t_col, test_size, random_state=123456):
    X = df.drop(t_col, axis='columns')
    y = df[[t_col]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X, y, X_train, X_test, y_train, y_test

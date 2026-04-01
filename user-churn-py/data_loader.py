import json
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


def filter_df(df, activation_period=7, churn_observation_period=7, churn_column='played_next_7_days'):
    df_copy = df.copy()

    # 각 유저의 처음 접속 날짜 찾기
    df_copy['first_login_date'] = df_copy.groupby('name')['date'].transform('min')

    ap = datetime.timedelta(days=activation_period)
    cop = datetime.timedelta(days=churn_observation_period)

    # 전체 범위 데이터에서 이탈 라벨 계산 -----------------------------------------------------------
    df_full = df_copy[df_copy['date'] < df_copy['first_login_date'] + ap + cop].copy()
    df_full[churn_column] = ((df_full['date'] > df_full['first_login_date'] + ap) &
                             (df_full['date'] < df_full['first_login_date'] + ap + cop)).astype(int)
    churn_labels = df_full.groupby('name')[churn_column].max().reset_index()

    # ★ 피처는 activation period 내 데이터만 사용 (데이터 누수 방지) ---------------------------------
    df_feat = df_copy[df_copy['date'] <= df_copy['first_login_date'] + ap].copy()
    df_feat = df_feat.sort_values(['name', 'date'])

    # 연속된 승/패 횟수 계산
    df_feat['start_of_streak'] = df_feat.groupby('name')['win'].diff().ne(0)
    df_feat['streak_id'] = df_feat.groupby('name')['start_of_streak'].cumsum()
    df_feat['streak_counter'] = df_feat.groupby(['name', 'streak_id']).cumcount() + 1

    df_feat['winning_streak'] = df_feat['streak_counter'] * df_feat['win']
    df_feat['losing_streak'] = df_feat['streak_counter'] * (1 - df_feat['win'])

    df_feat['win_count'] = df_feat['win']
    df_feat['lose_count'] = 1 - df_feat['win']
    df_feat['game_count'] = 1
    df_feat['active_date'] = df_feat['date'].dt.date

    # 시간 기반 피처
    df_feat['last_game'] = df_feat.groupby('name')['date'].transform('max')
    df_feat['engagement_hours'] = (df_feat['last_game'] - df_feat['first_login_date']).dt.total_seconds() / 3600
    df_feat['hour'] = df_feat['date'].dt.hour
    df_feat['prev_date'] = df_feat.groupby('name')['date'].shift(1)
    df_feat['gap_min'] = (df_feat['date'] - df_feat['prev_date']).dt.total_seconds() / 60

    # 피처 집계 ------------------------------------------------------------------------------------
    result_df = df_feat.groupby('name').agg(
        score_mean=('score', 'mean'),
        score_std=('score', 'std'),
        points_mean=('points', 'mean'),
        degree_mean=('degree', 'mean'),
        win_rate=('win', 'mean'),
        win_count=('win_count', 'sum'),
        lose_count=('lose_count', 'sum'),
        winning_streak=('winning_streak', 'max'),
        losing_streak=('losing_streak', 'max'),
        game_count=('game_count', 'sum'),
        active_days=('active_date', 'nunique'),
        engagement_hours=('engagement_hours', 'first'),
        avg_gap_min=('gap_min', 'mean'),
        hour_std=('hour', 'std'),
    ).reset_index()

    result_df['score_std'] = result_df['score_std'].fillna(0)
    result_df['hour_std'] = result_df['hour_std'].fillna(0)
    result_df['avg_gap_min'] = result_df['avg_gap_min'].fillna(0)
    result_df['games_per_day'] = result_df['game_count'] / result_df['active_days'].clip(lower=1)

    # 이탈 라벨 합치기
    result_df = result_df.merge(churn_labels, on='name')
    result_df.drop('name', axis=1, inplace=True)

    return result_df


def data_split(df, t_col, test_size, random_state=123456, scale=True):
    X = df.drop(t_col, axis='columns')
    y = df[[t_col]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X, y, X_train, X_test, y_train, y_test

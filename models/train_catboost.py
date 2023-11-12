import pandas as pd
from catboost import CatBoostClassifier
pd.set_option('display.max_columns', None)

y_train1 = pd.read_csv('./data/y_train.csv')
y_train1['month'] = y_train1['month'].astype('datetime64[ns]')
y_train2 = pd.read_csv('./data/feb/y_test.csv')
y_train2['month'] = y_train2['month'].astype('datetime64[ns]')
y_train = pd.concat([y_train1, y_train2])

dislok_wagons1 = pd.read_parquet(
    './data/dislok_wagons.parquet', engine='pyarrow')
dislok_wagons2 = pd.read_parquet(
    './data/feb/dislok_wagons.parquet', engine='pyarrow')
dislok_wagons = pd.concat([dislok_wagons1, dislok_wagons2])
del dislok_wagons1, dislok_wagons2
dislok_wagons = dislok_wagons.sort_values(by=['wagnum', 'plan_date'])
dislok_wagons['ost_prob'] = dislok_wagons['ost_prob'].astype(float)
dislok_wagons = dislok_wagons[
    dislok_wagons['plan_date'] >= pd.to_datetime('2022-10-01')]

freight_info = pd.read_parquet('./data/freight_info.parquet', engine='pyarrow')

wag_params = pd.read_parquet('./data/wag_params.parquet', engine='pyarrow')
wag_params['tipvozd'] = wag_params['tipvozd'].astype(float)
wag_params = wag_params.drop(['model', 'date_iskl'], axis=1)

dislok_wagons = dislok_wagons.merge(
    right=freight_info,
    left_on='fr_id',
    right_on='fr_id',
    how='left'
    ).merge(
        right=freight_info,
        left_on='last_fr_id',
        right_on='fr_id',
        how='left',
        suffixes=(None, '_last_fr_id')
    ).merge(
        right=wag_params,
        left_on='wagnum',
        right_on='wagnum',
        how='left'
    )

# Оставляем вагоны, временные ряды по которым начинаются с 2022-10-01
# и заканчиваются 2023-03-01 с длительностю 152 наблюдения
target_wagons = dislok_wagons.groupby('wagnum').apply(
    lambda df_:
        (df_.iloc[0]['plan_date'] == pd.to_datetime('2022-10-01'))
        & (df_.iloc[-1]['plan_date'] == pd.to_datetime('2023-03-01'))
        & (df_.shape[0] == 152)
    )
target_wagons = target_wagons[target_wagons]

# Оставляем наблюдения по соответствующим вагонам
dislok_wagons = dislok_wagons[
    dislok_wagons['wagnum'].isin(target_wagons.index.tolist())]

# Джойним таргет и обучаем CatBoost
dislok_wagons['date_to_merge'] = dislok_wagons['plan_date'].map(
    lambda x:
        '2023-03-01' if x.month == 2
        else '2023-02-01' if x.month == 1
        else '2023-01-01' if x.month == 12
        else '2022-12-01' if x.month == 11
        else '2022-11-01' if x.month == 10
        else '2022-10-01' if x.month == 9
        else '2022-09-01' if x.month == 8
        else '1970-01-01'
)
dislok_wagons['date_to_merge'] = dislok_wagons['date_to_merge']\
    .astype('datetime64[ns]')
dislok_wagons['date_kap'] = dislok_wagons['date_kap'].astype('datetime64[ns]')
dislok_wagons['date_dep'] = dislok_wagons['date_dep'].astype('datetime64[ns]')
dislok_wagons['days_to_predict'] = (
    dislok_wagons['date_to_merge'] - dislok_wagons['plan_date']).dt.days
dislok_wagons = dislok_wagons.merge(
    right=y_train,
    left_on=['wagnum', 'date_to_merge'],
    right_on=['wagnum', 'month'],
    how='left')

dislok_wagons = dislok_wagons.assign(
    plane_date_day=dislok_wagons['plan_date'].dt.day,
    plane_date_month=dislok_wagons['plan_date'].dt.month,
    plane_date_year=dislok_wagons['plan_date'].dt.year,
    plane_date_dayofweek=dislok_wagons['plan_date'].dt.day_of_week,

    date_kap_day=dislok_wagons['date_kap'].dt.day,
    date_kap_month=dislok_wagons['date_kap'].dt.month,
    date_kap_year=dislok_wagons['date_kap'].dt.year,
    date_kap_dayofweek=dislok_wagons['date_kap'].dt.day_of_week,

    date_dep_day=dislok_wagons['date_dep'].dt.day,
    date_dep_month=dislok_wagons['date_dep'].dt.month,
    date_dep_year=dislok_wagons['date_dep'].dt.year,
    date_dep_dayofweek=dislok_wagons['date_dep'].dt.day_of_week,

    date_pl_rem_day=dislok_wagons['date_pl_rem'].dt.day,
    date_pl_rem_month=dislok_wagons['date_pl_rem'].dt.month,
    date_pl_rem_year=dislok_wagons['date_pl_rem'].dt.year,
    date_pl_rem_dayofweek=dislok_wagons['date_pl_rem'].dt.day_of_week,

    date_build_day=dislok_wagons['date_build'].dt.day,
    date_build_month=dislok_wagons['date_build'].dt.month,
    date_build_year=dislok_wagons['date_build'].dt.year,
    date_build_dayofweek=dislok_wagons['date_build'].dt.day_of_week,

    srok_sl_day=dislok_wagons['srok_sl'].dt.day,
    srok_sl_month=dislok_wagons['srok_sl'].dt.month,
    srok_sl_year=dislok_wagons['srok_sl'].dt.year,
    srok_sl_dayofweek=dislok_wagons['srok_sl'].dt.day_of_week
)

train = dislok_wagons[
    dislok_wagons['plan_date'] <= pd.to_datetime('2022-12-31')]
val = dislok_wagons[
    (dislok_wagons['plan_date'] >= pd.to_datetime('2023-01-01'))
    & (dislok_wagons['plan_date'] <= pd.to_datetime('2023-01-31'))]
x_to_forecast_mar = dislok_wagons[
    (dislok_wagons['plan_date'] >= pd.to_datetime('2023-02-01'))
    & (dislok_wagons['plan_date'] <= pd.to_datetime('2023-02-28'))]
del dislok_wagons

train['ost_prob'] = train['ost_prob'].fillna(train['ost_prob'].median())
val['ost_prob'] = val['ost_prob'].fillna(val['ost_prob'].median())
x_to_forecast_mar['ost_prob'] = x_to_forecast_mar['ost_prob']\
    .fillna(x_to_forecast_mar['ost_prob'].median())

train = train.select_dtypes(exclude='datetime64[ns]')
val = val.select_dtypes(exclude='datetime64[ns]')
x_to_forecast_mar = x_to_forecast_mar.select_dtypes(exclude='datetime64[ns]')

val = val.fillna(0)
train = train.fillna(0)
x_to_forecast_mar = x_to_forecast_mar.fillna(0)

x_to_forecast_mar = x_to_forecast_mar\
    .drop(['target_month', 'target_day'], axis=1)
x_train = train.drop(['target_month', 'target_day'], axis=1)
x_val = val.drop(['target_month', 'target_day'], axis=1)

y_train_month = train['target_month']
y_train_day = train['target_day']

y_val_month = val['target_month']
y_val_day = val['target_day']

cat_day = CatBoostClassifier(
    iterations=150000,
    learning_rate=0.01,
    depth=5,
    # use_best_model=True,
    verbose=100,
    task_type='GPU'
)
cat_day.fit(
    x_train, y_train_day,
    # eval_set=(x_val, y_val_day)
)
cat_day.save_model(
    '/home/rustem/projs/pgk/models/cat_day.cbm',
    format='cbm'
)

cat_month = CatBoostClassifier(
    iterations=180000,
    learning_rate=0.01,
    depth=5,
    # use_best_model=True,
    verbose=100,
    task_type='GPU'
)
cat_month.fit(
    x_train, y_train_month,
    # eval_set=(x_val, y_val_month)
)
cat_month.save_model(
    '/home/rustem/projs/pgk/models/cat_month.cbm',
    format='cbm'
)

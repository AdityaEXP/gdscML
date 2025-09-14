import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore", FutureWarning)
import os

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import optuna

TRAIN_PATH = r"data/MiNDAT.csv"
TEST_PATH  = r"data/MiNDAT_UNK.csv"

print("Loading data...")
df = pd.read_csv(TRAIN_PATH)
columns_new = [f"F{i}" for i in range(len(df.columns))]
df.columns = columns_new

ID_COL = 'F0'
TARGET_COL = 'F47'
CATEGORICAL_COLS = ['F14', 'F22', 'F24']
NUMERICAL_COLS = [c for c in df.columns if c not in [ID_COL, TARGET_COL] + CATEGORICAL_COLS]

print("Cleaning 'ERROR' values...")
df['F24'] = df['F24'].fillna('ERROR')

print("Correcting dtypes...")
for col in NUMERICAL_COLS + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=[TARGET_COL], inplace=True)

print("Engineering new features...")
df['is_cyst_type'] = df['F14'].str.contains('cyst|corro', case=False, na=False).astype(int)
df['vortex1_x_cyst'] = df['F38'] * df['is_cyst_type']
df['ring1_x_cyst']   = df['F10'] * df['is_cyst_type']

df['vortex_ratio'] = df['F38'] / (df['F41'] + 1e-6)
df['ring_ratio'] = df['F10'] / (df['F13'] + 1e-6)



df['F14_target_mean'] = df.groupby('F14')[TARGET_COL].transform('mean')
df['F22_target_mean'] = df.groupby('F22')[TARGET_COL].transform('mean')
df['F14_target_smooth'] = 0.7 * df['F14_target_mean'] + 0.3 * df[TARGET_COL].mean()


f24_map = {'tonga': 0, 'ERROR': 4, 'karbon': 1, 'silikon': 2, 'kristal': 3}
df['F24_ordinal'] = df['F24'].map(f24_map)

CATEGORICAL_COLS.remove('F24')
NUMERICAL_COLS += ['is_cyst_type', 'vortex1_x_cyst', 'ring1_x_cyst', 'F24_ordinal']
print("Removing highly correlated features...")
corr_matrix = df[NUMERICAL_COLS].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.85)]
df.drop(columns=to_drop, inplace=True)
NUMERICAL_COLS = [c for c in NUMERICAL_COLS if c not in to_drop]

for col in CATEGORICAL_COLS:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Data prep complete.")

final_cat_cols = CATEGORICAL_COLS
df_processed = pd.get_dummies(df, columns=final_cat_cols, drop_first=True)

for col in df_processed.columns:
    if col not in [ID_COL, TARGET_COL]:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

X = df_processed.drop(columns=[ID_COL, TARGET_COL], errors='ignore')
y = df_processed[TARGET_COL]

assert all(X.dtypes != 'object'), "X still has object columns!"

print(f"Final X shape: {X.shape}, y shape: {y.shape}")

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_scores = []

    for tr_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        imputer, scaler = KNNImputer(5), StandardScaler()
        X_tr.loc[:, NUMERICAL_COLS] = imputer.fit_transform(X_tr[NUMERICAL_COLS])
        X_tr.loc[:, NUMERICAL_COLS] = scaler.fit_transform(X_tr[NUMERICAL_COLS])
        X_val.loc[:, NUMERICAL_COLS] = imputer.transform(X_val[NUMERICAL_COLS])
        X_val.loc[:, NUMERICAL_COLS] = scaler.transform(X_val[NUMERICAL_COLS])

        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        oof_scores.append(rmse)

    return np.mean(oof_scores)


best_params = {
    'n_estimators': 600,
    'max_depth': 1,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

print("Best params:", best_params)

imputer, scaler = KNNImputer(n_neighbors=5), StandardScaler()
X.loc[:, NUMERICAL_COLS] = imputer.fit_transform(X[NUMERICAL_COLS])
X.loc[:, NUMERICAL_COLS] = scaler.fit_transform(X[NUMERICAL_COLS])

champion_model = RandomForestRegressor(**best_params)
champion_model.fit(X, y)

train_preds = champion_model.predict(X)
rmse = np.sqrt(mean_squared_error(y, train_preds))
r2 = r2_score(y, train_preds)
print(f"Train RMSE: {rmse:.4f}, R2: {r2:.4f}")

try:
    test_df = pd.read_csv(TEST_PATH)
    test_generic_names = [c for c in columns_new if c != TARGET_COL]
    test_df.columns = test_generic_names
    test_ids = test_df.index

    test_df['F24'] = test_df['F24'].fillna('ERROR')
    test_df['F24_ordinal'] = test_df['F24'].map(f24_map)
    test_df['is_cyst_type'] = test_df['F14'].str.contains('cyst|corro', case=False, na=False).astype(int)
    if 'F38' in test_df.columns: test_df['vortex1_x_cyst'] = test_df['F38'] * test_df['is_cyst_type']
    if 'F10' in test_df.columns: test_df['ring1_x_cyst'] = test_df['F10'] * test_df['is_cyst_type']

    test_df['vortex_ratio'] = test_df['F38'] / (test_df['F41'] + 1e-6)
    test_df['ring_ratio'] = test_df['F10'] / (test_df['F13'] + 1e-6)



    for col in final_cat_cols:
        test_df[col] = test_df[col].fillna(df[col].mode()[0])
    test_df.drop(columns=[c for c in to_drop if c in test_df.columns], inplace=True)

    test_df_processed = pd.get_dummies(test_df, columns=final_cat_cols, drop_first=True)

    for col in test_df_processed.columns:
        if col not in [ID_COL, TARGET_COL]:
            test_df_processed[col] = pd.to_numeric(test_df_processed[col], errors='coerce')

    final_X_test = test_df_processed.reindex(columns=X.columns, fill_value=0)

    final_X_test.loc[:, NUMERICAL_COLS] = imputer.transform(final_X_test[NUMERICAL_COLS])
    final_X_test.loc[:, NUMERICAL_COLS] = scaler.transform(final_X_test[NUMERICAL_COLS])

    preds = champion_model.predict(final_X_test)
    submission = pd.DataFrame({'LOCAL_IDENTIFIER': test_ids, 'CORRUCYSTIC_DENSITY': preds})
    submission.to_csv('submission.csv', index=False)

    print("Submission saved as submission.csv")
    print(submission.head())

except FileNotFoundError:
    print("Check File")

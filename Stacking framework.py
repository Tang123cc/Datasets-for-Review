import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
import seaborn as sns
import plotly.express as px
from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor 
file_path = r'your file_path'
try:
    data = pd.read_excel(file_path)
    X = data.iloc[:, :-1]  
    y = data.iloc[:, -1]   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
except Exception as e:
    print(f"数据加载或预处理错误: {str(e)}")
    raise
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=xx, random_state=xx)
all_models = [
    ('lr', LinearRegression(xx)),
    ('ridge', Ridge(xx),
    ('mlp', MLPRegressor(hidden_layer_sizes=xx, max_iter=xx,  xx
                       ),
    ('svr', SVR(xx)),
    ('rf', RandomForestRegressor(xx)),
    ('gb', GradientBoostingRegressor(xx)),
    ('et', ExtraTreesRegressor(xx)),
    ('dt', DecisionTreeRegressor(xx))
]
all_model_performance = []
for name, model in all_models:
    try:
        mse_scores = cross_val_score(model, X_train, y_train, cv=10, 
                                   scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')
        mae_scores = cross_val_score(model, X_train, y_train, cv=10,
                                   scoring='neg_mean_absolute_error')
        mse = -np.mean(mse_scores)
        r2 = np.mean(r2_scores)
        mae = -np.mean(mae_scores)
        
        all_model_performance.append({
            'Model': name,
            'MSE': mse,
            'R2': r2,
            'MAE': mae,
        })
        print(f"{name}: MSE={mse:.4f}±{np.std(mse_scores):.4f}, "
              f"R2={r2:.4f}±{np.std(r2_scores):.4f}, MAE={mae:.4f}")
all_model_performance = [m for m in all_model_performance if m['MSE'] < np.median([x['MSE'] for x in all_model_performance]) * 2]
all_model_performance.sort(key=lambda x: x['MSE'])
for i, m in enumerate(all_model_performance):
    print(f"{i+1}. {m['Model']}: MSE={m['MSE']:.4f}")
best_combinations = []
for n in range(1, 9):
    top_n = [m['Model'] for m in all_model_performance[:n]]
    models = [(name, model) for name, model in all_models if name in top_n]
    trained_models = []
    for name, model in models:
        model.fit(X_train, y_train)
        trained_models.append(model)
    from sklearn.metrics import pairwise_distances
    preds = np.array([model.predict(X_test) for model in trained_models]).T
    diversity = np.mean(pairwise_distances(preds, metric='correlation'))
    stacking_reg = StackingRegressor(
        estimators=[(name, model) for name, model in zip([m[0] for m in models], trained_models)],
        final_estimator=LinearRegression(),
        cv=5,
        passthrough=True  
    )
    scores = cross_val_score(stacking_reg, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(scores)
    r2 = cross_val_score(stacking_reg, X_train, y_train, cv=10, scoring='r2').mean()
    best_combinations.append({
        'Number': n,
        'Models': top_n,
        'MSE': mse,
        'R2': r2,
        'Diversity': diversity
    })
    print(f"\n{n}combination: {top_n}")
    print(f"MSE={mse:.4f}, R2={r2:.4f}")
best_combo = min(best_combinations, key=lambda x: x['MSE'])
plt.subplot(1, 2, 2)
plt.bar([str(c['Number']) for c in best_combinations],
       [c['R2'] for c in best_combinations])
plt.xlabel('How many Mls')
plt.ylabel('R2 Score')
plt.title('different R2')
plt.tight_layout()
output_dir = r'xx'
plt.savefig(os.path.join(output_dir, 'ensemble_size_barplot.png'))
plt.close()
results_df = pd.DataFrame(best_combinations)
results_df = results_df[['Number', 'MSE', 'R2', 'Models']]
results_df.columns = ['number', 'MSE', 'R2', 'combination']
results_df.to_excel(os.path.join(output_dir, 'ensemble_results.xlsx'), index=False)
models = [(name, model) for name, model in all_models if name in best_combo['Models']]
cv_scores = []
for name, model in models:
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    cv_scores.append(np.mean(scores))
model_weights = []
for i, (name, model) in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    weight = (1/(mse+0.1)) * (r2+1) 
    model_weights.append(weight)
weights = [w/sum(model_weights) for w in model_weights]
print("\nweight:")
for i, (name, _) in enumerate(models):
    print(f"{name}: {weights[i]:.4f}")
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
stacking_reg = StackingRegressor(
    estimators=models,
    final_estimator=LinearRegression(),
    cv=10
)
stacking_reg.fit(X_train, y_train)
y_pred = stacking_reg.predict(X_test)
model_performance = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_performance.append({
        'Model': name,
        'MSE': mse,
        'R2': r2
    })
stacking_reg.fit(X_train, y_train)
y_pred = stacking_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
model_performance.append({
    'Model': 'Stacking',
    'MSE': mse,
    'R2': r2
})
performance_df = pd.DataFrame(model_performance)
print("\nModel Performance Comparison:")
print(performance_df.to_string(index=False))
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred, 
    'Error': y_test - y_pred
})
results.to_excel(os.path.join(output_dir, 'prediction_results.xlsx'), index=False)
print(f"Saved prediction_results.xlsx to {output_dir}")
plt.figure(figsize=(8, 6))
errors = y_test - y_pred
sns.histplot(errors, bins=30, kde=True)
plt.title('Error Distribution')
plt.xlabel('Error')
plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
print(f"Saved error_distribution.png to {output_dir}")
plt.close()
plt.figure(figsize=(8, 6))
plt.bar([name for name, _ in models], weights)
plt.title('Meta-Model Weights in Stacking')
plt.ylabel('Weight')
plt.savefig(os.path.join(output_dir, 'meta_model_weights.png'))
print(f"Saved meta_model_weights.png to {output_dir}")
plt.close()
print("\nEssential outputs saved:")
print("- prediction_results.xlsx (results)")
print("- error_distribution.png (error)")
print("- meta_model_weights.png (weight)")

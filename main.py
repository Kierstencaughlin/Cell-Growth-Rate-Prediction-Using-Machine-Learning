import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split, KFold, RepeatedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))

features = df.columns[1:]
target = df.columns[0]

x = df.loc[:, features].to_numpy()
y = df.loc[:, target].to_numpy()

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)

cv = KFold(n_splits=5,shuffle=True,random_state=0)

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)
#=================================================================================================================
#Understanding the Data
#=================================================================================================================

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), dpi=200, layout="constrained")

for i, ax in enumerate(axes.flat):
    ax.scatter(x_train[:, i], y_train, color="tab:blue", alpha=0.3)
    ax.set_xlabel(features[i])
    ax.set_ylabel("Specific Growth Rate [d$^{-1}$]")

fig.savefig(os.path.join("figures", "figure_data_exploration.png"))
plt.close(fig)

#=================================================================================================================
#Compare Linear and Polynomial Regression
#=================================================================================================================
comparison_rows=[]

for degree in range (1,4):
    polynomial_model=make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())

    cv_results=cross_validate(polynomial_model, x_train, y_train, cv=cv, scoring=("neg_root_mean_squared_error","r2"), return_train_score=True)

    ms_train_rmse = -cv_results["train_neg_root_mean_squared_error"]
    ms_train_r2 = cv_results["train_r2"]
    ms_test_rmse = -cv_results["test_neg_root_mean_squared_error"]
    ms_test_r2 = cv_results["test_r2"]

    comparison_rows.append({
        "Model": f"Polynomial degree {degree}",
        "Train RMSE [d^-1]": np.mean(ms_train_rmse),
        "Valid RMSE [d^-1]": np.mean(ms_test_rmse),
        "Train R2": np.mean(ms_train_r2),
        "Valid R2": np.mean(ms_test_r2)})

df_comparison = pd.DataFrame(comparison_rows)
df_comparison.to_csv(os.path.join("tables", "model_comparison.csv"), index=False, float_format="%.3f")

print("\nPolynomial Model Comparison:")
print(df_comparison.round(3).to_string(index=False))

# =================================================================================================================
# kNN
# =================================================================================================================
model_knn=make_pipeline(StandardScaler(),KNeighborsRegressor())

param_grid = dict(kneighborsregressor__n_neighbors=[i for i in range(1, 81)],kneighborsregressor__weights=["uniform", "distance"])

model_gscv=GridSearchCV(model_knn,param_grid=param_grid, cv=cv,scoring="neg_root_mean_squared_error", return_train_score=True)

model_gscv.fit(x_train,y_train)

df_knn=pd.DataFrame(model_gscv.cv_results_)
df_knn=df_knn.sort_values(by=["rank_test_score"],ascending=True)

columns_view=["rank_test_score","param_kneighborsregressor__n_neighbors","param_kneighborsregressor__weights","mean_test_score", "std_test_score","mean_train_score", "std_train_score"]


print(df_knn.loc[:, columns_view].head(5).to_string(index=False))
df_knn.loc[:, columns_view].to_csv(os.path.join("tables", "knn_grid_search.csv"), index = False, float_format="%.3f")

print("\nkNN Best Parameters:")
print(model_gscv.best_params_)
print(f"Validation RMSE: {-model_gscv.best_score_:.3f} d^-1")

# ============================================================================
# Polynomial Regression Coefficients
# ============================================================================

degree=2
model_polynomial=make_pipeline(StandardScaler(),PolynomialFeatures(degree,include_bias=False),LinearRegression())

cv_repeated=RepeatedKFold(n_splits=5,n_repeats=5,random_state=0)
repeated_results=[]

for name, candidate in [("Polynomial degree 2", model_polynomial),("kNN", model_gscv.best_estimator_)]:

    cv_results = cross_validate(candidate, x_train, y_train, cv=cv_repeated,

    scoring = "neg_root_mean_squared_error", return_train_score = True)

    ms_train_rmse = -cv_results["train_score"]
    ms_test_rmse = -cv_results["test_score"]

    repeated_results.append({
        "Model": name,
        "Train RMSE": np.mean(ms_train_rmse),
        "Valid RMSE": np.mean(ms_test_rmse),
        "Valid RMSE SD": np.std(ms_test_rmse)
    })

df_repeated = pd.DataFrame(repeated_results)
df_repeated.to_csv(os.path.join("tables", "repeated_cross_validation.csv"), index=False, float_format="%.3f")

model = model_polynomial

cv_results = cross_validate(model, x_train, y_train, cv=cv,
scoring = ("neg_root_mean_squared_error", "r2"),
return_train_score = True)

ms_train_rmse = -cv_results["train_neg_root_mean_squared_error"]
ms_train_r2 = cv_results["train_r2"]
ms_test_rmse = -cv_results["test_neg_root_mean_squared_error"]
ms_test_r2 = cv_results["test_r2"]

print(f"\nPolynomial model with {degree = }:")
print(f"  Train | RMSE | {np.mean(ms_train_rmse):6.3f} +/- {np.std(ms_train_rmse):6.3f} d^-1")
print(f"  Valid | RMSE | {np.mean(ms_test_rmse):6.3f} +/- {np.std(ms_test_rmse):6.3f} d^-1")
print(f"  Train | R^2  | {np.mean(ms_train_r2):6.3f} +/- {np.std(ms_train_r2):6.3f}")
print(f"  Valid | R^2  | {np.mean(ms_test_r2):6.3f} +/- {np.std(ms_test_r2):6.3f}")

# Fit model since cross_validate() trains copies of the model
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

m_train_rmse = root_mean_squared_error(y_train, y_train_pred)
m_test_rmse = root_mean_squared_error(y_test, y_test_pred)
m_train_r2 = r2_score(y_train, y_train_pred)
m_test_r2 = r2_score(y_test, y_test_pred)

print("\nFinal Test Results:")
print(f"  Test | RMSE | {m_test_rmse:.3f} d^-1")
print(f"  Test | R^2  | {m_test_r2:.3f}")

powers = model["polynomialfeatures"].powers_
coefs = model["linearregression"].coef_
intercept = model["linearregression"].intercept_

terms = []

for i in range(powers.shape[0]):
    partial_terms = []
    for j in range(powers.shape[1]):
        feature_name = features[j].split(" [")[0]

        if powers[i, j] == 0:
            pass
        elif powers[i, j] == 1:
            partial_terms.append(feature_name)
        else:
            partial_terms.append(f"{feature_name}^{powers[i, j]}")

    terms.append(" * ".join(partial_terms))

# Save the polynomial equation's coefficients in a table
df_coefs = pd.DataFrame({"Polynomial Feature": terms, "Coefficient": coefs})
df_coefs.loc[len(df_coefs)] = ["Intercept", intercept]
df_coefs.to_csv(os.path.join("tables", "polynomial_coefficients.csv"), index=False, float_format="%.3f")
terms=[]

print("\nPolynomial Regression Equation:")
print(f"Intercept: {intercept:.4f}")

for feature, coef in zip(powers, coefs):
    print(f"{coef:+8.3f} * {feature}")


# ============================================================================
# Create a Table of Polynomial Coefficients in the Regression Equation
# ============================================================================

print("\nPolynomial Coefficients:")
print(df_coefs.to_string(index=False))

os.makedirs("tables", exist_ok=True)

df_coefs.to_csv(os.path.join("tables", "polynomial_coefficients.csv"), index=False, float_format="%.3f")

# ============================================================================
# Effect of Operating Conditions on Cell Growth
# ============================================================================

# Calculate median values
x_median = np.median(x_train, axis=0)

# Create figure
fig, axes = plt.subplots(
    nrows=2, ncols=2,
    figsize=(12, 8),
    dpi=200,
    layout="constrained"
)

# Loop through each feature
for i, ax in enumerate(axes.flat):
    # Define feature range
    x_values = np.linspace(np.percentile(x_train[:, i], 5), np.percentile(x_train[:, i], 95),100)

    # Hold other features at their median
    x_plot = np.tile(x_median, (100, 1))
    x_plot[:, i] = x_values

    # Predict growth rate
    y_pred = model.predict(x_plot)

    # Plot experimental data
    ax.scatter(x_train[:, i], y_train,color="tab:blue",alpha=0.3,label="Experimental data")

    # Plot model predictions
    ax.plot(x_values, y_pred,color="tab:red",linewidth=2,label="Model prediction")

    ax.set_xlabel(features[i])
    ax.set_ylabel("Specific Growth Rate [d$^{-1}$]")
    ax.legend()

# Export figure
os.makedirs("figures", exist_ok=True)

fig.savefig(os.path.join("figures", "figure_feature_effects.png"))

plt.close(fig)

# ============================================================================
# Interaction Between Operating Conditions
# ============================================================================
#
# * Investigate interactions between temperature and pH,
#   and between glucose and lactate.
#
# * Hold the remaining features at their median values.
#
# ============================================================================
feature_pairs = [
    (0, 1, "Temperature and pH", "figure_temp_ph.png"),
    (2, 3, "Glucose and Lactate", "figure_glucose_lactate.png")]

for i, j, title, filename in feature_pairs:
    x1 = np.linspace(np.percentile(x_train[:, i], 5), np.percentile(x_train[:, i], 95), 100)
    x2 = np.linspace(np.percentile(x_train[:, j], 5), np.percentile(x_train[:, j], 95), 100)

    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_plot = np.tile(x_median, (x1_grid.size, 1))
    x_plot[:, i] = x1_grid.ravel()
    x_plot[:, j] = x2_grid.ravel()

    y_plot = model.predict(x_plot)
    growth_grid = y_plot.reshape(x1_grid.shape)

    idx_max = np.unravel_index(np.argmax(growth_grid), growth_grid.shape)
    idx_min = np.unravel_index(np.argmin(growth_grid), growth_grid.shape)

    print(f"\n{title}:")
    print(f"  Maximum: {growth_grid[idx_max]:.3f} d^-1 at ({x1_grid[idx_max]:.2f}, {x2_grid[idx_max]:.2f})")
    print(f"  Minimum: {growth_grid[idx_min]:.3f} d^-1 at ({x1_grid[idx_min]:.2f}, {x2_grid[idx_min]:.2f})")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), dpi=200, layout="constrained")
    contour = ax.contourf(x1_grid, x2_grid, growth_grid, levels=20, cmap="viridis")
    ax.scatter(x_train[:, i], x_train[:, j], color="white", edgecolors="black", s=12, alpha=0.4)
    ax.scatter(x1_grid[idx_max], x2_grid[idx_max], marker="*", s=180,
               color="red", edgecolors="black", label="Predicted maximum")

    ax.set_xlabel(features[i])
    ax.set_ylabel(features[j])
    ax.set_title(title)
    ax.legend()

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Predicted Growth Rate [d$^{-1}$]")

    fig.savefig(os.path.join("figures", filename))
    plt.close(fig)
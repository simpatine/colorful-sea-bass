import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/subsampling.csv", delimiter=";",header=0)

vals = data.groupby("subsample_ratio").mean()
errs = data.groupby("subsample_ratio").std()


# plt.semilogx(vals.index,vals.accuracy,"--",marker="",label="Accuracy")
# plt.semilogx(vals.index,vals.f1,":",marker="",label="F1")
plt.figure(figsize=(8,5))
plt.errorbar(vals.index,vals.accuracy,yerr=errs.accuracy,ls="--",marker="o",label="Accuracy")
plt.errorbar(vals.index,vals.f1,yerr=errs.f1,ls=":",marker="o",label="F1")
plt.xscale("log")
plt.title('Accuracy and F1 scores with random subsampling on all genes')
plt.xlabel("Subsampling ratio")
plt.ylabel("Accuracy/F1 score (%)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("subsample_plot.png")
plt.show()


exit(0)


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, alpha=errs.accuracy.to_numpy()**2, n_restarts_optimizer=9
)

X = np.linspace(np.log10(vals.index).min(),np.log10(vals.index).max()).reshape(-1, 1)
x_train = np.log10(vals.index.to_numpy())
y_train = vals.accuracy.to_numpy()

gaussian_process.fit(x_train.reshape(-1, 1),y_train)

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
# plt.errorbar(vals.index,vals.f1,yerr=errs.f1

plt.plot(X, mean_prediction, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.errorbar(
    x_train,
    y_train,
    yerr=errs.accuracy,
    linestyle="--",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations",
)
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.title("Gaussian process regression on a noisy dataset")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# file, title, outfile
groups = [
        ([("data/out_subsample.txt",None,None)],'on all genes',"subsample_plot.png"),
        ([("data/out_unifsub.txt",None,None)],'uniformly on the chromosomes',"uniform_sample_low_ratio.png"),
        ([(f"data/annotated/out_{function}.txt",'{function}',{"function":function}) for function in ["Open_chromatin","Enhancer","Promoter"]],
         "on annotated genes",f"subsample_annotated.png"),
        ([(f"data/ntissue/out_{ntissue}.txt","[{l},{r})",{"l":ntissue,"r":ntissue+10}) for ntissue in [0,10,20]],
         f'on genes with ntissue in given ranges' ,f"subsample_ntissue.png"),
        ]


for group, suptitle, outfile in groups:
    fig,ax = plt.subplots(1,len(group),figsize=(6*len(group)**0.7,4), sharey=True, sharex=True, dpi=100)
    if len(group) == 1: ax = [ax]
    # fig.suptitle("Accuracy and F1 scores with random subsampling " + suptitle)
    for i,(datapath,title,vals) in enumerate(group):
        ax[i].axhline(54.24, color="b", ls=":", label="Baseline Accuracy")
        ax[i].axhline(200/((100/54.24) + (100/45.76)), color="r", ls=":", label="Baseline F1 score")
        ax[i].set_xscale("log")
        if title is not None:
            ax[i].set_title(title.format(**vals))
        ax[i].set_xlabel("Subsampling ratio")
        ax[i].set_ylabel("Accuracy/F1 score (%)")
        print(f"Working on {datapath}...")
        data = pd.read_csv(datapath, delimiter=";",header=0)

        vals = data.groupby("subsample_ratio").mean()
        errs = data.groupby("subsample_ratio").std()

        # plt.semilogx(vals.index,vals.accuracy,"--",marker="",label="Accuracy")
        # plt.semilogx(vals.index,vals.f1,":",marker="",label="F1")
        ax[i].errorbar(vals.index,vals.accuracy,yerr=errs.accuracy,color="b",ls="--",marker="o",label="Accuracy", markersize=8, capsize=5)
        ax[i].errorbar(vals.index,vals.f1,yerr=errs.f1,color="r",ls="--",marker="o",label="F1", markersize=8, capsize=5)
        ax[i].grid()
        ax[i].legend()

    fig.tight_layout()
    fig.savefig(outfile)
    # plt.show()
    print("Done.")

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

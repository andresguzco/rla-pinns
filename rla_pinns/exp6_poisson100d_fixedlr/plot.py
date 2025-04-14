"""Plot the best runs from each tuned optimizer"""

from argparse import ArgumentParser
from itertools import product
from os import makedirs, path

from matplotlib import pyplot as plt
from palettable.colorbrewer import sequential
from tueplots import bundles

from rla_pinns.train import set_up_layers
from rla_pinns.wandb_utils import (
    WandbRunFormatter,
    WandbSweepFormatter,
    load_best_run,
    remove_unused_runs,
    show_sweeps,
)

entity = "rla-pinns"  # team name on wandb
project = "exp6_poisson100d_fixedlr"  # name from the 'Projects' tab on wandb

# information for title
equation = "poisson"
architecture = "mlp-tanh-768-768-512-512"
dim_Omega = 100
num_params = sum(
    p.numel()
    for layer in set_up_layers(architecture, equation, dim_Omega)
    for p in layer.parameters()
)

# Useful to map sweep ids to human-readable names
print_sweeps = False
if print_sweeps:
    show_sweeps(entity, project)
    raise Exception("Printed sweeps. Exiting...")


sweep_ids = {  # ids from the wandb agent
    # "agtgmknd": "SGD",
    # "p6bgdypg": "Adam",
    # "fdohey43": "ENGD",
    # "d5ujt0u0": "Hessian-free",
    "uw7ye8u0": "ENGD (Woodbury)",
    # "dvtd4rth": "ENGD (Nystrom)",
    "gh8vlzxy": "SPRING",
    # "qf0s6jg3": "SPRING (Nystrom)",
}

# color options: https://jiffyclub.github.io/palettable/colorbrewer/
colors = {
    "SGD": sequential.Reds_4.mpl_colors[-2],
    "Adam": sequential.Reds_4.mpl_colors[-1],
    "ENGD": sequential.Blues_5.mpl_colors[-3],
    "ENGD (Woodbury)": sequential.Blues_5.mpl_colors[-2],
    "ENGD (Nystrom)": sequential.Blues_5.mpl_colors[-1],
    "SPRING": sequential.Greens_4.mpl_colors[-2],
    "SPRING (Nystrom)": sequential.Greens_4.mpl_colors[-1],
    "Hessian-free": "black",
}

linestyles = {
    "SGD": "-",
    "Adam": "-",
    "ENGD": "-",
    "ENGD (Woodbury)": "-",
    "ENGD (Nystrom)": "-",
    "SPRING": "-",
    "SPRING (Nystrom)": "-",
    "Hessian-free": "-",
}

HEREDIR = path.dirname(path.abspath(__file__))
DATADIR = path.join(HEREDIR, "best_runs")
makedirs(DATADIR, exist_ok=True)

# enable this to remove all saved files from sweeps that are not plotted
clean_up = True
if clean_up:
    remove_unused_runs(keep=list(sweep_ids.keys()), best_run_dir=DATADIR)

if __name__ == "__main__":
    parser = ArgumentParser(description="Plot the best runs from each tuned optimizer.")
    parser.add_argument(
        "--local_files",
        action="store_false",
        dest="update",
        help="Use local files if possible.",
        default=True,
    )
    parser.add_argument(
        "--disable_tex",
        action="store_true",
        default=False,
        help="Disable TeX rendering in matplotlib.",
    )
    args = parser.parse_args()

    y_to_ylabel = {"loss": "Loss", "l2_error": "$L_2$ error"}
    x_to_xlabel = {"step": "Iteration", "time": "Time (s)"}

    for (x, xlabel), (y, ylabel) in product(x_to_xlabel.items(), y_to_ylabel.items()):
        with plt.rc_context(
            bundles.neurips2023(rel_width=1.0, usetex=not args.disable_tex)
        ):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(xlabel)
            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.set_title(f"{dim_Omega}d {equation.capitalize()} ($D={num_params}$)")
            ax.grid(True, alpha=0.5)

            for sweep_id, label in sweep_ids.items():
                df_history, _ = load_best_run(
                    entity,
                    project,
                    sweep_id,
                    save=True,
                    update=args.update,
                    savedir=DATADIR,
                )
                x_data = {
                    "step": df_history["step"] + 1,
                    "time": df_history["time"] - min(df_history["time"]),
                }[x]
                ax.plot(
                    x_data,
                    df_history[y],
                    label=None if "*" in label else label,
                    color=colors[label],
                    linestyle=linestyles[label],
                )

            ax.legend()
            plt.savefig(path.join(HEREDIR, f"{y}_over_{x}.pdf"), bbox_inches="tight")

    # export sweep and run descriptions to LaTeX
    # TEXDIR = path.join(HEREDIR, "tex")
    # makedirs(TEXDIR, exist_ok=True)

    # if args.update:  # only if online access is possible
    #     for sweep_id in sweep_ids:
    #         _, meta = load_best_run(entity, project, sweep_id, savedir=DATADIR)
    #         sweep_args = meta.to_dict()["config"][0]
    #         WandbRunFormatter.to_tex(TEXDIR, sweep_args)

    #     for sweep in show_sweeps(entity, project):
    #         WandbSweepFormatter.to_tex(TEXDIR, sweep.config)
    # else:
    #     print("Skipping LaTeX export of sweeps and best runs.")

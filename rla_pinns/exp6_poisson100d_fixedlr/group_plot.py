"""Create a pretty plot that groups together the results for 1d heat."""

from argparse import ArgumentParser
from os import path

from matplotlib import pyplot as plt
from tueplots import bundles

from rla_pinns.exp8_poisson5d_fixedlr import plot as SMALL
from rla_pinns.exp8_poisson5d_fixedlr.plot import colors, linestyles
from rla_pinns.exp15_heat4d_fixed import plot as LEFT
from rla_pinns.exp10_log_fokker_planck_fixedlr import plot as RIGHT
from rla_pinns.exp6_poisson100d_fixedlr import plot as BIG
from rla_pinns.wandb_utils import load_best_run

# kept for titles of the Poisson row
DIMS = [5, 100, 4, 9]
PARAMS = [10065, 1325057, 116865, 118145]

EXTRA = [
    {
        "entity": "andresguzco",  # project name
        "exp": "rla-pinns",
        "id": "tacjf0pi",
    },
    {
        "entity": "rla-pinns",
        "exp": "exp4_poisson100d",
        "id": "elqquiw6",
    },
    {
        "entity": "rla-pinns",
        "exp": "exp14_heat4d",
        "id": "zbdggni9",
    },
    {
        "entity": "rla-pinns",
        "exp": "exp5_log_fokker_planck",
        "id": "zbscn1ou",
    },
]
KFAC_VALUES = [0.00017338208691050153, 0.004302723223431071, 2.1103576920923328e-05, 0.025968427732340805]

# Time caps (seconds)
MAX_T_TOP_AND_BR = 6000  # top row and bottom-right
MAX_T_BL = 3000          # bottom-left

# Extra right-side frame padding (seconds) so lines end before the axis max
RIGHT_PADDING = 1000


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Summarize the experiments on heat 1d in one figure."
    )
    parser.add_argument(
        "--disable_tex",
        action="store_true",
        default=False,
        help="Disable TeX rendering in matplotlib.",
    )
    args = parser.parse_args()

    # Experiments laid out left-to-right, top-to-bottom
    COLUMNS = [SMALL, BIG, LEFT, RIGHT]
    TITLES_TEX = [
        rf"{DIMS[0]}d Poisson ($D={PARAMS[0]}$)",
        rf"{DIMS[1]}d Poisson ($D={PARAMS[1]}$)",
        rf"{DIMS[2]}d Heat ($D={PARAMS[2]}$)",
        rf"{DIMS[3]}+1d log-Fokker–Planck ($D={PARAMS[3]}$)",
    ]
    TITLES_PLAIN = [
        f"{DIMS[0]}d Poisson (D={PARAMS[0]})",
        f"{DIMS[1]}d Poisson (D={PARAMS[1]})",
        f"{DIMS[2]}d Heat (D={PARAMS[2]})",
        f"{DIMS[3]}+1d log-Fokker–Planck (D={PARAMS[3]})",
    ]
    IGNORE = {"ENGD (diagonal)"}

    # Axes labels
    y_key, y_label = "l2_error", r"$L_2$ error"
    x_key, x_label = "time", "Time [s]"

    # Build 2x2 grid
    with plt.rc_context(
        bundles.neurips2023(
            rel_width=1.0,
            nrows=2,
            ncols=2,
            usetex=not args.disable_tex,
        ),
    ):
        # enable siunitx grouping for D titles
        plt.rcParams["text.latex.preamble"] += (
            r"\usepackage[group-separator={,}, group-minimum-digits={3}]{siunitx}"
        )

        fig, axs = plt.subplots(2, 2)
        axs = axs.reshape(2, 2)

        # Iterate through all four experiments; map to 2x2 positions
        for idx, exp in enumerate(COLUMNS):
            r, c = divmod(idx, 2)
            ax = axs[r, c]

            # Decide per-subplot cap:
            # - top row (r==0) and bottom-right (r==1,c==1): 6000 s
            # - bottom-left (r==1,c==0): 3000 s
            # - otherwise: no cap
            if r == 0 or (r == 1 and c == 1):
                max_t = MAX_T_TOP_AND_BR
            elif r == 1 and c == 0:
                max_t = MAX_T_BL
            else:
                max_t = None

            # Axis formatting
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.5)
            ax.set_xlabel(x_label)
            if c == 0:
                ax.set_ylabel(y_label)

            # Titles (plain vs TeX)
            title = (TITLES_PLAIN if args.disable_tex else TITLES_TEX)[idx]
            ax.set_title(title)

            # Plot each optimizer run
            for sweep_id, name in exp.sweep_ids.items():
                if name in IGNORE:
                    continue
                df_history, _ = load_best_run(
                    exp.entity,
                    exp.project,
                    sweep_id,
                    save=False,
                    update=False,
                    savedir=exp.DATADIR,
                )
                x_data = df_history["time"] - df_history["time"].min()

                # Stop lines at the cap (if any)
                if max_t is not None:
                    mask = x_data <= max_t
                    x_plot = x_data[mask]
                    y_plot = df_history[y_key][mask]
                else:
                    x_plot = x_data
                    y_plot = df_history[y_key]

                label = name if (idx == 0 and "*" not in name) else None
                ax.plot(
                    x_plot,
                    y_plot,
                    label=label,
                    color=colors[name],
                    linestyle=linestyles[name],
                )

            # Overlay the corresponding line-search curve for this subplot
            line_search, _ = load_best_run(
                EXTRA[idx]["entity"],
                EXTRA[idx]["exp"],
                EXTRA[idx]["id"],
                save=False,
                update=False,
                savedir=exp.DATADIR,
            )
            x_ls = line_search["time"] - line_search["time"].min()

            # Stop at the cap (if any)
            if max_t is not None:
                mask_ls = x_ls <= max_t
                x_ls_plot = x_ls[mask_ls]
                y_ls_plot = line_search[y_key][mask_ls]
            else:
                x_ls_plot = x_ls
                y_ls_plot = line_search[y_key]

            ax.plot(
                x_ls_plot,
                y_ls_plot,
                label=("ENGD-W (Line Search)" if idx == 0 else None),
                color=colors["SGD"],
                linestyle=linestyles["SGD"],
            )
            ax.axhline(
                y=KFAC_VALUES[idx],
                linestyle="--",
                color="black",
                linewidth=1.2,
                label=("KFAC" if idx == 0 else None),
            )

            # X limits: ensure positive left bound for log-scale; add right-side buffer if capped
            if max_t is not None:
                ax.set_xlim(1, max_t + RIGHT_PADDING)  # e.g., 6k cap -> frame at 7k; 3k -> 4k
            else:
                ax.set_xlim(left=1)

        # Shared legend at bottom — only gather from first axes (labels added there)
        handles, labels = axs[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.07),
                ncol=len(labels),
                handlelength=1.35,
                columnspacing=0.9,
                frameon=True,
            )

        out_file = path.join(path.dirname(path.abspath(__file__)), "l2_grouped.pdf")
        plt.savefig(out_file, bbox_inches="tight")
        print(f"Saved combined figure to {out_file}")

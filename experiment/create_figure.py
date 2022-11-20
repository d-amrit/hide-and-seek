import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os


def create_chart_outline(x_label, y_label, set_yaxis_as_percent, title=None):
    """
    Setup the default figure with figure size, font size, axis labels, and title.
    """
    fig = plt.figure(1, (8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if set_yaxis_as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    return ax


def create_figure_1(r_dict, q_list, save_path):
    """
    r_dict = {Clause Density: list of u values}
    q_list = List of q values.
    """
    ax = create_chart_outline(
        x_label='Deceptive parameter q',
        y_label='% of clauses (u/rn)',
        set_yaxis_as_percent=True
    )

    for key, value in r_dict.items():
        y_values = [i / (key * 100) for i in value]
        plt.plot(q_list, y_values, label='r = {}'.format(key))

    ax.grid()
    ax.set_xticks(q_list)
    ax.legend(fontsize=16)

    plt.savefig(os.path.join(save_path, 'figure_1.png'), dpi=1200)
    plt.show()

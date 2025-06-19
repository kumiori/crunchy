import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.collections as collections


def plot_energies(data, signature=""):
    """
    Plots the elastic energy, fracture energy, and total energy over load steps.

    Parameters:
        data (dict): A dictionary containing 'elastic_energy', 'fracture_energy',
                     'total_energy', and optionally 'load_steps' or 'time_steps'.
    """
    # Extract energy components from the dataset
    elastic_energy = data.get("elastic_energy", [])
    fracture_energy = data.get("fracture_energy", [])
    total_energy = data.get("total_energy", [])
    loads = data.get("load", [])
    load_steps = data.get(
        "load_steps", range(len(elastic_energy))
    )  # Use index if load_steps are not provided
    linewidth = 5
    # Ensure we have data to plot
    if elastic_energy.empty or fracture_energy.empty or total_energy.empty:
        print("Error: Missing energy data for plotting.")
        return

    # Plot the energies
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the energies
    ax.plot(
        loads,
        elastic_energy,
        label="Elastic",
        linewidth=linewidth,
        marker="o",
        markersize=10,
    )
    ax.plot(
        loads,
        fracture_energy,
        label="Fracture",
        linewidth=linewidth,
        marker="o",
        markersize=10,
    )
    ax.plot(
        loads,
        total_energy,
        label="Total",
        color="k",
        linestyle="-",
        marker="o",
        markersize=10,
        linewidth=linewidth,
    )

    # Add plot details
    ax.set_title("Evolution - Energy", fontsize=20)
    ax.set_xlabel("Load", fontsize=18)
    ax.set_ylabel("Energy", fontsize=18)
    ax.legend(fontsize=16)

    ax.text(
        1.05,
        0.5,  # Position outside the right axis (normalized coordinates)
        "signature: " + signature[0:6],
        fontsize=18,
        color="gray",
        alpha=0.7,
        ha="center",
        va="center",
        rotation=90,  # Rotate the text vertically
        transform=ax.transAxes,  # Use axis transformation for positioning
    )

    fig.tight_layout()

    return fig, ax


def plot_spectrum(data, signature=""):
    """
    Plot the spectrum (eigenvalues) of eigs_ball and eigs_cone for each load step.

    Parameters:
        data (dict): Dictionary containing load steps and eigenvalue data.
        signature (str): Signature of the simulation for watermarking.
    Returns:
        fig, ax: Matplotlib figure and axes objects for further customization.
    """
    load_steps = data.get("load", [])
    eigs_ball = data.get("eigs_ball", [])
    eigs_cone = data.get("eigs_cone", [])

    # # Ensure we have data for plotting
    # if load_steps.empty:
    #     print("No data to plot.")

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot eigs_ball for each load step
    for step_idx, load_step in enumerate(load_steps):
        # if not eigs_ball.empty:  # Check if eigs_ball exists for the step
        if len(eigs_ball[step_idx]) > 0:  # Check if eigs_ball exists for the step
            _eigs_ball_step = eigs_ball[step_idx]
            ax.scatter(
                [load_step] * len(_eigs_ball_step),  # Load step as x-coordinates
                _eigs_ball_step,  # Eigenvalues as y-coordinates
                color="C0",
                alpha=0.7,
                s=100,
                label="Eigs Ball" if step_idx == 0 else None,  # Label only once
            )
        # Plot eigs_cone, if available
        if eigs_cone[step_idx]:
            _eigs_cone_step = eigs_cone[step_idx]
            ax.scatter(
                load_step,
                _eigs_cone_step,
                color="C3",
                alpha=0.7,
                s=100,
                label="Eigs Cone" if step_idx == 0 else None,
            )

    # Add horizontal reference line
    ax.axhline(y=0, color="black", linestyle="-", linewidth=3)

    # Customize plot appearance
    ax.set_title("Evolution - Spectrum", fontsize=20)
    ax.set_xlabel("Load", fontsize=18)
    ax.set_ylabel("Eigenvalue", fontsize=18)
    ax.legend(fontsize=16)
    # ax.grid(True, linestyle="--", alpha=0.7)

    # Add watermark with simulation signature
    ax.text(
        1.05,
        0.5,  # Position outside the right axis (normalized coordinates)
        "signature: " + signature[0:6],
        fontsize=18,
        color="gray",
        alpha=0.7,
        ha="center",
        va="center",
        rotation=90,  # Rotate vertically
        transform=ax.transAxes,  # Use axis transformation for positioning
    )
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis="x", which="minor", length=4, color="gray", labelsize=10)

    fig.tight_layout()

    # Return the figure and axes for further customization
    return fig, ax


def superimpose_figures(fig1, fig2, alpha=0.5):
    """
    Superimpose the lines/plots from fig2 onto fig1, ensuring that the lines
    of fig1 have an alpha transparency.

    Parameters:
        fig1: matplotlib.figure.Figure
            The first figure to modify.
        fig2: matplotlib.figure.Figure
            The second figure whose plots will be overlaid on fig1.
        alpha: float
            The transparency level for the lines/plots of fig1.
    """
    # Extract axes from both figures
    ax1 = fig1.axes[0]  # Assuming fig1 has one Axes object
    ax2 = fig2.axes[0]  # Assuming fig2 has one Axes object

    # Adjust alpha of the first figure's lines
    for line in ax1.get_lines():
        line.set_alpha(alpha)

    # Overlay lines from the second figure
    for line in ax2.get_lines():
        new_line = ax1.plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
        )
        # Maintain the alpha of the second figure's lines if needed
        new_line[0].set_alpha(1.0)

    # Overlay scatter points from the second figure
    for collection in ax2.collections:
        if isinstance(
            collection, collections.PathCollection
        ):  # Check if it's a scatter plot
            offsets = collection.get_offsets()  # Get scatter point positions
            ax1.scatter(
                offsets[:, 0],
                offsets[:, 1],  # X and Y positions
                label=collection.get_label(),
                color=collection.get_facecolor(),
                edgecolors=collection.get_edgecolor(),
                s=collection.get_sizes(),
                alpha=1.0,  # Keep new scatter points fully visible
            )

    # Copy legends if necessary
    ax1.legend()
    fig1.tight_layout()

    # Return the updated first figure
    return fig1, ax1

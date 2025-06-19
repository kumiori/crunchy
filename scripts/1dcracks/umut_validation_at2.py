import hashlib
import logging
import os
import numpy as np
import yaml
import dolfinx
from dolfinx.fem import (
    Function,
    assemble_scalar,
    form,
    locate_dofs_geometrical,
    dirichletbc,
)
import pyvista
import matplotlib.pyplot as plt

from ufl import Measure as measure
from ufl import inner, grad, dx
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.algorithms.am import HybridSolver
from irrevolutions.algorithms.ls import StabilityStepper, LineSearch
from irrevolutions.utils import (
    setup_logger_mpi,
    _logger,
    Visualization,
    # history_data,
    ColorPrint,
)
from dolfinx.common import list_timings

from crunchy.core import (
    initialise_functions,
    create_function_spaces_nd,
    setup_output_directory,
    save_parameters,
    initialise_functions,
)
from irrevolutions.models.one_dimensional import FilmModel1D as ThinFilm
from irrevolutions.solvers.function import vec_to_functions
from irrevolutions.utils.viz import plot_profile
from irrevolutions.utils.plots import plot_AMit_load, plot_energies
from crunchy.plots import plot_spectrum
from crunchy.utils import dump_output, write_history_data, new_history_data

comm = MPI.COMM_WORLD


def run_computation(parameters, storage=None, logger=_logger):
    Lx = parameters["geometry"]["Lx"]
    _nameExp = parameters["geometry"]["geom_type"]
    parameters["model"]["ell"]

    history_data = new_history_data()

    # Get geometry model
    parameters["geometry"]["geom_type"]
    # N = parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"]
    N = parameters["geometry"]["N"]
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, int(N))
    outdir = os.path.join(os.path.dirname(__file__), "output")

    # prefix = setup_output_directory(storage, parameters, outdir)
    prefix = storage

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    save_parameters(parameters, prefix)

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    # Functional Setting
    V_u, V_alpha = create_function_spaces_nd(mesh, dim=1)
    u, u_, alpha, β, v, state = initialise_functions(V_u, V_alpha)
    # Bounds
    alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
    alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")

    u_zero = Function(V_u, name="InelasticDisplacement")
    eps_t = dolfinx.fem.Constant(mesh, np.array(1.0, dtype=PETSc.ScalarType))
    u_zero.interpolate(lambda x: eps_t / 2.0 * (2 * x[0] - Lx))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    for f in [u, u_zero, alpha_lb, alpha_ub]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
    # Redundant boundary conditions
    bcs_u = [dirichletbc(u_zero, dofs_u_right), dirichletbc(u_zero, dofs_u_left)]
    dx = measure("dx", domain=mesh)

    bcs_alpha = []
    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
    model = ThinFilm(parameters["model"])
    total_energy = (
        model.elastic_energy_density(state, u_zero) + model.damage_energy_density(state)
    ) * dx

    loads = np.linspace(
        parameters["loading"]["min"],
        parameters["loading"]["max"],
        parameters["loading"]["steps"],
    )
    iterator = StabilityStepper(loads)

    equilibrium = HybridSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    bifurcation = BifurcationSolver(
        total_energy, state, bcs, bifurcation_parameters=parameters.get("stability")
    )

    stability = StabilitySolver(
        total_energy, state, bcs, cone_parameters=parameters.get("stability")
    )

    linesearch = LineSearch(
        total_energy,
        state,
        linesearch_parameters=parameters.get("stability").get("linesearch"),
    )

    logging.basicConfig(level=logging.INFO)

    arclength = []
    perturbation_log = {
        "steps": [],
        "time_stamps": [],
        "perturbations": [],
    }

    while True:
        try:
            # i_t = next(iterator)
            i_t = next(iterator) - 1
        except StopIteration:
            break

        # Perform your time step with t
        eps_t.value = loads[i_t]
        u_zero.interpolate(lambda x: eps_t / 2.0 * (2 * x[0] - Lx))
        u_zero.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
        alpha_lb.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # Log current load
        _logger.critical(f"-- Solving for t = {loads[i_t]:3.2f} --")
        with dolfinx.common.Timer(f"~First Order: Equilibrium") as timer:
            equilibrium.solve(alpha_lb)

        is_unique = bifurcation.solve(alpha_lb)
        is_elastic = not bifurcation._is_critical(alpha_lb)
        inertia = bifurcation.get_inertia()

        # The initial guess for the stability solver is the
        # first negative eigenmode (if it exists) or the first eigenmode
        z0 = (
            bifurcation._spectrum[0]["xk"]
            if bifurcation._spectrum and "xk" in bifurcation._spectrum[0]
            else bifurcation.spectrum[0]["xk"]
        )
        stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)

        _logger.info(f"Stability of state at load {loads[i_t]:.2f}: {stable}")
        ColorPrint.print_bold(f"Evolution is unique: {is_unique}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        ColorPrint.print_bold(f"State's stable: {stable}")
        ColorPrint.print_bold(f"===================- {storage} -=================")

        fracture_energy, elastic_energy = postprocess(
            parameters,
            _nameExp,
            prefix,
            v,
            β,
            state,
            loads[i_t],
            u_zero,
            dx,
            loads,
            equilibrium,
            bifurcation,
            stability,
            inertia,
            i_t,
            model=model,
            history_data=history_data,
            signature=signature,
        )

        if not stable:
            iterator.pause_time()
            _logger.info(f"Time paused at {loads[i_t]:.2f}")
            vec_to_functions(stability.solution["xt"], [v, β])
            perturbation = {"v": v, "beta": β}
            interval = linesearch.get_unilateral_interval(state, perturbation)

            order = 4
            h_opt, energies_1d, p, _ = linesearch.search(
                state, perturbation, interval, m=order
            )
            arclength.append((i_t, loads[i_t], h_opt))
            linesearch.perturb(state, perturbation, h_opt)

            perturbation_entry = {
                "step": i_t,
                "time": loads[i_t],
                "interval": interval,
                "h_opt": h_opt,
                "cone_lambda_t": stability.solution["lambda_t"],
                "eigenvalues": bifurcation.data["eigs"],
                "energy_profile": energies_1d,
            }
            perturbation_log["steps"].append(perturbation_entry)

            with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
                _logger.critical(f" *> State is unstable: {not stable}")
                _logger.critical(f"line search interval is {interval}")
                _logger.critical(f"perturbation energies: {energies_1d}")
                _logger.critical(f"hopt: {h_opt}")
                _logger.critical(f"lambda_t: {stability.solution['lambda_t']}")

                h_steps = np.linspace(interval[0], interval[1], order + 1)

                fig, axes = plt.subplots(1, 1)
                plt.scatter(h_steps, energies_1d)
                plt.scatter(
                    h_opt, 0, c="k", s=40, marker="|", label=f"$h^*={h_opt:.2f}$"
                )
                plt.scatter(h_opt, p(h_opt), c="k", s=40, alpha=0.5)
                xs = np.linspace(interval[0], interval[1], 30)
                axes.plot(xs, p(xs), label="Energy slice along perturbation")
                axes.set_xlabel("h")
                axes.set_ylabel("$E_h - E_0$")
                axes.set_title(f"Polynomial Interpolation - order {order}")
                axes.legend()
                axes.spines["top"].set_visible(False)
                axes.spines["right"].set_visible(False)
                axes.spines["left"].set_visible(False)
                axes.spines["bottom"].set_visible(False)
                axes.set_yticks([0])
                axes.axhline(0, c="k")
                fig.savefig(f"{prefix}/energy_interpolation-order-{order}.png")
                plt.close()

                tol = 1e-3
                assert Lx == 1
                npoints = int(
                    (
                        parameters["model"]["ell"]
                        / parameters["geometry"]["mesh_size_factor"]
                    )
                    ** (-1)
                )
                xs = np.linspace(0 + tol, Lx - tol, npoints)
                points = np.zeros((3, npoints))
                points[0] = xs

                plotter = pyvista.Plotter(
                    title="Perturbation profile",
                    window_size=[800, 600],
                    shape=(1, 1),
                )

                plot, data = plot_profile(
                    # β,
                    perturbation["beta"],
                    # stability.perturbation['β'],
                    points,
                    plotter,
                    lineproperties={"c": "k", "label": f"$\\beta$"},
                )
                plot.gca()
                plot.legend()
                plot.fill_between(data[0], data[1].reshape(len(data[1])))
                plot.title("Perurbation")
                plot.savefig(f"{prefix}/perturbation-profile-{i_t}.png")
                plot.close()
        else:
            dump_output(
                _nameExp,
                prefix,
                history_data,
                u,
                alpha,
                equilibrium,
                bifurcation,
                stability,
                inertia,
                loads[i_t],
                perturbation_log,
                fracture_energy,
                elastic_energy,
            )

    return history_data, stability.data, state


def load_parameters(file_path, ndofs, model="at2"):
    """
    Load parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML parameter file.

    Returns:
        dict: Loaded parameters.
    """
    import hashlib

    with open(file_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["model"]["model_dimension"] = 1
    parameters["model"]["model_type"] = "1D"
    parameters["geometry"]["geom_type"] = "thinfilm"
    # Get mesh parameters

    if model == "at2":
        parameters["loading"]["min"] = 0.0
        parameters["loading"]["max"] = 3.5
        parameters["loading"]["steps"] = 30

    parameters["geometry"]["geom_type"] = "1d-bar"
    parameters["geometry"]["mesh_size_factor"] = 4
    parameters["geometry"]["N"] = ndofs

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-5
    parameters["stability"]["cone"]["cone_rtol"] = 1e-5
    parameters["stability"]["cone"]["scaling"] = 1e-3

    parameters["model"]["w1"] = 1
    parameters["model"]["at_number"] = 2
    parameters["model"]["ell"] = 0.158114
    parameters["model"]["k_res"] = 0.0
    parameters["model"]["mu"] = 1
    # parameters["model"]["kappa"] = (0.34) ** (-2)
    parameters["model"]["kappa"] = (0.34) ** (-2)

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


def postprocess(
    parameters,
    _nameExp,
    prefix,
    β,
    v,
    state,
    t,
    u_zero,
    dx,
    loads,
    equilibrium,
    bifurcation,
    stability,
    inertia,
    i_t,
    model,
    history_data,
    signature="",
):
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        _alpha_anal = t**2 / (2 + t**2)

        fracture_energy = comm.allreduce(
            assemble_scalar(form(model.damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(model.elastic_energy_density(state, u_zero) * dx)),
            op=MPI.SUM,
        )

        l2_norm_alpha = np.sqrt(
            assemble_scalar(form(inner(state["alpha"], state["alpha"]) * dx))
        )
        l2_norm_u = np.sqrt(assemble_scalar(form(inner(state["u"], state["u"]) * dx)))
        l2_norm = np.sqrt(l2_norm_alpha**2 + l2_norm_u**2)

        # H1 Norms
        h1_norm_alpha = np.sqrt(
            assemble_scalar(
                form(
                    inner(state["alpha"], state["alpha"]) * dx
                    + inner(grad(state["alpha"]), grad(state["alpha"])) * dx
                )
            )
        )
        h1_norm_u = np.sqrt(
            assemble_scalar(
                form(
                    inner(state["u"], state["u"]) * dx
                    + inner(grad(state["u"]), grad(state["u"])) * dx
                )
            )
        )
        h1_norm = np.sqrt(h1_norm_alpha**2 + h1_norm_u**2)
        history_data["y_norms"].append(
            {
                "l2": l2_norm,
                "h1": h1_norm,
                # "infty": infty_norm,
                "alpha_l2": l2_norm_alpha,
                "u_l2": l2_norm_u,
                "alpha_sh1": np.sqrt(
                    assemble_scalar(
                        form(inner(grad(state["alpha"]), grad(state["alpha"])) * dx)
                    )
                ),
                "u_sh1": np.sqrt(
                    assemble_scalar(
                        form(inner(grad(state["u"]), grad(state["u"])) * dx)
                    )
                ),
            }
        )

        write_history_data(
            equilibrium=equilibrium,
            bifurcation=bifurcation,
            stability=stability,
            history_data=history_data,
            t=loads[i_t],
            inertia=inertia,
            stable=np.nan,
            energies=[elastic_energy, fracture_energy],
        )

        fig_state, ax1 = plt.subplots()

        if comm.rank == 0:
            plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
            plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
            try:
                fig, ax = plot_spectrum(history_data, signature=signature)
                ax.set_ylim(-0.1, 0.1)
                fig.savefig(f"{prefix}/spectrum.png")
                plt.close(fig)
            except Exception as e:
                _logger.error(f"Error plotting spectrum: {e}")
            # plot_force_displacement(
            #     history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf"
            # )

        # xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True

        plotter = pyvista.Plotter(
            title="Profiles",
            window_size=[800, 600],
            shape=(1, 1),
        )
        # npoints = int(
        #     (parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"])
        #     ** (-1)
        # )
        npoints = parameters["geometry"]["N"]
        tol = 1e-3
        xs = np.linspace(0 + tol, parameters["geometry"]["Lx"] - tol, npoints)
        points = np.zeros((3, npoints))
        points[0] = xs

        _plt, data = plot_profile(
            state["alpha"],
            points,
            plotter,
            lineproperties={
                "c": "k",
                "label": f"$\\alpha$ with $\\ell$ = {parameters['model']['ell']:.2f}",
            },
        )
        ax = _plt.gca()

        _plt, data = plot_profile(
            state["u"],
            points,
            plotter,
            fig=_plt,
            ax=ax,
            lineproperties={
                "c": "g",
                "label": "$u$",
                "marker": "o",
            },
        )

        _plt, data = plot_profile(
            u_zero,
            points,
            plotter,
            fig=_plt,
            ax=ax,
            lineproperties={"c": "r", "lw": 3, "label": "$u_0$"},
        )
        _plt.title("Solution state")
        # ax.set_ylim(-2.1, 2.1)
        ax.axhline(0, color="k", lw=0.5)

        # anal solution
        ax.axhline(_alpha_anal, color="k", lw=0.5)
        ax.plot(
            xs,
            t * (2 * xs - 1) / 2,
            label=f"Analytical $u$",
        )
        _plt.legend()

        _plt.savefig(f"{prefix}/state_profile-{i_t}.png")

        if bifurcation.spectrum:
            fig_bif, ax = plt.subplots()

            vec_to_functions(bifurcation.spectrum[0]["xk"], [v, β])

            _plt, data = plot_profile(
                β,
                points,
                plotter,
                fig=fig_bif,
                ax=ax,
                lineproperties={
                    "c": "k",
                    "label": f"$\\beta, \\lambda = {bifurcation.spectrum[0]['lambda']:.0e}$",
                },
            )
            ax.axhline(0, color="k", lw=0.5)
            ax2 = ax.twinx()
            if hasattr(stability, "perturbation"):
                if stability.perturbation["λ"] < 0:
                    _colour = "r"
                    _style = "--"
                else:
                    _colour = "b"
                    _style = ":"

                _plt, data = plot_profile(
                    stability.perturbation["β"],
                    points,
                    plotter,
                    fig=_plt,
                    ax=ax2,
                    lineproperties={
                        "c": _colour,
                        "ls": _style,
                        "lw": 3,
                        "label": f"$\\beta^+, \\lambda = {stability.perturbation['λ']:.0e}$",
                    },
                )
                ax2.axhline(0, color="b", lw=0.5)
                ax2.yaxis.label.set_color("b")
                ax2.tick_params(axis="y", colors="b", **dict(size=4, width=1.5))

                ax2.legend(loc="upper right")

            _plt.legend(loc="upper left")
            fig_bif.savefig(f"{prefix}/_second_order_profiles-{i_t}.png")
            plt.close("all")

        if bifurcation._spectrum:
            fig_bif, ax = plt.subplots()

            vec_to_functions(bifurcation._spectrum[0]["xk"], [v, β])

            _plt, data = plot_profile(
                β,
                points,
                plotter,
                fig=fig_bif,
                ax=ax,
                lineproperties={
                    "c": "k",
                    "label": f"$\\beta, \\lambda = {bifurcation._spectrum[0]['lambda']:.0e}$",
                },
            )
            _plt.legend()
            # _plt.fill_between(data[0], data[1].reshape(len(data[1])))

            if hasattr(stability, "perturbation"):
                if stability.perturbation["λ"] < 0:
                    _colour = "r"
                    _style = "--"
                else:
                    _colour = "b"
                    _style = ":"

                _plt, data = plot_profile(
                    stability.perturbation["β"],
                    points,
                    plotter,
                    fig=_plt,
                    ax=ax,
                    lineproperties={
                        "c": _colour,
                        "ls": _style,
                        "lw": 3,
                        "label": f"$\\beta^+, \\lambda = {stability.perturbation['λ']:.0e}$",
                    },
                )

                _plt.legend()
                # _plt.fill_between(data[0], data[1].reshape(len(data[1])))
                _plt.title("Perturbation profiles")
                # ax.set_ylim(-2.1, 2.1)
                ax.axhline(0, color="k", lw=0.5)
                fig_bif.savefig(f"{prefix}/second_order_profiles-{i_t}.png")

        # close figures
        plt.close("all")

    return fracture_energy, elastic_energy


if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    # Load parameters
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "umut_benchmark_parameters.yaml"),
        ndofs=101,
        model="at2",
    )
    # Run computation
    _storage = f"output/umut_benchmark_1d/{signature[0:6]}"

    visualization = Visualization(_storage)
    prefix = setup_output_directory(_storage, parameters, "")
    logger = setup_logger_mpi(filename=f"{_storage}/evolution.log")

    with dolfinx.common.Timer("~Computation Experiment") as timer:
        history_data, stability_data, state = run_computation(parameters, _storage)

    from irrevolutions.utils import table_timing_data

    tasks = [
        "~First Order: Equilibrium",
        "~First Order: AltMin-Damage solver",
        "~First Order: AltMin-Elastic solver",
        "~Postprocessing and Vis",
        "~Output and Storage",
        "~Computation Experiment",
    ]

    _timings = table_timing_data(tasks)
    _timings = table_timing_data()
    visualization.save_table(_timings, "timing_data")
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    ColorPrint.print_bold(f"===================- {signature} -=================")
    ColorPrint.print_bold(f"===================- {_storage} -=================")

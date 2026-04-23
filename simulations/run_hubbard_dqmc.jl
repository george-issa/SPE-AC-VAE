"""
Generate Hubbard-model DQMC data on a 2-D square lattice using MPI.
Based directly on tutorials/hubbard_square_mpi.jl from SmoQyDQMC.

Each MPI process runs an independent walker; final estimates are the
average across all walkers (= independent Monte Carlo simulations).

Filling is controlled by the chemical potential μ:
  - Half-filling (n=1):  μ = 0.0  (exact on bipartite lattice, ph_sym_form=true)
  - Other fillings:       μ ≠ 0.0  (tune manually, or use density-tuning tutorial)

Output folders (one per (U, μ, β)):
    Data/datasets/real/hubbard_square_U{U}_mu{μ}_L{L}_b{β}-{sID}/
        model_summary.toml
        time-displaced/greens/position/bin-{n}_pID-{pID}.jld2
        ...

Compatible with data_process_real.QMCPositionDataset.

Usage
-----
    julia --project simulations/run_hubbard_dqmc.jl                # full grid
    julia --project simulations/run_hubbard_dqmc.jl <sID> <U> <μ> <β>

Run from the project root (SPE-AC-VAE/). --project activates the environment once
at the Julia level; mpirun passes it through to all worker processes automatically.
"""

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf
using MPI
using Random
using Printf

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
const L              = 4        # L×L square lattice
const T_HOP          = 1.0      # nearest-neighbor hopping (sets energy scale)
const T_PRIME        = 0.0      # next-nearest hopping (0 = pure Hubbard)
const Δτ             = 0.05     # imaginary-time step
const N_THERM        = 5_000    # thermalization sweeps
const N_MEASUREMENTS = 10_000   # total measurement sweeps
const N_BINS         = 100      # bins written to disk (samples for VAE)
const N_UPDATES      = 10       # reflection+local updates per measurement step
const N_STAB         = 10       # Green's function stabilization period
const δG_MAX         = 1e-6     # numerical stability threshold
const SEED           = 42
const N_WALKERS      = 8        # MPI processes (independent walkers); total bins = N_WALKERS × N_BINS

# ── Parameter grid ──────────────────────────────────────────────────────────
# μ = 0.0 → half-filling (exact for repulsive Hubbard on bipartite lattice).
# To scan other fillings, add μ values here (negative μ = below half-filling).
const US   = [6.0]
const MUS  = [0.0]                    # chemical potentials to simulate
const BETAS = [8.0, 10.0]

const OUTPUT_ROOT = joinpath(@__DIR__, "..", "Data", "datasets", "real")
# ═══════════════════════════════════════════════════════════════════════════

function run_simulation(
    comm::MPI.Comm;
    sID::Int,
    U::Float64,
    μ::Float64,
    β::Float64,
    t′::Float64  = T_PRIME,
    L::Int       = L,
    Δτ::Float64  = Δτ,
    n_stab::Int  = N_STAB,
    δG_max::Float64    = δG_MAX,
    symmetric::Bool    = false,
    checkerboard::Bool = false,
    seed::Int          = SEED + sID,
    filepath::String   = OUTPUT_ROOT
)
    datafolder_prefix = @sprintf "hubbard_square_U%.2f_mu%.2f_L%d_b%.2f" U μ L β
    pID = MPI.Comm_rank(comm)

    simulation_info = SimulationInfo(
        filepath              = filepath,
        datafolder_prefix     = datafolder_prefix,
        write_bins_concurrent = (L > 10),
        sID                   = sID,
        pID                   = pID
    )

    initialize_datafolder(comm, simulation_info)

    t_start = time()
    rng = Xoshiro(seed)

    metadata = Dict{String, Any}(
        "N_therm"               => N_THERM,
        "N_measurements"        => N_MEASUREMENTS,
        "N_updates"             => N_UPDATES,
        "N_bins"                => N_BINS,
        "n_stab_init"           => n_stab,
        "dG_max"                => δG_max,
        "symmetric"             => symmetric,
        "checkerboard"          => checkerboard,
        "seed"                  => seed,
        "local_acceptance_rate"      => 0.0,
        "reflection_acceptance_rate" => 0.0
    )

    # ── Geometry ──────────────────────────────────────────────────────────
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
        basis_vecs   = [[0.0, 0.0]]
    )
    lattice = lu.Lattice(L = [L, L], periodic = [true, true])
    model_geometry = ModelGeometry(unit_cell, lattice)

    bond_px    = lu.Bond(orbitals = (1,1), displacement = [1,  0])
    bond_px_id = add_bond!(model_geometry, bond_px)
    bond_py    = lu.Bond(orbitals = (1,1), displacement = [0,  1])
    bond_py_id = add_bond!(model_geometry, bond_py)
    bond_nx    = lu.Bond(orbitals = (1,1), displacement = [-1, 0])
    bond_nx_id = add_bond!(model_geometry, bond_nx)
    bond_ny    = lu.Bond(orbitals = (1,1), displacement = [0, -1])
    bond_ny_id = add_bond!(model_geometry, bond_ny)
    bond_pxpy  = lu.Bond(orbitals = (1,1), displacement = [1,  1])
    bond_pxpy_id = add_bond!(model_geometry, bond_pxpy)
    bond_pxny  = lu.Bond(orbitals = (1,1), displacement = [1, -1])
    bond_pxny_id = add_bond!(model_geometry, bond_pxny)

    t = T_HOP

    # ── Models ────────────────────────────────────────────────────────────
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds  = [bond_px, bond_py, bond_pxpy, bond_pxny],
        t_mean   = [t, t, t′, t′],
        t_std    = [0., 0., 0., 0.],
        ϵ_mean   = [0.],
        ϵ_std    = [0.],
        μ        = μ
    )

    hubbard_model = HubbardModel(
        ph_sym_form = true,
        U_orbital   = [1],
        U_mean      = [U],
        U_std       = [0.],
    )

    model_summary(
        simulation_info     = simulation_info,
        β                   = β,
        Δτ                  = Δτ,
        model_geometry      = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions        = (hubbard_model,)
    )

    # ── Parameters ────────────────────────────────────────────────────────
    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model,
        model_geometry      = model_geometry,
        rng                 = rng
    )

    hubbard_parameters = HubbardParameters(
        model_geometry = model_geometry,
        hubbard_model  = hubbard_model,
        rng            = rng
    )

    hst_parameters = HubbardSpinHirschHST(
        β                  = β,
        Δτ                 = Δτ,
        hubbard_parameters = hubbard_parameters,
        rng                = rng
    )

    # ── Measurements ──────────────────────────────────────────────────────
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    initialize_measurements!(measurement_container, tight_binding_model)
    initialize_measurements!(measurement_container, hubbard_model)

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry        = model_geometry,
        correlation           = "greens",
        time_displaced        = true,
        pairs                 = [(1, 1)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry        = model_geometry,
        correlation           = "density",
        time_displaced        = false,
        integrated            = true,
        pairs                 = [(1, 1)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry        = model_geometry,
        correlation           = "pair",
        time_displaced        = false,
        integrated            = true,
        pairs                 = [(1, 1)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry        = model_geometry,
        correlation           = "spin_z",
        time_displaced        = false,
        integrated            = true,
        pairs                 = [(1, 1)]
    )

    initialize_composite_correlation_measurement!(
        measurement_container = measurement_container,
        model_geometry        = model_geometry,
        name                  = "d-wave",
        correlation           = "pair",
        ids                   = [bond_px_id, bond_nx_id, bond_py_id, bond_ny_id],
        coefficients          = [0.5, 0.5, -0.5, -0.5],
        time_displaced        = false,
        integrated            = true
    )

    # ── Fermion path integral ─────────────────────────────────────────────
    fermion_path_integral_up = FermionPathIntegral(
        tight_binding_parameters = tight_binding_parameters,
        β = β, Δτ = Δτ,
        forced_complex_potential = (U < 0),
        forced_complex_kinetic   = false
    )
    fermion_path_integral_dn = FermionPathIntegral(
        tight_binding_parameters = tight_binding_parameters,
        β = β, Δτ = Δτ,
        forced_complex_potential = (U < 0),
        forced_complex_kinetic   = false
    )

    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_parameters)
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hst_parameters)

    Bup = initialize_propagators(fermion_path_integral_up, symmetric = symmetric, checkerboard = checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric = symmetric, checkerboard = checkerboard)

    fermion_greens_calculator_up     = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn     = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)
    fermion_greens_calculator_up_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_up)
    fermion_greens_calculator_dn_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_dn)

    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))

    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calculator_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn)

    Gup_ττ = similar(Gup); Gup_τ0 = similar(Gup); Gup_0τ = similar(Gup)
    Gdn_ττ = similar(Gdn); Gdn_τ0 = similar(Gdn); Gdn_0τ = similar(Gdn)

    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    # ── Thermalization ────────────────────────────────────────────────────
    for _ in 1:N_THERM

        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hst_parameters,
            fermion_path_integral_up          = fermion_path_integral_up,
            fermion_path_integral_dn          = fermion_path_integral_dn,
            fermion_greens_calculator_up      = fermion_greens_calculator_up,
            fermion_greens_calculator_dn      = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt  = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt  = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn, rng = rng
        )
        metadata["reflection_acceptance_rate"] += accepted

        (acc, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hst_parameters,
            fermion_path_integral_up     = fermion_path_integral_up,
            fermion_path_integral_dn     = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn,
            δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = true
        )
        metadata["local_acceptance_rate"] += acc
    end

    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    # ── Production / measurement loop ─────────────────────────────────────
    bin_size = N_MEASUREMENTS ÷ N_BINS

    for measurement in 1:N_MEASUREMENTS

        for _ in 1:N_UPDATES

            (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hst_parameters,
                fermion_path_integral_up          = fermion_path_integral_up,
                fermion_path_integral_dn          = fermion_path_integral_dn,
                fermion_greens_calculator_up      = fermion_greens_calculator_up,
                fermion_greens_calculator_dn      = fermion_greens_calculator_dn,
                fermion_greens_calculator_up_alt  = fermion_greens_calculator_up_alt,
                fermion_greens_calculator_dn_alt  = fermion_greens_calculator_dn_alt,
                Bup = Bup, Bdn = Bdn, rng = rng
            )
            metadata["reflection_acceptance_rate"] += accepted

            (acc, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hst_parameters,
                fermion_path_integral_up     = fermion_path_integral_up,
                fermion_path_integral_dn     = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn,
                δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
                update_stabilization_frequency = true
            )
            metadata["local_acceptance_rate"] += acc
        end

        (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = make_measurements!(
            measurement_container,
            logdetGup, sgndetGup, Gup, Gup_ττ, Gup_τ0, Gup_0τ,
            logdetGdn, sgndetGdn, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
            fermion_path_integral_up     = fermion_path_integral_up,
            fermion_path_integral_dn     = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn,
            δG_max = δG_max, δG = δG, δθ = δθ,
            model_geometry           = model_geometry,
            tight_binding_parameters = tight_binding_parameters,
            coupling_parameters      = (hubbard_parameters, hst_parameters)
        )

        write_measurements!(
            measurement_container = measurement_container,
            simulation_info       = simulation_info,
            model_geometry        = model_geometry,
            measurement           = measurement,
            bin_size              = bin_size,
            Δτ                    = Δτ
        )
    end

    # ── Finalize ──────────────────────────────────────────────────────────
    merge_bins(simulation_info)

    total = N_THERM + N_MEASUREMENTS * N_UPDATES
    metadata["local_acceptance_rate"]      /= total
    metadata["reflection_acceptance_rate"] /= total
    metadata["n_stab_final"] = fermion_greens_calculator_up.n_stab
    metadata["dG"] = δG
    metadata["runtime_seconds"] = time() - t_start
    metadata["n_walkers"] = MPI.Comm_size(comm)
    metadata["hostname"] = gethostname()

    save_simulation_info(simulation_info, metadata)

    process_measurements(
        comm;
        datafolder          = simulation_info.datafolder,
        n_bins              = N_BINS,
        export_to_csv       = true,
        scientific_notation = false,
        decimals            = 7,
        delimiter           = " "
    )

    Rafm, ΔRafm = compute_correlation_ratio(
        comm;
        datafolder           = simulation_info.datafolder,
        correlation          = "spin_z",
        type                 = "equal-time",
        id_pairs             = [(1, 1)],
        id_pair_coefficients = [1.0],
        q_point              = (L÷2, L÷2),
        q_neighbors          = [
            (L÷2+1, L÷2), (L÷2-1, L÷2),
            (L÷2, L÷2+1), (L÷2, L÷2-1)
        ]
    )
    metadata["Rafm_mean_real"] = real(Rafm)
    metadata["Rafm_mean_imag"] = imag(Rafm)
    metadata["Rafm_std"]       = ΔRafm
    save_simulation_info(simulation_info, metadata)

    return nothing
end

# ── Entry point ───────────────────────────────────────────────────────────────
function main()
    MPI.Init()
    comm = MPI.COMM_WORLD

    # If launched as a single process but N_WALKERS > 1, re-launch with mpirun.
    if MPI.Comm_size(comm) == 1 && N_WALKERS > 1
        MPI.Finalize()
        run(`$(MPI.mpiexec()) -np $N_WALKERS $(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $(@__FILE__) $(ARGS...)`)
        return
    end

    if length(ARGS) == 4
        run_simulation(comm;
            sID = parse(Int,     ARGS[1]),
            U   = parse(Float64, ARGS[2]),
            μ   = parse(Float64, ARGS[3]),
            β   = parse(Float64, ARGS[4])
        )
    else
        sID = 1
        for β in BETAS, μ in MUS, U in US
            run_simulation(comm; sID = sID, U = U, μ = μ, β = β)
        end
    end

    println("All simulations complete.")
    MPI.Finalize()
end

main()

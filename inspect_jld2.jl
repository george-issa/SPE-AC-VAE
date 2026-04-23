using JLD2, HDF5, Statistics

jld2_path = joinpath(@__DIR__, "Data", "datasets", "george_325.jld2")
h5_path   = joinpath(@__DIR__, "Data", "datasets", "george_325_plain.h5")

println("=" ^ 60)
println("Loading: $jld2_path")
println("=" ^ 60)

f = jldopen(jld2_path, "r")

println("\nKeys in file: ", keys(f))
println()

for key in keys(f)
    arr = f[key]
    T   = typeof(arr)
    sz  = size(arr)
    nz  = count(!iszero, arr)
    tot = length(arr)
    println("── $key ──────────────────────────────")
    println("  Julia type : $T")
    println("  size       : $sz")
    println("  eltype     : $(eltype(arr))")
    println("  nonzero    : $nz / $tot")
    if nz > 0
        println("  min        : $(minimum(arr))")
        println("  max        : $(maximum(arr))")
        println("  mean       : $(mean(arr))")
        # Show a small slice
        if ndims(arr) >= 2
            println("  arr[1,1,...]: $(arr[1,1, (1 for _ in 3:ndims(arr))...])")
        else
            println("  arr[1:5]   : $(arr[1:min(5,end)])")
        end
    else
        println("  *** ALL ZEROS ***")
    end
    println()
end

close(f)

# ── Write plain HDF5 ───────────────────────────────────────────────────────
println("=" ^ 60)
println("Writing plain HDF5 to: $h5_path")
println("=" ^ 60)

f    = jldopen(jld2_path, "r")
G_r  = f["G_r"]
dos  = f["dos"]
ws   = f["ws"]
close(f)

h5open(h5_path, "w") do fh
    fh["G_r"] = G_r
    fh["dos"] = dos
    fh["ws"]  = ws
    # Store parameter grids as attributes for convenience
    attrs(fh["G_r"])["axes"] = "ns × Ωs × βs × ntau × bins"
    attrs(fh["dos"])["axes"] = "ns × Ωs × βs × ws"
    attrs(fh["ws"])["axes"]  = "omega grid"
    attrs(fh)["ns"]     = collect(0.05:0.05:1.0)
    attrs(fh)["Omegas"] = collect(0.5:0.5:2.0)
    attrs(fh)["betas"]  = collect(5.0:1.0:20.0)
end

println("Done.")

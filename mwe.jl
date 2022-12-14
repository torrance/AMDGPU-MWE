using AMDGPU
using BenchmarkTools

struct Origin
    u0::Float32
    v0::Float32
    w0::Float32
end

# CPU version used for testing
function dift!(subgrid::Array, origin, us, vs, data)
    idxs = CartesianIndices(size(subgrid))

    Threads.@threads :static for idx in idxs
        # lpx, mpx = Tuple(idx)
        lpx, mpx = 5, 4

        for (u, v, datum) in zip(us, vs, data)
            phase = 2 * (
                (u - origin.u0) * lpx +
                (v - origin.v0) * mpx
            )
            subgrid[idx] += datum * phase # cispi(phase)
        end
    end
end

# GPU version
@inbounds function dift!(subgrid::ROCDeviceArray{ComplexF32, 2}, origin::Origin, us::AbstractVector{Float32}, vs::AbstractVector{Float32}, data::AbstractVector{ComplexF32})
    idx = (workgroupDim().x * (workgroupIdx().x - 1)) + workitemIdx().x
    if idx > length(subgrid)
        return
    end

    # mpx, lpx = fldmod1(idx, size(subgrid, 1))
    mpx, lpx = 4, 5

    cell = zero(ComplexF32)
    for (u, v, datum) in zip(us, vs, data)
        phase = 2 * (
            (u - origin.u0) * lpx +
            (v - origin.v0) * mpx
        )::Float32
        # No cispi() available, and sincospi() seems to kill the kernel
        # imag, real = sincospi(phase)
        cell += datum * phase # ComplexF32(cospi(phase), sinpi(phase))
    end

    subgrid[idx] = cell
    return nothing
end

# Create dummy data
subgrid = ROCArray{ComplexF32}(undef, 96, 96)
fill!(subgrid, zero(eltype(subgrid)))

origin = Origin(rand(3)...)

N = 1024 * 200
us = rand(Float32, N) .- 0.5f0
vs = rand(Float32, N) .- 0.5f0
data = rand(ComplexF32, N)

# Calculcate expected result on CPU
expected = zeros(ComplexF32, 96, 96)
dift!(expected, origin, us, vs, data)

# Transfer to GPU
us, vs, data = ROCArray(us), ROCArray(vs), ROCArray(data)

# Test @roc version
@roc threads=512 blocks=cld(length(subgrid), 512) dift!(subgrid, origin, us, vs, data)
@assert expected ≈ Array(subgrid) "@roc"
println("@roc OK")

# Benchmark @roc kernel
b = @benchmark begin
    wait(@roc threads=512 blocks=cld(length(data), 512) dift!($subgrid, $origin, $us, $vs, $data))
end samples=10
show(stdout, MIME"text/plain"(), b)
println()
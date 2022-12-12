using AMDGPU
using BenchmarkTools

function kernel1!(ys, xs)
    idx = (workgroupDim().x * (workgroupIdx().x - 1)) + workitemIdx().x
    if idx > length(xs)
        return
    end

    @inbounds x = xs[idx]
    @inbounds ys[idx] = ComplexF32(cos(x), sin(x))
    return
end

function kernel2!(ys, xs)
    idx = (workgroupDim().x * (workgroupIdx().x - 1)) + workitemIdx().x
    if idx > length(xs)
        return
    end

    @inbounds x = xs[idx]
    imag, real = sincos(x)
    @inbounds ys[idx] = ComplexF32(real, imag)
    return
end

function kernel3!(ys, xs)
    idx = (workgroupDim().x * (workgroupIdx().x - 1)) + workitemIdx().x
    if idx > length(xs)
        return
    end

    @inbounds x = xs[idx]
    @inbounds ys[idx] = cis(x)
    return
end


xs = AMDGPU.rand(Float32, 100_000)
ys = AMDGPU.zeros(ComplexF32, 100_000)
expected = cis.(Array(xs))


println("sin() / cos()")

fill!(ys, 0)
b = @benchmark begin
    wait(@roc threads=512 blocks=cld(length(xs), 512) kernel1!(ys, xs))
end samples=100
show(stdout, MIME"text/plain"(), b)
println()

@assert Array(ys) ≈ expected

println("sincos()")

fill!(ys, 0)
b = @benchmark begin
    wait(@roc threads=512 blocks=cld(length(xs), 512) kernel2!(ys, xs))
end samples=100
show(stdout, MIME"text/plain"(), b)
println()

@assert Array(ys) ≈ expected

println("cis()")

fill!(ys, 0)
b = @benchmark begin
    wait(@roc threads=512 blocks=cld(length(xs), 512) kernel3!(ys, xs))
end samples=100
show(stdout, MIME"text/plain"(), b)
println()

@assert Array(ys) ≈ expected
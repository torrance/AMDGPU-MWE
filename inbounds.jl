using AMDGPU
using BenchmarkTools

@inbounds function kernel1!(zs, xs, ys)
    idx = (workgroupDim().x * (workgroupIdx().x - 1)) + workitemIdx().x
    if idx > length(zs)
        return
    end

    tmp::Float32 = 0
    for i in eachindex(xs)
        tmp += xs[i] + ys[i]
    end
    zs[idx] = tmp

    return
end

function kernel2!(zs, xs, ys)
    idx = (workgroupDim().x * (workgroupIdx().x - 1)) + workitemIdx().x
    if idx > length(zs)
        return
    end

    tmp::Float32 = 0
    for i in eachindex(xs)
        @inbounds tmp += xs[i] + ys[i]
    end
    @inbounds zs[idx] = tmp

    return
end

function kernel3!(zs, xs, ys)
    idx = (workgroupDim().x * (workgroupIdx().x - 1)) + workitemIdx().x
    if idx > length(zs)
        return
    end

    tmp::Float32 = 0
    @inbounds for (x, y) in zip(xs, ys)
        tmp += x + y
    end
    @inbounds zs[idx] = tmp

    return
end

xs = AMDGPU.rand(Float32, 100_000)
ys = AMDGPU.rand(Float32, 100_000)
zs = AMDGPU.zeros(Float32, 1_000_000)

println("Function @inbounds")
b = @benchmark begin
    wait(@roc threads=512 blocks=cld(length(zs), 512) kernel1!(zs, xs, ys))
end samples=100
show(stdout, MIME"text/plain"(), b)
println()

println("Internal @inbounds")
b = @benchmark begin
    wait(@roc threads=512 blocks=cld(length(zs), 512) kernel2!(zs, xs, ys))
end samples=100
show(stdout, MIME"text/plain"(), b)
println()

println("Using zip()")
b = @benchmark begin
    wait(@roc threads=512 blocks=cld(length(zs), 512) kernel3!(zs, xs, ys))
end samples=100
show(stdout, MIME"text/plain"(), b)
println()

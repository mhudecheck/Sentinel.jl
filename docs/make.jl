using Documenter, Sentinel, CUDA
#push!(LOAD_PATH,"../src/")
makedocs(sitename="My Documentation")
deploydocs(
    repo = "github.com/mhudecheck/Sentinel.jl.git",
)

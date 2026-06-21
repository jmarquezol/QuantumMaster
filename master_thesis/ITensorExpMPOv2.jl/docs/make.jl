using ITensorExpMPO: ITensorExpMPO
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(
  ITensorExpMPO, :DocTestSetup, :(using ITensorExpMPO); recursive=true
)

include("make_index.jl")

makedocs(;
  modules=[ITensorExpMPO],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="ITensorExpMPO.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/ITensorExpMPO.jl",
    edit_link="main",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(;
  repo="github.com/ITensor/ITensorExpMPO.jl", devbranch="main", push_preview=true
)

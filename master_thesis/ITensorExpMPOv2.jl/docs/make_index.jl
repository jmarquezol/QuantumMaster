using Literate: Literate
using ITensorExpMPO: ITensorExpMPO

Literate.markdown(
  joinpath(pkgdir(ITensorExpMPO), "examples", "README.jl"),
  joinpath(pkgdir(ITensorExpMPO), "docs", "src");
  flavor=Literate.DocumenterFlavor(),
  name="index",
)

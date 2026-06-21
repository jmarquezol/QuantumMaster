using Literate: Literate
using ITensorExpMPO: ITensorExpMPO

Literate.markdown(
  joinpath(pkgdir(ITensorExpMPO), "examples", "README.jl"),
  joinpath(pkgdir(ITensorExpMPO));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)

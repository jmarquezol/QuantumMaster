Loschmidt echo, emerging dual unitarity and scaling of
generalized temporal entropies after quenches to the critical
point
Stefano Carignano1 and Luca Tagliacozzo2
1BarcelonaSupercomputingCenter,08034Barcelona,Spain
2InstituteofFundamentalPhysicsIFF-CSIC,CalleSerrano113b,Madrid28006,Spain
We show how the Loschmidt echo of a prod- The relation between these classes is still not com-
uct state after a quench to a conformal invari- pletely clear. For example, dual unitary circuits are
ant critical point and its leading finite time quantum circuits that describe the Floquet dynamics
corrections can be predicted by using con- of a system driven by local interactions encoded in
formal field theories (CFT). We check such elementary gates, which possess an underlying space-
predictions with tensor networks, finding ex- time duality that results in a circuit encoding a uni-
cellent agreement. As a result, we can use tary dynamics in both time and space. Dual unitary
the Loschmidt echo to extract the universal gates have also been used to construct toy models for
information of the underlying CFT including holography which display emerging discrete Lorentz
the central charge, the operator content, and and conformal invariance [19], hinting to a possible
its generalized temporal entropies. We are connection between dual unitarity and conformal in-
also able to predict and confirm an emerging variance.
dual-unitarity of the evolution at late times,
In this work, we start from Hamiltonian dynamics
since the spatial transfer matrix operator that
and focus on the Loschmidt echo, namely the return
evolvesthesysteminspacebecomesunitaryin
probability of an initial product state, after a quench
such limit. Our results on the growth of tem-
to the critical conformal invariant point. The behav-
poral entropies also imply that, using state-of-
ior of such return probability provides key informa-
the art tensor networks algorithms, such cal-
tion of the dynamics and can be used as a probe of
culations only require resources that increase
quantum chaos since its behavior can be related to
polynomially with the duration of the quench,
that of out-of-time ordered correlations [20] (for ear-
thus providing an example of numerically effi-
lier studies see also [21, 22, 23], for a review see [24]).
ciently solvable out-of-equilibrium scenario.
We map the problem into a path integral calculation
of a conformal field theory (CFT) on a strip where
the boundary correspond to the initial state. Such
1 Introduction.
boundary CFT (bCFT) can be solved using the pow-
erfulmachineryofconformalmaps,whichallowsusto
Strongly-correlated quantum systems out-of-
obtain predictions about the leading Loschmidt echo
equilibrium still challenge our understanding due to
exponential decay at large times. We unveil that the
the exponential complexity involved in making pre-
leading decay is dictated by the central charge of the
dictions for them. This complexity has far-reaching
CFT [25, 26], which measures the number of degrees
implications, such as the ongoing debate surrounding
of freedom of the theory [27]. We also show how the
the existence of the many-body localized phase, a
finite-time corrections to this leading exponential de-
problem that demands simulations of large systems
caydependontheinitialstate,andaredictatedbythe
over long time scales [1, 2].
operator content of the corresponding bCFT [28, 29].
As a result, currently most of our understanding of
We find that the late-time Loschmidt echo of a sys-
the out-of-equilibrium dynamics emerges either from
temdescribedbyCFTgivesrisetoaunitarytransfer
short-time numerical simulations or from scenarios
matrix in space, providing an example of emerging
where analytical calculations can be performed. This
dual-unitarity at late times.
is possible e.g. in integrable models [3, 4, 5, 6], con-
formal field theories [7, 8, 9, 10, 11, 12] and Floquet Finally,wedefineandcharacterizegeneralizedtem-
systems such as random [13] or dual unitary circuits poral entropies, that arise by studying reduced tran-
[14, 15, 16, 17, 18]. sition matrices corresponding to a time-like cut in
the path-integral. These complex-valued entropies
StefanoCarignano: stefano.carignano@bsc.es
are known to carry geometric information in holo-
LucaTagliacozzo: luca.tagliacozzo@iff.csic.es
graphic field theories where they have first been pro-
1
5202
peS
9
]hcem-tats.tam-dnoc[
5v60741.5042:viXra

|     |     |     |     |     |     |     | 2              | Setup | and                | CFT         | predictions. |               |               |         |
| --- | --- | --- | --- | --- | --- | --- | -------------- | ----- | ------------------ | ----------- | ------------ | ------------- | ------------- | ------- |
|     |     |     |     |     |     |     | Given          | a 1D  | quantum            | many-body   |              | system        | made          | by      |
|     |     |     |     |     |     |     | L constituents |       | and                | described   | by           | a Hamiltonian |               | H,      |
|     |     |     |     |     |     |     | we             | focus | on the contraction |             | of           | a TN          | encoding      | the     |
|     |     |     |     |     |     |     | Loschmidt      |       | echo (namely       |             | the return   | probability   |               | of a    |
|     |     |     |     |     |     |     | state          | of a  | system to          | its initial | state        | after         | the           | out-of- |
|     |     |     |     |     |     |     | equilibrium    |       | evolution          | for         | a time       | T) and        | its intensive |         |
part,
logL
|e−iHT
|     |     |     |     |     |     |     |     | L=|⟨ψ | 0   | |ψ  | 0 ⟩|, | l=− | .   | (1) |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | ----- | --- | --- | --- |
TL
|     |     |     |     |     |     |     | Our     | setup       | is described    |         | in Fig.     | 1(a-c): | through      | sim-   |
| --- | --- | --- | --- | --- | --- | --- | ------- | ----------- | --------------- | ------- | ----------- | ------- | ------------ | ------ |
|     |     |     |     |     |     |     | ple     | tensor      | decompositions, |         | we can      | express | the          | evolu- |
|     |     |     |     |     |     |     | tion    | for an      | infinitesimal   | time    | step        | U(δt)   | as a         | matrix |
|     |     |     |     |     |     |     | product | operator    | (MPO)           |         | with finite | bond    | dimension    |        |
|     |     |     |     |     |     |     | (see    | e.g.        | [43, 44]).      | As a    | result,     | Eq.     | (1) is given | by     |
|     |     |     |     |     |     |     | the     | contraction | of a            | regular | 2D          | TN made | by           | L×N    |
T
|     |     |     |     |     |     |     | (N  | = T/δt) | elementary |     | four-leg | tensors, | one | for ev- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ---------- | --- | -------- | -------- | --- | ------- |
T
|     |     |     |     |     |     |     | ery      | space-time | point.     | Since        | everything |          | is translation- |          |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | ---------- | ------------ | ---------- | -------- | --------------- | -------- |
|     |     |     |     |     |     |     | ally     | invariant, | we         | can directly |            | consider | the             | thermo-  |
|     |     |     |     |     |     |     | dynamic  | limit      | by sending |              | L →        | ∞, thus  | building        | an       |
|     |     |     |     |     |     |     | infinite | strip      | with a     | width        | of N       | tensors  | in              | the time |
T
direction.
|     |     |     |     |     |     |     | We        | now | focus on | critical | dynamics |     | and     | use the |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | -------- | -------- | -------- | --- | ------- | ------- |
|     |     |     |     |     |     |     | machinery |     | of CFTs. | This     | is done  | by  | mapping | our     |
Figure1: TheLoschmidtechocanbestudiednumericallyby
defining the tensor network made by a) the initial product time evolution problem to an equivalent path inte-
|     |     |     |     |     |     |     | gral | formulation, | using | well-established |     |     | prescriptions |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | ------------ | ----- | ---------------- | --- | --- | ------------- | --- |
stateb)theMPOoftheevolutionoperatorandc)itsmatrix
element. It can also be characterized using field theories, by that allow to connect with statistical physics and
analytically continuing the results obtained by mapping the field theory (see Appendix). In analogy to those sys-
CFT on the plane to the finite geometries shown in d). The tems,therelevantobjectherewillbethetransferma-
fieldtheoryinfrareddivergenceishandledbystudyingafinite
|     |     |     |     |     |     |     | trix | [45, 46, | 47], which | generates |     | spatial | translations. |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | -------- | ---------- | --------- | --- | ------- | ------------- | --- |
spatial extent L that is then sent to infinity, while the UV In the CFT, the initial states are conformally invari-
| divergenciesarecuredbyintroducingafiniteβ |     |     |     |     | 0 closetothe |     |     |     |     |     |     |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
antboundarystates|b⟩[29,9]andtheLoschmidtecho
| conformally | invariant | boundary | states    | |b⟩.           |     |      |                |     |               |          |             |         |               |          |
| ----------- | --------- | -------- | --------- | -------------- | --- | ---- | -------------- | --- | ------------- | -------- | ----------- | ------- | ------------- | -------- |
|             |           |          |           |                |     |      | is described   |     | by the        | geometry | illustrated |         | in Fig.       | 1(d),    |
|             |           |          |           |                |     |      | representing   |     | an infinitely |          | long        | strip   | with boundary |          |
|             |           |          |           |                |     |      | conditions     |     | given by      | the      | chosen      | |b⟩.    | On the        | lattice, |
|             |           |          |           |                |     |      | we accordingly |     | evolve        | an       | initial     | product | state         | corre-   |
| posed [30,  | 31,       | 32, 33]  | but their | interpretation |     | in a |                |     |               |          |             |         |               |          |
|             |           |          |           |                |     |      | sponding       |     | to |b⟩ for    | some     | short       | amount  | of euclidean  |          |
quantum information context is still unclear. Our timeβ [7,8,9,48],leadingtoatransversesizeT+2β
|     |     |     |     |     |     |     |     | 0   |     |     |     |     |     | 0   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
results for such generalized temporal entropies in forourstrip. TheLoschmidtechoforaproductstate
| Eq. (6) | are consistent |     | with the | predictions |     | obtained |             |     |           |     |         |     |     |     |
| ------- | -------------- | --- | -------- | ----------- | --- | -------- | ----------- | --- | --------- | --- | ------- | --- | --- | --- |
|         |                |     |          |             |     |          | is obtained |     | by taking | the | limit β | →0. |     |     |
0
from holography and alternative CFT approaches Following well-established prescriptions [7, 8, 48],
| [30, 31, | 32, 33, | 34, 35, | 36], | and show | that | for our |     |        |             |     |               |     |              |     |
| -------- | ------- | ------- | ---- | -------- | ---- | ------- | --- | ------ | ----------- | --- | ------------- | --- | ------------ | --- |
|          |         |         |      |          |      |         | we  | obtain | predictions | for | the real-time |     | calculations |     |
setup they only grow logarithmically in time, as op- by performing an analytic continuation of the results
posed to the standard entanglement entropies which of CFTs in Euclidean space. We start with the stan-
| grow linearly | in  | time. | As a | result, | we can | use re- |      |         |            |      |        |       |              |     |
| ------------- | --- | ----- | ---- | ------- | ------ | ------- | ---- | ------- | ---------- | ---- | ------ | ----- | ------------ | --- |
|               |     |       |      |         |        |         | dard | results | on a strip | with | extent | β and | analytically |     |
cently devised tensor network (TN) algorithm based continuethemtothecaseβ →iT+2β [7,33,34,49].
0
on temporal matrix product states in order to cross- TheCFTpredictionsatfiniteβ areobtainedbyusing
checksuchpredictionforverylargetimesandsystems conformalmaps2: Theinfinitestripdescribedbycom-
in the thermodynamic limit. plex coordinate w related to the upper 2D plane by
v β
WeperformsuchTNsimulationsusingourrecently themapw = log(z), wherev isthesoundvelocity.
π
introducedalgorithm1 The transfer operator T produces translations
[42]ontwoexemplaryminimal
|         |           |     |                  |     |       |        | along | the | strip, and | can | be expressed |     | as an | integral |
| ------- | --------- | --- | ---------------- | --- | ----- | ------ | ----- | --- | ---------- | --- | ------------ | --- | ----- | -------- |
| models, | the Ising | and | the three-states |     | Potts | model, |       |     |            |     |              |     |       |          |
finding perfect agreement with the CFT predictions. of the stress-energy tensor of the CFT. For our strip
|     |     |     |     |     |     |     | geometry, |     | after analytic |     | continuation |     | for real | time |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | -------------- | --- | ------------ | --- | -------- | ---- |
2ExplicitCFTpredictionsfortheLoschmidtechoafterlocal
1Forsimilaralgorithmsseealso[37,38,39,40,41].
quencheshavebeenobtainedin[9].
2

and keeping only terms at first order in β /T, with on the unit circle for β → 0. This suggests that T
0 0
β ≪ T, we find that the TM describing our quench becomesunitaryinthelimitofT →∞,somethingwe
0
is given by refertoasanemergingdualunitarity ofthedynamics
in the large-T limit.
T =exp (cid:20) −i (cid:16) − κ + π L (cid:17)(cid:18) 1+2i β 0 (cid:19)(cid:21) , (2) We now focus on the computational complexity of
Tv Tv 0 T the evolution, that is dictated by the growth of the
temporal entropy. Such growth can be obtained from
wherewedroppedthehigherordertermsO(1/(Tv)2)
CFT: in particular, from the two-point correlation
vanishing in the large T limit. This equation is our
functions of twist fields we can access the Tsallis en-
bridge connecting the Loschmidt echo with the uni-
tropies of order n [57, 58, 59, 60, 61]. For a time-like
versal quantities of the underlying CFT: here κ =
separation along the imaginary axis, they correspond
−πcδt/24, L is the relevant generator of the Vira-
0 to the traces of the n-th power of the reduced transi-
soro algebra and c is the central charge [50, 51, 52,
tion matrices, defined as
53, 54, 55].
If we now define {τ i } the eigenvalues of T (with TL|R =tr h TL|R i , TL|R = |R⟩⟨L| , (5)
|τ | > |τ | > ...) and λ = log(−τ ), one can see t Nt−t ⟨L|R⟩
0 1 i i
that the intensive part l of the Loschmidt echo is dic-
where |R⟩ and ⟨L| are the right and left dominant
tatedbyλ ,thelogarithmoftheleadingeigenvalueof
0
T, as l converges to −|λ0| exponentially in the ther- eigenvectors of T. By continuing these results to
T n = 1 we obtain the results for the generalized en-
modynamic limit for a gapped T. Eq. (2) identifies
tanglemententropies[32]of|R⟩and⟨L|,whichinthe
the spectrum of T with that of L , which is diagonal
0
limit of small β becomes
in the basis of scaling fields. As a result, the CFT 0
predicts the full spectrum of T, which is given by iπc c h2T (cid:16)πt(cid:17)i
the operator content of the corresponding CFT with S gen =s 0 + 12 + 6 log π sin T , (6)
boundaries [28, 29].
Tomakecontactwiththelatticeresults,usingstan- with s 0 a constant.
dard procedures we shift the set of {λ } by two non- Notice that even though T becomes unitary, the
i
universaltermsaβ+b(β beingagainthewidthofthe leadingeigenvectorsonlyhavelogarithmicgeneralized
strip), that encode the normalization of the Hamilto- entropies. This fact is due to the CFT Hilbert space
nian and boundary effects [25, 26]. For the dominant structure,whosenumberofstates(atlowenergy)only
eigenvalue λ 0 , after analytical continuation we thus diverges polynomially with T. The small finite β 0
obtain at leading order project onto that subspace. Similar effects have been
observed in [48, 62, 63].
DrawinganalogieswiththeCFTpredictionsforthe
κ
Re(λ )=2β av+b, Im(λ )=avT − , (3) spatial entropy of grounds state of critical quantum
0 0 0 vT
chains, we can associate the logarithmic growth of
whereas for the gaps of T we find Re(λ −λ ) = 0 the generalized entropies to the closing of a gap in
1 0
and thetransfermatrixspectrum. Aswemoveawayfrom
pπx
Im(λ −λ )=− 1 , (4) the critical point, we expect this gap to open again,
1 0 vT resulting in an area law for the generalized temporal
where x is the smallest boundary critical exponent. entropies due to the finite correlation length of the
1
The leading behavior of the imaginary parts of the system: S ∼ c log(ξ), ξ being proportional to the
gen 6
eigenvalues of the TM for our Loschmidt echo is thus inverse of the gap. In turn, this property guarantees
determined by the universal quantities of the under- an even more efficient representation of the dominant
lyingCFT,namelythecentralchargeandthecritical vectors of the TM in terms of tMPS.
exponents. We can now check the above results using matrix
OurpredictionsextendtothewholespectrumofT, product states (MPS). We define a transfer-matrix
dictatingthatallitsgapsshrinktozeroasT increases, temporal MPO (tMPO) T by contracting a vertical
and are determined by the various critical exponents strip of tensors, and find its leading left and right
ofthemodel,whichareselectedbytheboundarycon- eigenvectors in the form of temporal MPS (tMPS)
ditions [29, 56]. As a result, by studying the decay of [37, 38, 39, 64, 40, 41] as shown in Fig. 2.
Loschmidt echo as a function of time, we have access
to a full set of universal properties of the underlying
3 Numerical results.
critical dynamics, including the central charge and
critical exponents.
WeconsiderboththetransversefieldIsingmodelwith
These results also unveil a connection between the
Hamiltonian
CFTsanddualunitaryevolutions: uptohigher-order
corrections, which vanish as T increases, T has full H (g)=−
Xh
σiσi+1+gσi
i
, (7)
Ising x x z
sectorsofgapsthatarepurelyimaginary,andcollapse
i
3

| Figure2:                                | ThetransversetransfermatrixT |     |     |     | involvesthecon- |     |     |     |     |     |     |     |     |     |
| --------------------------------------- | ---------------------------- | --- | --- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tractionofacolumnofelementarytensorsa). |                              |     |     |     | Itsleadingleft  |     |     |     |     |     |     |     |     |     |
andrighteigenvectorsb)allowtocharacterizetheLoschmidt
| echo (1)  | in the thermodynamic |      |     | limit. Generalized |     | temporal |            |         |                   |      |          |                    |              |       |
| --------- | -------------------- | ---- | --- | ------------------ | --- | -------- | ---------- | ------- | ----------------- | ---- | -------- | ------------------ | ------------ | ----- |
| entropies | are extracted        | from | the | reduced transition |     | matrices |            |         |                   |      |          |                    |              |       |
|           |                      |      |     |                    |     |          | Figure4:   | Toprow: | Imaginarypartsofλ |      |          | fordifferentvalues |              |       |
| defined   | in Eq. (5).          |      |     |                    |     |          |            |         |                   |      |          | 0                  |              |       |
|           |                      |      |     |                    |     |          | of β . The | points  | are               | data | from our | TN                 | calculation, | solid |
0
|     |       |     |     |     |       |     | lines are  | fits using | the   | expected | CFT     | form,  | which     | we use |
| --- | ----- | --- | --- | --- | ----- | --- | ---------- | ---------- | ----- | -------- | ------- | ------ | --------- | ------ |
|     |       |     |     |     |       |     | to extract | the        | value | of the   | central | charge | (see main | text). |
|     | Ising |     |     |     | Potts |     |            |            |       |          |         |        |           |        |
ThefitsshowanexcellentagreementwithdataforlargerT,
|          |                         |     |     |                    |     |     | as expected.  | Bottom        |               | row:    | Imaginary   | part          | of the      | first gap |
| -------- | ----------------------- | --- | --- | ------------------ | --- | --- | ------------- | ------------- | ------------- | ------- | ----------- | ------------- | ----------- | --------- |
|          |                         |     |     |                    |     |     | λ −λ          | for different |               | β and   | BC, and     | corresponding |             | fits to   |
|          |                         |     |     |                    |     |     | 1 0           |               |               | 0       |             |               |             |           |
|          |                         |     |     |                    |     |     | the CFT       | formulas      | to            | extract | x 1 . Left: | Results       | for         | the Ising |
|          |                         |     |     |                    |     |     | model, Right: |               | Potts model.  |         |             |               |             |           |
| Figure3: | Realvsimaginarypartsofτ |     |     | ,thedominanteigen- |     |     |               |               |               |         |             |               |             |           |
|          |                         |     |     | 0                  |     |     | The           | results       | for different |         | values      | of β          | 0 are shown | in        |
value of T for various values of T and β 0 for the Ising (left) Fig. 4 (top). All of them give comparable results, for
and Potts (right) models. For a fixed β , as we vary T they Ising we obtain a ≃1.27 and κ≃0.032, which from
|            |              |     |          | 0              |     |     |           |           | 0          |      |       |       |              |     |
| ---------- | ------------ | --- | -------- | -------------- | --- | --- | --------- | --------- | ---------- | ---- | ----- | ----- | ------------ | --- |
| distribute | on an almost |     | constant | radius circle. |     |     |           |           |            |      |       |       |              |     |
|            |              |     |          |                |     |     | Eq. (3)   | with      | v =v Ising | =2   | [64], | gives | c fit =0.49, | in  |
|            |              |     |          |                |     |     | excellent | agreement |            | with | c     | =1/2. | For Potts,   | the |
Ising
|             |              |     |                 |     |         |        | fits give        | κ≃0.039 | and    | using   | for   | the velocity        |            | v ≃   |
| ----------- | ------------ | --- | --------------- | --- | ------- | ------ | ---------------- | ------- | ------ | ------- | ----- | ------------------- | ---------- | ----- |
|             |              |     |                 |     |         |        | √                |         |        |         |       |                     |            | Potts |
| where       | σ x ,σ z are | the | Pauli matrices, |     | and the | three- |                  |         |        |         |       |                     |            |       |
|             |              |     |                 |     |         |        | 3 3/2            | [66] we | obtain | a value | c     | ≃0.78,              | compatible |       |
| state Potts | model        |     |                 |     |         |        |                  |         |        |         |       | fit                 |            |       |
|             |              |     |                 |     |         |        | withtheexpectedc |         |        |         | =4/5. | Asimilaranalysisfor |            |       |
Potts
|         |       | Xh(cid:16) |         |     | (cid:17) | i    |                  |     |                                 |     |     |     |     |     |
| ------- | ----- | ---------- | ------- | --- | -------- | ---- | ---------------- | --- | ------------------------------- | --- | --- | --- | --- | --- |
|         |       |            | σ† +σ†σ |     |          | +τ†) | therealpartofλ   |     | 0 (seeAppendixD)alsoconfirmsthe |     |     |     |     |     |
| H Potts | (g)=− | σ          | i       | i+1 | +g(τ     | i ,  |                  |     |                                 |     |     |     |     |     |
|         |       |            | i+1     | i   |          | i    | CFT predictions. |     |                                 |     |     |     |     |     |
i
|     |     |     |     |     |     | (8) | Fromthefirstexcitationλ |     |     |     | 1 ofT | weextractthefirst |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | --- | ----- | ----------------- | --- | --- |
with the matrices σ = P ωs|s⟩⟨s|, ω = ei2π/3 gap. Focusing again on the imaginary parts, we fit
s=0,1,2
|       | P   |           |     |       |              |     | Im(λ −λ | )   | with f | (T)=a | /T  | +a /T3, | inspired | by  |
| ----- | --- | --------- | --- | ----- | ------------ | --- | ------- | --- | ------ | ----- | --- | ------- | -------- | --- |
| and τ | =   | |s⟩⟨s+1|, |     | where | the addition | is  | 1       | 0   |        | 3     | 1   | 3       |          |     |
s=0,1,2
modulo 3. In both cases, the critical point (g = 1) Eq. (4), see Fig. 4 (bottom). The results also confirm
is described by a CFT. We consider different tMPO’s the CFT predictions [29]: For the Ising model with
lengths T, which acts as IR cutoff, investigating also fixed BC we obtain a 1 ≃ 3.155, in agreement with
thedependenceoftheresultsonβ havingtheroleof πx /v for x = 2. For free BC we get instead
|      |            |       |         | 0                |     |      | 1 Ising   |     | 1    |            |     |        |        |     |
| ---- | ---------- | ----- | ------- | ---------------- | --- | ---- | --------- | --- | ---- | ---------- | --- | ------ | ------ | --- |
|      |            |       |         |                  |     |      | a = 0.786 | ≃   | π/4, | consistent |     | with x | = 1/2. | For |
| a UV | cutoff. In | order | to make | more connections |     | with | 1         |     |      |            |     |        | 1      |     |
the CFT, we will consider both free and fixed bound- Potts,weobtaina 1 ≃0.788forfreeBCanda 1 ≃2.41
ary conditions (BC). In our formulation, the former for fixed BC state, respectively. This corresponds to
√
|             |     |       |           |           |          |     | x =v    | a   | /π ≃0.65and1.98,whichwecanmatch |     |     |     |     |     |
| ----------- | --- | ----- | --------- | --------- | -------- | --- | ------- | --- | ------------------------------- | --- | --- | --- | --- | --- |
| corresponds | to  | a |↑⟩ | state for | Ising and | a |111⟩/ | 3   | 1 Potts | 1   |                                 |     |     |     |     |     |
state for Potts, while the latter is a |+⟩ state in Ising with the expected 2/3 for the free and 2 for fixed
|           |          |         |          |              |     |          | boundary             | conditions |     | [29,        | 56]. |           |     |         |
| --------- | -------- | ------- | -------- | ------------ | --- | -------- | -------------------- | ---------- | --- | ----------- | ---- | --------- | --- | ------- |
| and |001⟩ | in Potts | [65].   | Our      | calculations |     | are per- |                      |            |     |             |      |           |     |         |
| formed    | using a  | Trotter | step of  | δt=0.1.      |     |          |                      |            |     |             |      |           |     |         |
|           |          |         |          |              |     |          | 3.2 Generalized      |            |     | entropies.  |      |           |     |         |
| 3.1       | Transfer | matrix  | spectra. |              |     |          |                      |            |     |             |      |           |     |         |
|           |          |         |          |              |     |          | The (complex-valued) |            |     | generalized |      | entropies |     | are ob- |
AswevarythelengthT ofthetMPOT withβ fixed, tained from the dominant eigenvectors of the TM.
0
its dominant eigenvalues become distributed over cir- They have an holographic meaning [30, 31, 32, 33,
cles (see Fig. 3), whose radius quickly collapses to a 34, 35, 36, 67], and provide a measure for the com-
constant as T increases, consistent with the predic- plexity of the TN contraction and thus for the cost
tionsofEq. (3). Weobtainthevalueofκinthesame for simulating the Loschmidt echo with tensor net-
equation from which we extract c by fitting the nu- works[42,37,41,39]. TheresultsinFig.5showthat
merical results for Im(λ )/T with a functional form bothforIsingandPottstheagreementwiththeCFT
0
f (T)=a +κ/T2+a /T4. predictions is excellent already for relatively short T.
| 1   | 0   |     | 4   |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
4

Figure6: Realpartofthegeneralizedtemporalentropiesfor
thedominantvectorsofthetransfermatrixoftheIsingmodel
forT =14andβ =0.2. Aswemoveawayfromthecritical
0
point g = 1, the entropies saturate, exhibiting an area law.
The left plot corresponds to fixed boundary conditions, the
right to free ones.
4 Conclusions
Figure 5: Numerical results for the real (left) and imaginary In this work, we have unveiled the universal behav-
(right)partsofthegeneralizedentropiesforthecriticalIsing ior of the Loschmidt echo after a quench to a critical
(toprow)andcriticalPotts(bottomrow)withfreeBC.Solid Hamiltonian. We show that its leading decay is dic-
lines denote the CFT predictions (6). For Ising, we use c= tated by the central charge of the underlying CFT,
1/2 central charge and s 0 =0.3 the constant providing the while finite-time corrections are governed by the crit-
best match for the real part, whereas for Potts we use the ical exponents of the theory. By leveraging temporal
central charge c=4/5 and s =0.46. The agreement with
0 matrixproductstates,wefindthatthecomputational
the CFT predictions is excellent for both real and imaginary
complexity of calculating the Loschmidt echo is con-
parts,anditimprovesaswegotolongerchains,asexpected.
trolled by the growth of temporal entropies. Using
CFT, we derive a closed-form expression for the scal-
ing of these temporal entropies as a function of time,
demonstrating a logarithmic growth. Consequently,
Neglecting the boundary effects, the imaginary part
thecomplexityofsimulatingtheLoschmidtechowith
of the entropy becomes constant and matches the ex-
MPSreducesfromexponentialtopolynomialintime.
pected Im(S ) = πc/12, and the real part is well
CFT
This result establishes that the computation of the
fitted by Eq. (6).
Loschmidt echo is feasible with classical algorithms
Different boundary conditions can affect the rate
(for alternative approaches that similarly reduce sim-
of convergence of the curves to the CFT predictions.
ulationcomplexity, see[71,72,73,74,75,76]). Thus,
ThespectrumofT isthatofaboundaryCFT,sothe
we can perform accurate classical simulations of this
operatorcontentdependsontheboundaryconditions
scenario using tensor networks, confirming our ana-
[29]. Similarly,theentanglementspectrumismapped
lytical predictions. Moreover, our numerical simula-
to a different bCFT spectrum. This boundary spec-
tions extract the key universal properties, supporting
trum gives rise to unusual finite size corrections of
theideathatthesecanalsobeprobedexperimentally
the type T−xn, which depend both on the Renyi en-
by studying the Loschmidt echo for relatively short
tropyindexnandtheboundaryspectrumintheform
times. Finally, our results reveal that the time evo-
x = x /n [68, 69]. Here, the dependence on the
n α lution operator generated by a critical Hamiltonian
boundary conditions is in the allowed values of x .
α leads, at long times, to a unitary transfer matrix in
The detailed numerical analysis of the consequences
space—an outcome expected in dual-unitary dynam-
of these observations are presented in the Appendix.
ics. Conformalsymmetrythusseemstonaturallylead
Our numerical results also validate equivalent pre-
to the emergence of dual-unitary behavior at large
dictionsforthegeneralizedentropieswhichhavebeen
times.
obtained with different techniques based on hologra-
phy and CFTs in [32, 34, 70, 49].
We finally check numerically in Fig. 6 our predic- Acknowledgments
tion of an area law for the temporal entropies as we
move away from criticality. When moving away from We would like to thank M.C. Ban˜uls, P. Calabrese,
the critical point the generalized temporal entropies E. Lopez, G. Sierra, E. Tonni for discussions re-
exhibit the same behavior at the boundaries of the lated to this project. We thank A. Tomut for
temporal chain but saturate in the middle. This gen- suggesting the reference [65] clarifying the role of
erally results in lower bond dimensions required for the lattice BC for the Potts model. LT acknowl-
the description of the dominant vectors in terms of edges support from the Proyecto Sin´ergico CAM
tMPS,openingthepossibilityforefficientsimulations Y2020/TCS-6545 NanoQuCo-CM, the CSIC Re-
of out-of-equilibrium dynamics. search Platform on Quantum Technologies PTI-001,
5

and from the Grant TED2021-130552B-C22 funded [13] Adam Nahum, Jonathan Ruhman, Sagar Vi-
by MCIN/AEI/10.13039/501100011033 and by jay, and Jeongwan Haah. “Quantum Entan-
the “European Union NextGenerationEU/PRTR”, glement Growth Under Random Unitary Dy-
and Grant PID2021-127968NB-I00 funded by namics”. Phys. Rev. X 7, 031016 (2017).
| MCIN/AEI/10.13039/501100011033. |     |     |     |     |     |     | arxiv:1608.06950.                       |           |     |         |          |       |         |      |
| ------------------------------- | --- | --- | --- | --- | --- | --- | --------------------------------------- | --------- | --- | ------- | -------- | ----- | ------- | ---- |
|                                 |     |     |     |     |     |     | [14] Bruno                              | Bertini,  |     | Pavel   | Kos, and | Tomaz | Prosen. |      |
| References                      |     |     |     |     |     |     | “ExactSpectralFormFactorinaMinimalModel |           |     |         |          |       |         |      |
|                                 |     |     |     |     |     |     | of                                      | Many-Body |     | Quantum | Chaos”.  |       | Phys.   | Rev. |
[1] J. Sˇuntajs, J. Bonˇca, T. Prosen, and L. Vid- Lett. 121, 264101 (2018). arxiv:1805.00931.
| mar.     | “Whither | many-body |     | localization?”. |         | Jour-   |            |          |     |       |          |        |         |     |
| -------- | -------- | --------- | --- | --------------- | ------- | ------- | ---------- | -------- | --- | ----- | -------- | ------ | ------- | --- |
|          |          |           |     |                 |         |         | [15] Bruno | Bertini, |     | Pavel | Kos, and | Tomaˇz | Prosen. |     |
| nal Club | for      | Condensed |     | Matter          | Physics | (2023). |            |          |     |       |          |        |         |     |
“EntanglementSpreadinginaMinimalModelof
[2] Anushya Chandran and Philip Crowley. “Con- Maximal Many-Body Quantum Chaos”. Phys.
| straining | Many-Body |     | Localization”. |     |     | Physics 17, |      |      |        |         |     |     |     |     |
| --------- | --------- | --- | -------------- | --- | --- | ----------- | ---- | ---- | ------ | ------- | --- | --- | --- | --- |
|           |           |     |                |     |     |             | Rev. | X 9, | 021033 | (2019). |     |     |     |     |
24 (2024).
|                     |           |      |          |             |     |            | [16] Bruno | Bertini,    |     | Pavel     | Kos, and     | Tomaˇz           | Prosen. |       |
| ------------------- | --------- | ---- | -------- | ----------- | --- | ---------- | ---------- | ----------- | --- | --------- | ------------ | ---------------- | ------- | ----- |
| [3] Jean-S´ebastien |           | Caux | and      | Fabian      | H.  | L. Essler. |            |             |     |           |              |                  |         |       |
|                     |           |      |          |             |     |            | “Exact     | Correlation |     | Functions |              | for Dual-Unitary |         |       |
| “Time               | Evolution |      | of Local | Observables |     | After      |            |             |     |           |              |                  |         |       |
|                     |           |      |          |             |     |            | Lattice    | Models      |     | in $1+1$  | Dimensions”. |                  |         | Phys. |
Quenching to an Integrable Model”. Phys. Rev. Rev. Lett. 123, 210601 (2019).
| Lett.                               | 110,    | 257203                     | (2013). |                   |     |          |                                        |           |        |         |                   |              |        |      |
| ----------------------------------- | ------- | -------------------------- | ------- | ----------------- | --- | -------- | -------------------------------------- | --------- | ------ | ------- | ----------------- | ------------ | ------ | ---- |
|                                     |         |                            |         |                   |     |          | [17] PieterW.ClaeysandAustenLamacraft. |           |        |         |                   |              | “Maxi- |      |
| [4] F.H.L.EsslerandA.J.J.M.deKlerk. |         |                            |         |                   |     | “Statis- |                                        |           |        |         |                   |              |        |      |
|                                     |         |                            |         |                   |     |          | mumvelocityquantumcircuits”.           |           |        |         |                   | Phys.Rev.Re- |        |      |
| ticsofmatrix                        |         | elementsoflocaloperatorsin |         |                   |     | inte-    |                                        |           |        |         |                   |              |        |      |
|                                     |         |                            |         |                   |     |          | search                                 | 2, 033032 |        | (2020). | arxiv:2003.01133. |              |        |      |
| grable                              | models” | (2023).                    |         | arxiv:2307.12410. |     |          |                                        |           |        |         |                   |              |        |      |
|                                     |         |                            |         |                   |     |          | [18] Pieter                            | W.        | Claeys | and     | Austen            | Lamacraft.   |        | “Er- |
[5] BrunoBertini,MarioCollura,JacopoDeNardis,
and Maurizio Fagotti. “Transport in Out-of- godicandnon-ergodicdual-unitaryquantumcir-
|             |     | $XXZ$      |         |       |       |             | cuits  | with  | arbitrary | local | Hilbert | space  |         | dimen- |
| ----------- | --- | ---------- | ------- | ----- | ----- | ----------- | ------ | ----- | --------- | ----- | ------- | ------ | ------- | ------ |
| Equilibrium |     |            | Chains: |       | Exact | Profiles of |        |       |           |       |         |        |         |        |
|             |     |            |         |       |       |             | sion”. | Phys. | Rev.      | Lett. | 126,    | 100603 | (2021). |        |
| Charges     | and | Currents”. |         | Phys. | Rev.  | Lett. 117,  |        |       |           |       |         |        |         |        |
arxiv:2009.03791.
| 207201 | (2016). |     |     |     |     |     |            |          |     |           |            |     |     |       |
| ------ | ------- | --- | --- | --- | --- | --- | ---------- | -------- | --- | --------- | ---------- | --- | --- | ----- |
|        |         |     |     |     |     |     | [19] Lluis | Masanes. |     | “Discrete | holography |     | in  | dual- |
[6] OlallaA.Castro-Alvaredo,BenjaminDoyon,and
Takato Yoshimura. “Emergent Hydrodynamics unitary circuits” (2023). arxiv:2301.02825.
| in Integrable |       | Quantum |      | Systems | Out     | of Equilib- |          |              |        |            |     |              |           |     |
| ------------- | ----- | ------- | ---- | ------- | ------- | ----------- | -------- | ------------ | ------ | ---------- | --- | ------------ | --------- | --- |
|               |       |         |      |         |         |             | [20] Bin | Yan,         | Lukasz | Cincio,    |     | and Wojciech |           | H.  |
| rium”.        | Phys. | Rev.    | X 6, | 041065  | (2016). |             |          |              |        |            |     |              |           |     |
|               |       |         |      |         |         |             | Zurek.   | “Information |        | Scrambling |     | and          | Loschmidt |     |
[7] Pasquale Calabrese and John Cardy. “Evolution Echo”. Phys. Rev. Lett. 124, 160603 (2020).
| of entanglement |          | entropy |       | in one-dimensional |     | sys-    | arxiv:1903.02651. |          |            |     |      |        |     |     |
| --------------- | -------- | ------- | ----- | ------------------ | --- | ------- | ----------------- | -------- | ---------- | --- | ---- | ------ | --- | --- |
| tems”.          | J. Stat. | Mech.   | 2005, | P04010             |     | (2005). |                   |          |            |     |      |        |     |     |
|                 |          |         |       |                    |     |         | [21] B.           | Pozsgay. | “Dynamical |     | free | energy | and | the |
[8] Pasquale Calabrese and John Cardy. “Entangle- Loschmidt-echo for a class of quantum quenches
| ment | and correlation |     | functions |     | following | a local |     |                |     |      |         |         |     |         |
| ---- | --------------- | --- | --------- | --- | --------- | ------- | --- | -------------- | --- | ---- | ------- | ------- | --- | ------- |
|      |                 |     |           |     |           |         | in  | the Heisenberg |     | spin | chain”. | Journal | of  | Statis- |
quench: A conformal field theory approach”. J. tical Mechanics: Theory and Experiment 2013,
| Stat.          | Mech. | 2007,    | P10004 | (2007).  |         |      |         |            |     |                  |     |            |     |       |
| -------------- | ----- | -------- | ------ | -------- | ------- | ---- | ------- | ---------- | --- | ---------------- | --- | ---------- | --- | ----- |
|                |       |          |        |          |         |      | P10028  | (2013).    |     | arXiv:1308.3087. |     |            |     |       |
| [9] Jean-Marie |       | St´ephan | and    | J´erˆome | Dubail. | “Lo- |         |            |     |                  |     |            |     |       |
|                |       |          |        |          |         |      | [22] F. | Andraschko | and | J. Sirker.       |     | “Dynamical |     | quan- |
calquantumquenchesincriticalone-dimensional
|          |            |               |     |          |           |       | tum                       | phase  | transitions |                  | and the | Loschmidt       |     | echo: |
| -------- | ---------- | ------------- | --- | -------- | --------- | ----- | ------------------------- | ------ | ----------- | ---------------- | ------- | --------------- | --- | ----- |
| systems: |            | Entanglement, |     | the      | Loschmidt | echo, |                           |        |             |                  |         |                 |     |       |
|          |            |               |     |          |           |       | Atransfermatrixapproach”. |        |             |                  |         | PhysicalReviewB |     |       |
| and      | light-cone | effects”.     |     | J. Stat. | Mech.     | 2011, |                           |        |             |                  |         |                 |     |       |
|          |            |               |     |          |           |       | 89,                       | 125120 | (2014).     | arXiv:1312.4165. |         |                 |     |       |
| P08019   | (2011).    |               |     |          |           |       |                           |        |             |                  |         |                 |     |       |
[23] LorenzoPiroli,Bal´azsPozsgay,andEricVernier.
| [10] John    | Cardy | and                | Erik | Tonni. | “Entanglement |       |       |     |         |     |          |        |     |        |
| ------------ | ----- | ------------------ | ---- | ------ | ------------- | ----- | ----- | --- | ------- | --- | -------- | ------ | --- | ------ |
|              |       |                    |      |        |               |       | “From | the | Quantum |     | Transfer | Matrix |     | to the |
| Hamiltonians |       | in two-dimensional |      |        | conformal     | field |       |     |         |     |          |        |     |        |
$XXZ$
theory”. J. Stat. Mech.: Theory Exp. 2016, Quench Action: The Loschmidt echo in
|                                                |         |              |           |          |      |           | Heisenberg  |            | spin              | chains”. | Journal |              | of Statisti- |       |
| ---------------------------------------------- | ------- | ------------ | --------- | -------- | ---- | --------- | ----------- | ---------- | ----------------- | -------- | ------- | ------------ | ------------ | ----- |
| 123103                                         | (2016). |              |           |          |      |           |             |            |                   |          |         |              |              |       |
|                                                |         |              |           |          |      |           | cal         | Mechanics: |                   | Theory   | and     | Experiment   |              | 2017, |
| [11] J.Dubail.“Entanglementscalingofoperators: |         |              |           |          |      | A         |             |            |                   |          |         |              |              |       |
|                                                |         |              |           |          |      |           | 023106      | (2017).    | arXiv:1611.06126. |          |         |              |              |       |
| conformal                                      |         | field theory | approach, |          | with | a glimpse |             |            |                   |          |         |              |              |       |
|                                                |         |              |           |          |      |           | [24] Arseni | Goussev,   |                   | Rodolfo  |         | A. Jalabert, |              | Ho-   |
| of simulability                                |         | of           | long-time | dynamics |      | in 1+1d”. |             |            |                   |          |         |              |              |       |
J. Phys. A: Math. Theor. 50, 234001 (2017). racio M. Pastawski, and Diego Wisniacki.
“LoschmidtEcho”.Scholarpedia7,11687(2012).
arxiv:1612.08630.
arxiv:1206.6348.
| [12] Jacopo | Surace, |     | Luca | Tagliacozzo, |     | and Erik |     |     |     |     |     |     |     |     |
| ----------- | ------- | --- | ---- | ------------ | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
Tonni. “Operator content of entanglement spec- [25] J. L. Cardy. “Conformal invariance and univer-
trainthetransversefieldIsingchainafterglobal sality in finite-size scaling”. J. Phys. A: Math.
quenches”. Phys. Rev. B 101, 241107 (2020). Gen. 17, L385–L387 (1984).
6

[26] Ian Affleck. “Universal term in the free energy [41] Alessio Lerose, Michael Sonner, and Dmitry A.
at a critical point and the conformal anomaly”. Abanin. “Overcoming the entanglement barrier
Phys. Rev. Lett. 56, 746–748 (1986). inquantummany-bodydynamicsviaspace-time
|           |        |     |      |            |     |         | duality”. | Phys. | Rev. B 107, | L060305 |     | (2023). |
| --------- | ------ | --- | ---- | ---------- | --- | ------- | --------- | ----- | ----------- | ------- | --- | ------- |
| [27] John | Cardy. |     | “The | ubiquitous | ‘c  | ’: From |           |       |             |         |     |         |
the Stefan–Boltzmann law to quantum informa- [42] StefanoCarignano,CarlosRamosMarim´on, and
tion*”. JournalofStatisticalMechanics: Theory LucaTagliacozzo. “Ontemporalentropyandthe
and Experiment 2010, P10004 (2010). complexityofcomputingtheexpectationvalueof
|           |        |     |           |     |         |         | localoperatorsafteraquench”. |     |     |     | PhysicalReview |     |
| --------- | ------ | --- | --------- | --- | ------- | ------- | ---------------------------- | --- | --- | --- | -------------- | --- |
| [28] John | Cardy. |     | “Operator |     | content | of two- |                              |     |     |     |                |     |
dimensional conformally invariant theories”. Research 6, 033021 (2024). arXiv:2307.11649.
Nucl. Phys. B 270, 186–204 (1986). [43] Ian P McCulloch. “From density-matrix renor-
|           |              |        |         |                    |     |            | malization  | group | to matrix     | product |         | states”. J. |
| --------- | ------------ | ------ | ------- | ------------------ | --- | ---------- | ----------- | ----- | ------------- | ------- | ------- | ----------- |
| [29] John | L.           | Cardy. | “Effect | of boundary        |     | conditions |             |       |               |         |         |             |
|           |              |        |         |                    |     |            | Stat. Mech. | 2007, | P10014–P10014 |         | (2007). |             |
| on        | the operator |        | content | of two-dimensional |     | con-       |             |       |               |         |         |             |
formally invariant theories”. Nuclear Physics B [44] B Pirvu, V Murg, J I Cirac, and F Verstraete.
275, 200–218 (1986). “Matrixproductoperatorrepresentations”. New
|     |     |     |     |     |     |     | J. Phys. | 12, 025012 | (2010). |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | ------- | --- | --- | --- |
[30] K.Narayan.“DeSitterextremalsurfaces”.Phys.
Rev. D 91, 126011 (2015). arxiv:1501.03019. [45] XiaoqunWangandTaoXiang. “Transfer-matrix
[31] K. Narayan. “De Sitter space and extremal sur- density-matrix renormalization-group theory for
|       |     |           |         |         |     |           | thermodynamics |     | of one-dimensional |     |     | quantum |
| ----- | --- | --------- | ------- | ------- | --- | --------- | -------------- | --- | ------------------ | --- | --- | ------- |
| faces | for | spheres”. | Physics | Letters | B   | 753, 308– |                |     |                    |     |     |         |
314 (2016). arxiv:1504.07430. systems”. Physical Review B 56, 5061–
5064 (1997).
| [32] Yoshifumi |     | Nakata, | Tadashi | Takayanagi, |     | Yusuke |     |     |     |     |     |     |
| -------------- | --- | ------- | ------- | ----------- | --- | ------ | --- | --- | --- | --- | --- | --- |
Taki, Kotaro Tamaoka, and Zixia Wei. “Holo- [46] J. Sirker and A. Klu¨mper. “Real-time dynamics
atfinitetemperaturebyDMRG:Apath-integral
| graphic | Pseudo |     | Entropy”. | Phys. | Rev. | D 103, |     |     |     |     |     |     |
| ------- | ------ | --- | --------- | ----- | ---- | ------ | --- | --- | --- | --- | --- | --- |
026005 (2021). arxiv:2005.13801. approach” (2005). arXiv:cond-mat/0504091.
[33] Kazuki Doi, Jonathan Harper, Ali Mollabashi, [47] J.Sirker.“Entanglementmeasuresandthequan-
|     |     |     |     |     |     |     | tum to | classical | mapping”. | Journal |     | of Statisti- |
| --- | --- | --- | --- | --- | --- | --- | ------ | --------- | --------- | ------- | --- | ------------ |
TadashiTakayanagi,andYusukeTaki.“Timelike
entanglemententropy”(2023). arxiv:2302.11695. cal Mechanics: Theory and Experiment 2012,
|             |      |          |     |         |     |             | P12012 | (2012). | arXiv:1206.4829. |     |     |     |
| ----------- | ---- | -------- | --- | ------- | --- | ----------- | ------ | ------- | ---------------- | --- | --- | --- |
| [34] Kazuki | Doi, | Jonathan |     | Harper, | Ali | Mollabashi, |        |         |                  |     |     |     |
Tadashi Takayanagi, and Yusuke Taki. “Pseudo [48] Niall F. Robertson, Jacopo Surace, and Luca
|     |     |     |     |     |     |     | Tagliacozzo. | “On | quenches | to  | the critical | point |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | -------- | --- | ------------ | ----- |
EntropyindS/CFTandTime-likeEntanglement
Entropy”. Phys. Rev. Lett. 130, 031601 (2023). ofthethreestatesPottsmodel–MatrixProduct
|     |     |     |     |     |     |     | State simulations |     | and CFT”. | Phys. | Rev. | B 105, |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --------- | ----- | ---- | ------ |
arxiv:2210.09457.
|         |         |     |        |     |        |           | 195103 (2022). |     | arxiv:2110.07078. |     |     |     |
| ------- | ------- | --- | ------ | --- | ------ | --------- | -------------- | --- | ----------------- | --- | --- | --- |
| [35] K. | Narayan | and | Hitesh | K.  | Saini. | “Notes on |                |     |                   |     |     |     |
time entanglement and pseudo-entropy” (2023). [49] Wu-zhong Guo, Song He, and Yu-Xuan Zhang.
arxiv:2303.01307. “Relation between timelike and spacelike entan-
[36] Ze Li, Zi-Qing Xiao, and Run-Qiu Yang. glement entropy” (2024). arxiv:2402.00268.
“On holographic time-like entanglement en- [50] A. A. Belavin, A. M. Polyakov, and A. B.
tropy”. J. High Energ. Phys. 2023, 4 (2023). Zamolodchikov. “Infiniteconformalsymmetryin
|     |     |     |     |     |     |     | two-dimensional |     | quantum | field | theory”. | Nuclear |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ------- | ----- | -------- | ------- |
arxiv:2211.14883.
|     |     |     |     |     |     |     | Physics | B 241, | 333–380 | (1984). |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------ | ------- | ------- | --- | --- |
[37] M.C.Ban˜uls,M.B.Hastings,F.Verstraete,and
J. I. Cirac. “Matrix Product States for Dynam- [51] John L. Cardy. “CONFORMAL INVARIANCE
ical Simulation of Infinite Chains”. Phys. Rev. AND STATISTICAL MECHANICS”. In Les
Lett. 102, 240603 (2009). Houches Summer School in Theoretical Physics:
|                |     |                 |     |     |         |            | Fields, | Strings, | Critical | Phenomena. |     | (1989). |
| -------------- | --- | --------------- | --- | --- | ------- | ---------- | ------- | -------- | -------- | ---------- | --- | ------- |
| [38] Alexander |     | Mu¨ller-Hermes, |     | J.  | Ignacio | Cirac, and |         |          |          |            |     |         |
url: https://www-thphys.physics.ox.ac.uk/
| Mari | Carmen |     | Ban˜uls. | “Tensor | network | tech- |     |     |     |     |     |     |
| ---- | ------ | --- | -------- | ------- | ------- | ----- | --- | --- | --- | --- | --- | --- |
people/JohnCardy/lh.pdf.
| niques | for | the computation |     | of  | dynamical | observ- |     |     |     |     |     |     |
| ------ | --- | --------------- | --- | --- | --------- | ------- | --- | --- | --- | --- | --- | --- |
ablesin1Dquantumspinsystems”.NewJ.Phys. [52] Paul Ginsparg. “Applied Conformal Field The-
14, 075003 (2012). arxiv:1204.5080. ory” (1988). arxiv:hep-th/9108028.
[39] M. B. Hastings and R. Mahajan. “Connect- [53] Philippe Francesco, Pierre Mathieu, and David
ing Entanglement in Time and Space: Improv- S´en´echal. “Conformal Field Theory”. Graduate
ing the Folding Algorithm”. Phys. Rev. A 91, TextsinContemporaryPhysics.Springer-Verlag.
| 032306 | (2015). |     | arxiv:1411.7950. |     |     |     | New York | (1997). |     |     |     |     |
| ------ | ------- | --- | ---------------- | --- | --- | --- | -------- | ------- | --- | --- | --- | --- |
[40] Alessio Lerose, Michael Sonner, and Dmitry A. [54] Malte Henkel. “Conformal Invariance and Crit-
Abanin. “Influence Matrix Approach to Many- ical Phenomena”. Theoretical and Mathe-
Body Floquet Dynamics”. Phys. Rev. X 11, matical Physics. Springer-Verlag. Berlin Heidel-
| 021040 | (2021). |     |     |     |     |     | berg (1999). |     |     |     |     |     |
| ------ | ------- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- |
7

[55] John Cardy. “Conformal Field Theory and [70] Wu-zhong Guo, Song He, and Yu-Xuan Zhang.
Statistical Mechanics”. 0807.3472 (2008). “Onthereal-timeevolutionofpseudo-entropyin
arxiv:0807.3472. 2dCFTs”.J.HighEnerg.Phys.2022,94(2022).
arxiv:2206.11818.
[56] Ian Affleck, Masaki Oshikawa, and Hubert
Saleur. “Boundary critical phenomena in the [71] J. Surace, M. Piani, and L. Tagliacozzo. “Sim-
three-statePottsmodel”. J.Phys.A:Math.Gen. ulating the out-of-equilibrium dynamics of lo-
31, 5827 (1998). calobservablesbytradingentanglementformix-
[57] MarkSrednicki. “Entropyandarea”. Phys.Rev. ture”. Phys. Rev. B 99, 235115 (2019).
Lett. 71, 666 (1993).
[72] A. Strathearn, P. Kirton, D. Kilda, J. Keeling,
[58] Curtin Callan and Frank Wilczek. “On Geomet- and B. W. Lovett. “Efficient non-Markovian
ric Entropy” (1994). arxiv:hep-th/9401072. quantum dynamics using time-evolving ma-
[59] G. Vidal, J. I. Latorre, E. Rico, and A. Ki- trix product operators”. Nat Commun 9,
taev. “Entanglement in Quantum Critical Phe- 3322 (2018).
nomena”. Phys. Rev. Lett. 90, 227902 (2003).
[73] ChristopherDavidWhite,MichaelZaletel,Roger
[60] Pasquale Calabrese and John Cardy. “Entangle- S. K. Mong, and Gil Refael. “Quantum dynam-
mententropyandquantumfieldtheory”. J.Stat. ics of thermalizing systems”. Phys. Rev. B 97,
Mech. 2004, P06002 (2004). 035127 (2018). arxiv:1707.01506.
[61] Michele Caraglio and Ferdinando Gliozzi. “En-
[74] Tibor Rakovszky, C. W. von Keyserlingk, and
tanglement Entropy and Twist Fields”. J.
Frank Pollmann. “Dissipation-assisted operator
High Energy Phys. 2008, 076–076 (2008).
evolution method for capturing hydrodynamic
arxiv:0808.4094.
transport” (2020). arxiv:2004.05177.
[62] M. Grundner, P. Westhoff, F. B. Kugler,
[75] Miguel Fr´ıas-P´erez and Mari Carmen Ban˜uls.
O. Parcollet, and U. Schollw¨ock. “Complex
“Light cone tensor network and time evolu-
Time Evolution in Tensor Networks” (2023).
tion”. Phys. Rev. B 106, 115117 (2022).
arxiv:2312.11705.
arxiv:2201.08402.
[63] Xiaodong Cao, Yi Lu, E. Miles Stoudenmire,
[76] Wen-Yuan Liu, Si-Jing Du, Ruojing Peng, John-
and Olivier Parcollet. “Dynamical correlation
nie Gray, and Garnet Kin-Lic Chan. “Ten-
functions from complex time evolution” (2024).
sor Network Computations That Capture Strict
arxiv:2311.10909.
Variationality, Volume Law Behavior, and the
[64] Emanuele Tirrito, Neil J. Robinson, Maciej
Efficient Representation of Neural Network
Lewenstein, Shi-Ju Ran, and Luca Taglia-
States” (2024). arxiv:2405.03797.
cozzo. “Characterizing the quantum field the-
ory vacuum using temporal Matrix Product [77] JohnB.Kogut.“Anintroductiontolatticegauge
states” (2022). arxiv:1810.08050. theory and spin systems”. Reviews of Modern
Physics 51, 659–713 (1979).
[65] Wei Tang, Lei Chen, Wei Li, X. C. Xie, Hong-
Hao Tu, and Lei Wang. “Universal Boundary [78] Kouichi Okunishi, Tomotoshi Nishino, and Hi-
Entropies in Conformal Field Theory: A Quan- roshi Ueda. “Developments in the Tensor Net-
tumMonteCarloStudy”. PhysicalReviewB96, work – from Statistical Mechanics to Quantum
115136 (2017). arxiv:1708.04022. Entanglement” (2022). arXiv:2111.12223.
[66] Alexander A. Eberharter, Laurens Vander-
[79] StefanoCarignano.“Theitransverse.jllibraryfor
straeten, Frank Verstraete, and Andreas M.
transverse tensor network contractions” (2025).
L¨auchli. “Extracting the Speed of Light from
arXiv:2509.03699.
Matrix Product States”. Phys. Rev. Lett. 131,
226502 (2023). [80] Matthew Fishman, Steven R. White, and
E. Miles Stoudenmire. “The ITensor Software
[67] Kotaro Shinmyo, Tadashi Takayanagi, and
Library for Tensor Network Calculations”. Sci-
Kenya Tasuki. “Pseudo entropy under joining
Post Phys. CodebasesPage 4 (2022).
local quenches” (2023). arxiv:2310.12542.
[68] John Cardy and Pasquale Calabrese. “Unusual [81] Wei Tang, Frank Verstraete, and Jutho Haege-
corrections to scaling in entanglement entropy”. man. “Matrix Product State Fixed Points
JournalofStatisticalMechanics: TheoryandEx- of Non-Hermitian Transfer Matrices” (2023).
periment 2010, P04023 (2010). arxiv:2311.18733.
[69] Gil Young Cho, Andreas W. W. Ludwig, and [82] Vincenzo Alba, Luca Tagliacozzo, and Pasquale
ShinseiRyu. “Universalentanglementspectraof Calabrese.“Entanglemententropyoftwodisjoint
gapped one-dimensional field theories”. Physical blocks in critical Ising models”. Physical Review
Review B 95, 115122 (2017). B 81, 60411 (2010).
8

[83] Vincenzo Alba, Luca Tagliacozzo, and Pasquale Tonni. “Entanglement negativity in the critical
Calabrese. “Entanglement entropy of two dis- Isingchain”.J.Stat.Mech.2013,P05002(2013).
joint intervals in c = 1 theories”. Journal of Sta- [85] AndreaCoser,LucaTagliacozzo,andErikTonni.
| tistical | Mechanics: | Theory |     | and Experiment | 06, |     |     |        |           |                       |         |
| -------- | ---------- | ------ | --- | -------------- | --- | --- | --- | ------ | --------- | --------------------- | ------- |
|          |            |        |     |                |     |     | “On | R´enyi | entropies | of disjoint intervals | in con- |
012 (2011).
|               |            |        |              |          |          |     | formal | field   | theory”. | J. Stat. Mech. | 2014, |
| ------------- | ---------- | ------ | ------------ | -------- | -------- | --- | ------ | ------- | -------- | -------------- | ----- |
|               |            |        |              |          |          |     | P01008 | (2014). |          |                |       |
| [84] Pasquale | Calabrese, | Luca   | Tagliacozzo, |          | and Erik |     |        |         |          |                |       |
| A Mapping     | the        | quench |              | geometry | to       | CFT |        |         |          |                |       |
At the core of this work lies the mapping of the amplitude associated with the return probability to the initial
state after the evolution for a time T after a quench to the critical point of a lattice Hamiltonian with the path
integral of a field theory on a specific geometry which can be studied using conformal field theory.
ThismappinggreatlysimplifiesgiventhattheHamiltonianisdefinedatacriticalpointwhere,duetouniver-
sality, strong fluctuations are significant, though their microscopic details are irrelevant in the renormalization
group sense. By employing a formalism that links thermal and quantum fluctuations in statistical mechanics
and quantum field theory, we can translate these analogies into computational tools for predicting physical
phenomena.
The idea is that the partition function of a statistical system with short-range interactions can then be
seen equivalently as either a sum over classical variables in a d-dimensional euclidean space with a classical
e−τHˆ({ϕ})
Hamiltonian H(s ), or as the trace of a time evolution operator U(τ) = associated with a quantum
i
Hamiltonian Hˆ in d−1 dimensions of certain appropriate variables ϕ. Furthermore, for our problem we can
equivalently adopt either an operator or a functional approach, which allows to substitute the sum over the
classical discrete variables s in terms of a path integral on continuous variables ϕ(x), weighted by the classical
i
| action S. | We thus have |     |     |     |     |     |     |     |     |     |     |
| --------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Z
|     |     |     | X   | e−H(si) |     | Y   | e−τHˆ(ϕ) |     | Dϕe−S[ϕ], |     |     |
| --- | --- | --- | --- | ------- | --- | --- | -------- | --- | --------- | --- | --- |
|     |     |     | Z = |         | =   | Tr  |          | ≈   |           |     |     |
ϕ
|     |     |     | {s}  |      |         | τ    |      |     |     |     |     |
| --- | --- | --- | ---- | ---- | ------- | ---- | ---- | --- | --- | --- | --- |
|     |     |     | stat | mech | quantum | time | evol |     | QFT |     |     |
andwecaninterpretoursetupeitherasaTrotterizedtimeevolutioninanappropriatebasisorafieldtheory
discretizedonalatticewithaproperlydefinedcontinuumlimit. Suchmappingisparticularlysimpleatacritical
point, when the correlation length is much larger than the lattice spacing and using universality arguments we
| can focus | on very simple | lattice | models. |     |     |     |     |     |     |     |     |
| --------- | -------------- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
In the following, we recall some of the ideas revolving around the equivalence of these formalisms, referring
theinterestedreadertothestandardliterature[77,51,53,54,55]foradditionaldetails(seealso[78]foramore
historical overview related to the development of tensor network algorithms).
|     |     |     |     |     |     |     |     |     |     | |e−iHˆT | Hˆ  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- |
Our starting point for the computation of the Loschmidt echo is the amplitude ⟨ψ |ψ ⟩, where
|     |     |     |     |     |     |     |     |     |     | 0 0 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
is the quantum Hamiltonian of our system. In the Feynman path integral formalism, this amplitude can be
mapped to a functional integral over the action S. The derivation for a zero-dimensional problem (ie. a single
particle) is textbook material: one begins by splitting the time evolution operator U(T) = e−iHˆT in smaller
intervals T = N δT, an operation which will allow to work with a Trotter approximation for non-commuting
T
terms in the exponential. More specifically, taking a generic nonrelativistic Hamiltonian Hˆ =Kˆ +Vˆ, Kˆ being
Vˆ
the kinetic term and the potential, we consider an infinitesimal time evolution operator sandwiched between
| two position | eigenstates                     |     |     |     |                 |     |     |     |           |         |     |
| ------------ | ------------------------------- | --- | --- | --- | --------------- | --- | --- | --- | --------- | ------- | --- |
|              |                                 |     |     |     |                 |     |     | n   | hm(x−x′)2 | io      |     |
|              | ⟨x′|U(δT)|x⟩=⟨x′|e−iKˆδTe−iVˆδT |     |     |     | +O(δT)2)|x⟩∼exp |     |     |     |           |         |     |
|              |                                 |     |     |     |                 |     |     | iδT |           | −V(x) , | (9) |
|              |                                 |     |     |     |                 |     |     |     | 2 δT2     |         |     |
whereweinsertedacompletesetofmomentumeigenstatesandperformedthemomentumintegration,neglecting
termsoforderδT2. WecanthenidentifytheresultastheexponentialoftheactionS forthisinfinitesimalstep
from x to x′, and putting together the various amplitudes we arrive at ⟨x |U(T)|x ⟩ = R DxeiS(x), where the
|     |     |     |     |     |     |     |     |     | f   | i   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
R
integral Dx is meant as a sum over all possible intermediate positions of the particle, that is, a sum over all
paths.
At this point, it is convenient to perform the standard analytic continuation to imaginary time t → −iτ,
τ ∈ [0,β], which also allows to make a connection with statistical mechanics. In particular, the path integral
expressionforthetransitionamplitudebecomesequivalenttoaclassicalpartitionfunctionforaone-dimensional
system, whose sites are labelled by the different time steps of the time evolution. The (now euclidean) action
9

describesthecouplingofthepositionvariablesviathediscretizedderivative(cf.Eq.(9))fordifferenttimesteps.
In turn, this implies that we can interpret the infinitesimal propagator ⟨x′|U(δτ)|x⟩ as an element T(x′,x) of
thetransfermatrixevolvingthesystemfromthepositionxtox′ atthenexttimestep,andthewholeamplitude
can be written as
Z Z
Z = Y dx T(x ,x )= ⟨x |TˆNT|x ⟩ , (10)
i i+1 i f i
i
sothatthetransfermatrixcontains all the informationabouttheevolutionofthesystem. Wecanthenusethe
transfermatrixbuiltuponthetimeevolutionoperatorasabridgebetweenthepathintegraltotheHamiltonian
formalism,aconnectionwhichbecomesclearinthelimitofinfinitesimaltimesteps,whereonehasTˆ ≈1−iδτHˆ.
Thesametypeofconsiderationscanbenaturallyextendedtohigher-dimensionalsystems,simplyinterpreting
the transfer matrix as a linear operator that builds the partition function of a d-dimensional statistical lattice
model with short-range interactions by relating the states of two adjacent (d−1)-dimensional slices.
More specifically, for the dynamics of a one-dimensional spin chain such as the ones we consider in this work,
wecanmapourreturnamplitudeafterimaginarytimerotationtoatwo-dimensionalclassicalstatisticalmodel.
Since one of the two dimensions will be associated with euclidean time, we can define the classical model on an
anisotropic grid with different couplings in the two directions, in order to allow for a separate treatment of the
twodirections. Wenowconsidertworowsofspinsalignedalongthespatialdirectionanddefineageneralization
of the transfer matrix from above, which now connects the two rows, and has a number of constituents which
grows exponentially with the system size. This transfer matrix again provides the link between the formalisms,
as it can both be used to construct the full 2d partition function and to connect to the underlying quantum
Hamiltonian in the limit of vanishing timesteps, after appropriately correcting the statistical model couplings
to describe the same physics in such limit.
Ananalogousmappingforthissetupcanbedonebetweenthetwo-dimensionalstatisticalmechanicsproblem
and quantum field theory in 1+1 (time+space) dimensions after imaginary time rotation. Here the analogy
is even more transparent, as quantum field theory naturally incorporates time, and if we regularize its path
integral on a grid and dissect it along timelike slices, we end up with the same structure as the one discussed
above, allowing the connection with the Hamiltonian formalism.
At a critical point the continuum limit is well-defined, allowing us to disregard microscopic details and focus
on the long-wavelength properties of the time-evolved one-dimensional quantum system or the equivalent 2d
classical statistical system. These properties can be characterized using a corresponding quantum field theory
in the same universality class. Importantly, at criticality, this field theory is massless and conformal, enabling
us to utilize the extensive tools developed for characterizing CFTs in 1+1 dimensions.
As a result, we can obtain from CFT predictions for the quantities of interest, the transition matrix T and
the Loschmidt echo L. The latter can thus be seen as equivalent to a partition function, and its intensive part
l is finite and plays the role of a free energy in statistical mechanics.
Wefinallynotethatalltheseequivalencesariseinanaturalwayinthepictorialdescriptionoftensornetworks:
Fig. 1(c)inthemaintextcanbeseenequivalentlyasthepartitionfunctionofatwo-dimensionallatticesystem,
asaTrotterizedtimeevolutionofaone-dimensionalchain,aswellasafunctionalintegralwhichisUV-regularized
by discretization on a lattice.
B CFT Predictions for the Transfer Matrix
Having shown how the return amplitude for a one-dimensional chain can be mapped to a path integral in
1+1-dimensional CFT, let us now recap some of the main ideas behind the derivations for the transfer matrix
spectrum which we employ in our manuscript. For more details, we refer to the reviews [51, 52, 55].
Aconformalfieldtheory(CFT)isafieldtheorywhoseactionhasaspecialsymmetrysinceitisinvariantunder
conformal transformations. Such a symmetry is typically expected to be a consequence of scale invariance and
locality of the Hamiltonian. As a result, the physics of many critical points of lattice models can be described
at large distances by a CFT. From the action S of the field theory, one can define the stress energy tensor T
µν
as the response of the action to an infinitesimal change of coordinates, rµ →rµ+αµ(r),
1 Z
δS =− (T (r)∂ναµ(r))dr, (11)
2π µν
and as such the stress tensor can be viewed as the generator of scale and conformal transformations. In 2D the
conformal algebra becomes infinite dimensional and the CFT is highly constrained. In particular, any analytic
change of coordinates becomes a symmetry of the theory, and thus rather than thinking of the CFT as defined
on a 2D space-time, one usually thinks of it as defined on the complex plane. It is worth noting already at this
10

point that in Euclidean CFT the two coordinates play the same role, an aspect which will become important
for us when we later focus on a transverse transfer matrix. If we consider the field coordinates z ∈ C and its
conjugatez¯, wecanmapthetheoryfromtheplanetoothergeometriesviaanyanalyticchangesofcoordinates.
Scale invariance is enough to understand that correlation functions in a CFT decay algebraically, e.g. for a
| two-point | function | on  | the infinite | plane | we  | have |     |     |     |     |     |     |
| --------- | -------- | --- | ------------ | ----- | --- | ---- | --- | --- | --- | --- | --- | --- |
1
|     |     |     |     | ⟨ϕ(z | )ϕ(z | )⟩= |         |          |         | ,    |     | (12) |
| --- | --- | --- | --- | ---- | ---- | --- | ------- | -------- | ------- | ---- | --- | ---- |
|     |     |     |     |      | 1    | 2   |         |          |         | )2h¯ |     |      |
|     |     |     |     |      |      |     | (z 1 −z | 2 )2h(z¯ | 1 −z¯ 2 |      |     |      |
{h,h¯}
dictated by a set of critical exponents that define the universality class of the model.
Conformal symmetry now dictates that, under a coordinate change ω =f(z),
|     |     |     |      |      |     |       |          | ⟨ϕ(z )ϕ(z | )⟩         |     |      |      |
| --- | --- | --- | ---- | ---- | --- | ----- | -------- | --------- | ---------- | --- | ---- | ---- |
|     |     |     | ⟨ϕ(ω | )ϕ(ω | )⟩= |       |          | 1         | 2          |     | ,    | (13) |
|     |     |     |      | 1    | 2   |       |          |           | ))h¯(f′(z¯ |     | ))h¯ |      |
|     |     |     |      |      |     | (f′(z | ))h(f′(z | ))h(f′(z¯ |            |     |      |      |
|     |     |     |      |      |     |       | 1        | 2         | 1          |     | 2    |      |
and we now just need to express the old coordinates in terms of the new ones.
For example, if we want to describe the physics on a cylinder rather than a plane, as in the case of thermal
statesorgroundstatesonfiniterings,wecandefineω = β log(z),whereω =s+iuisnowdefinedonthedesired
2π
cylinder with circumference β along the imaginary axis. We then have f′(z) = β and z(ω) = exp(2πω/β),
2πz
| and the | two point | function | on  | the cylinder |     | becomes |     |     |     |     |     |     |
| ------- | --------- | -------- | --- | ------------ | --- | ------- | --- | --- | --- | --- | --- | --- |
(π/β)2x
|          |          |         | ⟨ϕ(ω | )ϕ(ω | )⟩=  |          |      |            |          |       | ,           | (14) |
| -------- | -------- | ------- | ---- | ---- | ---- | -------- | ---- | ---------- | -------- | ----- | ----------- | ---- |
|          |          |         |      | 1 2  |      | (cid:16) |      | (cid:17)2h | (cid:16) |       | (cid:17)2h¯ |      |
|          |          |         |      |      |      | π(ω      |      |            | π(ω¯     |       |             |      |
|          |          |         |      |      | sinh |          | 1 −ω | 2 ) sinh   |          | 1 −ω¯ | 2 )         |      |
|          |          |         |      |      |      | β        |      |            | β        |       |             |      |
| with the | exponent | x=h+h¯. |      |      |      |          |      |            |          |       |             |      |
Conformal symmetry also dictates how the stress energy tensor changes under a conformal map,
c
|     |     |     |     | T(ω)→T(z)=f′(ω(z))2T(z)+ |     |     |     |     |     | {z,ω}, |     | (15) |
| --- | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- | ------ | --- | ---- |
12
| with {z,ω} | the | Schwartzian | derivative |     | of f. |     |     |     |     |     |     |     |
| ---------- | --- | ----------- | ---------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
In the traditional Hilbert space formalism, the quantum Hamiltonian Hˆ is related to the integral of the
T
time-time component of the stress-energy tensor µν over the space-like curve on which one quantizes the
theory3.
It is natural to associate the imaginary axis of the complex coordinates with the temporal direction. We can
then write
1 Z
|     |     |     |     |     |     | Hˆ  |     | dsT     |     |     |     |      |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | ---- |
|     |     |     |     |     |     | =   |     | uu (s). |     |     |     | (16) |
2π
As seen in the previous section, the quantum Hamiltonian is intimately related to the transfer matrix, which is
| the object | we are | interested | in  | for characterizing |     |     | our system. |     |     |     |     |     |
| ---------- | ------ | ---------- | --- | ------------------ | --- | --- | ----------- | --- | --- | --- | --- | --- |
Using the rule that dictates how the stress energy tensor changes under conformal maps in Eq. (15), we can
thus predict the form of the transfer matrix in different geometries. For example, the spatial transfer matrix
| defined | on an infinitely |     | long cylindrical |     | geometry |     | with radius | β   | reads |     |     |     |
| ------- | ---------------- | --- | ---------------- | --- | -------- | --- | ----------- | --- | ----- | --- | --- | --- |
|         |                  |     |                  |     |          | "   |             |     | !#    |     |     |     |
κ 2π
|     |     |     |     |     | T =exp | −   | +   | (L +L¯ | )   | ,   |     | (17) |
| --- | --- | --- | --- | --- | ------ | --- | --- | ------ | --- | --- | --- | ---- |
|     |     |     |     |     |        |     | β   | β 0    | 0   |     |     |      |
H
where κ = −πcδt/6, c being the central charge. Here L = 1 dzzT(z) is the generator of the holomorphic
|     |     |     |     |     |     |     |     | 0 2πi |     |     |        |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | ------ | --- |
|     |     |     |     |     | L¯  |     |     |       |     |     | T(z)=T |     |
part of Virasoro algebra in the plane and 0 of the anti-holomorphic part of it, and zz in the original
complex plane coordinates. This in turn implies that the eigenvalues of Hˆ are in one to one correspondence
with those of L , and thus the scaling operators of the theory, which will depend on the corresponding scaling
0
exponents x. These results thus allow us to completely determine the spectrum of the transfer matrices in our
geometry.
As discussed in the main text, the Loschmidt echo of a product state requires working with boundaries
representing the initial and final states, resulting in a strip-like geometry (see Fig.1 main text). This can be
| obtained | via the | mapping4 | ω   | ≡s+iu= | β   | log(z). |     |     |     |     |     |     |
| -------- | ------- | -------- | --- | ------ | --- | ------- | --- | --- | --- | --- | --- | --- |
π
3This is nothing but the euclidean version of the QFT statement that the Hamiltonian is the space integral of the time-time
componentoftheenergy-momentumtensor.
4HerewestrictlyfocusonCFTresultsandomitadditionalfactorswhichenterthecalculationsperformedonthelatticemodel
suchasthesoundvelocityandthenon-universalterms(seemaintext).
11

| The corresponding |     |     | integral | of the | energy | tensor | then gives |     |     |     |     |
| ----------------- | --- | --- | -------- | ------ | ------ | ------ | ---------- | --- | --- | --- | --- |
|                   |     |     |          |        |        |        | π          | πc  |     |     |     |
Hˆ
|     |     |     |     |     |     |     | = L 0 − | ,   |     |     | (18) |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | ---- |
|     |     |     |     |     |     |     | β       | 24β |     |     |      |
leading to the following expression for the spectrum of the transfer operator:
|           |              |     |     |     |        | "   |       |          | !#         |     |      |
| --------- | ------------ | --- | --- | --- | ------ | --- | ----- | -------- | ---------- | --- | ---- |
|           |              |     |     |     |        |     | κ s π | (cid:16) | 1 (cid:17) |     |      |
|           |              |     |     |     | T =exp | −   | + L   | +O       |            | ,   | (19) |
|           |              |     |     |     |        |     | β β 0 | β2       |            |     |      |
| where now | κ =−πcδt/24. |     |     |     |        |     |       |          |            |     |      |
s
We can extract even more informations from the CFT. In particular, if we consider Eq. (14) when the two
points are separated only along the temporal direction, their distance being i(u −u ), we obtain
|     |     |     |     |      |      |     |                     |          |             | 1 2 |      |
| --- | --- | --- | --- | ---- | ---- | --- | ------------------- | -------- | ----------- | --- | ---- |
|     |     |     |     |      |      |     | (cid:18) (cid:19)2x | (cid:20) | (cid:21)−2x |     |      |
|     |     |     |     |      |      |     | π                   | π        |             |     |      |
|     |     |     |     | ⟨ϕ(ω | )ϕ(ω | )⟩= | sin                 | (u −u    | )           | .   | (20) |
|     |     |     |     |      | 1    | 2   | β                   | β 1      | 2           |     |      |
c
and, as discussed in the main text, for twist fields where x becomes ∆ n = (n−1/n), such a prediction gives
24
access to the Tsallis entropies of order n [57, 58, 59, 60, 61], and consequently to the generalized temporal
| entropies | of our | problem | after  | analytic | continuation |     | n→1. |     |     |     |     |
| --------- | ------ | ------- | ------ | -------- | ------------ | --- | ---- | --- | --- | --- | --- |
| C Tensor  |        | Network | setup, |          | symmetric    |     | MPO  |     |     |     |     |
In this section we briefly describe the methods used in this work for building the MPO of the transition matrix
T and extract its eigenvalues. Calculations have been performed using the ITransverse.jl library [79], built
| on top of | ITensors.jl |     | [80]. |     |     |     |     |     |     |     |     |
| --------- | ----------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
The concrete algorithm was introduced in [42] inspired on those presented in [37, 39]. It is basically a power
methodthatiterativelyappliesthetransfermatrixT toaninitialMPSstateandusesalowrankapproximation
| of the reduced |     | transition | matrices |     | in order | to compress | the | MPS. |     |     |     |
| -------------- | --- | ---------- | -------- | --- | -------- | ----------- | --- | ---- | --- | --- | --- |
Our temporal MPO T is defined by contracting one column of the infinite tensor network shown in Fig. 2
in the main text, containing N +2N (with N =β /δt) tensors. The basic ingredients for the MPO are the
|     |     |     |     | T   | β   | β   | 0   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tensors W i (δt) associated with a given site and timestep, building up the time evolution operator U(δt) (see
| Fig. 1 in | the main | text). |     |     |     |     |     |     |     |     |     |
| --------- | -------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Since we work with translationally invariant systems, there is no explicit space dependence, and all tensors
havethesamedependenceonδt. Theonlyinhomogeneitiesinthetimedirectionareinducedbytheinitialstate
| |ψ ⟩, which | we  | take as | a product | state. |     |     |     |     |     |     |     |
| ----------- | --- | ------- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- |
0
For a gapped T, the intensive Loschmidt echo converges exponentially fast to l = −|λ |/T and in order to
0
extract λ we use an MPS ansatz for the dominant eigenvectors |R⟩ and ⟨L| of T.
0
Therolesofvirtualandphysicallegsareinterchangedifoneperformsatransversecontraction,asthetemporal
MPO T is applied sideways to the boundary MPS which will result in the dominant left and right vectors. In
this sense, an asymmetry in the virtual legs of the MPO tensors will result in a left dominant vector for T
which is different from the right one, ie. |L⟩ ̸= |R⟩. Nevertheless, we were able to work with tensors which
are symmetric on both the physical and virtual legs, so that this additional complication does not arise. More
specifically, for the Ising model we employ the compact representation proposed in [44] at second-order in the
| Trotter | expansion, | for | which | we obtain | for           | the MPO | tensors |                  |     |          |      |
| ------- | ---------- | --- | ----- | --------- | ------------- | ------- | ------- | ---------------- | --- | -------- | ---- |
|         |            |     |       |           | (cid:18)      |         | p       |                  |     | (cid:19) |      |
|         |            |     |       |           | cos(δt)(a1+bσ |         | )       | isin(δt)cos(δt)σ |     |          |      |
|         |            |     |       | W =       |               |         | z       |                  |     | x ,      | (21) |
p
|                       |     |     |     |                           | isin(δt)cos(δt)σ |     | x   | isin(δt)(a1+bσ |     | z ) |     |
| --------------------- | --- | --- | --- | ------------------------- | ---------------- | --- | --- | -------------- | --- | --- | --- |
| with a=1−2sin(gδt/2)2 |     |     | and | b=2isin(gδt/2)cos(gδt/2). |                  |     |     |                |     |     |     |
The generalization of this type of construction unfortunately would not lead to a symmetric left-right MPO
in the case of the Potts model, due to the presence of both σ and σ† terms (cfr. Eq. (11) in the main text).
|     |     |     |     |     |     |     |     | i   | i   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Nevertheless,wearestillabletoextractasymmetricexpressionbyconsideringthetwo-bodyoperatorU (δt)
i,i+1
and separating it via a symmetric SVD factorization. The resulting MPO tensors are again symmetric in both
| physical | and virtual |     | legs. |     |     |     |        |     |     |     |     |
| -------- | ----------- | --- | ----- | --- | --- | --- | ------ | --- | --- | --- | --- |
|          |             |     |       |     |     |     | T¯ =Tt |     |     |     |     |
Being able to work in this symmetric gauge in which (where the bar denotes complex conjugation),
wecanthusfocusonextracting|R⟩andthenobtain⟨L|=⟨R¯|byasimpletransposition.
Thisalsosignificantly
improves the numerical stability of our TN simulations (see eg. [81] and references therein for a discussion on
| the challenges |     | presented | by  | non-hermitian |     | eigenvalue | problems). |     |     |     |     |
| -------------- | --- | --------- | --- | ------------- | --- | ---------- | ---------- | --- | --- | --- | --- |
12

Figure7: Numericalresultsforthereal(left)andimaginary(right)partsofthegeneralizedentropiesforthecriticalIsing(top
row) and Potts (bottom row) with fixed BC. Thick solid lines denote the CFT predictions.
Havingobtainedtheformofthetransfermatrixtensors,webuildthetransverseMPOforT byincorporating
N steps of imaginary time (δt → −iδβ ) at the edges of the TM, where the real time evolution is performed
| β   |     |     | 0   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
in the center.
ThedeterminationofthedominanteigenvaluesofT wasperformedviaasymmetricpowermethodalgorithm
which simply goes as follows: starting from an initial guess for the right dominant vector |R⟩ in the form of
MPS along the temporal direction, at each iteration T is applied to |R⟩ and the resulting MPS is truncated
by optimizing the overlap ⟨R¯|R⟩, since, as already discussed, in our case we have ⟨L| = ⟨R¯|. This is done by
appropriately truncating on the singular values of the reduced transition matrices, Eq.(8) main text (see [42]
for additional details). The iterations are repeated until convergence is reached, and the eigenvalues of the
transition matrices are used to compute the entropies along the temporal chains.
| D Fits | on real | parts of | the dominant | eigenvalues. |     |     |
| ------ | ------- | -------- | ------------ | ------------ | --- | --- |
ByfittingtherealpartofthedominanteigenvaluesofT wecangetfurtherconfirmationofourCFTpredictions,
although the procedure is a bit more involved. When it comes to the real parts, by comparing with Eq. (4) in
the main text we see that a single term a =2β av+b does not allow us to resolve the pieces involved, but we
|     |     |     | 0   | 0   |     |     |
| --- | --- | --- | --- | --- | --- | --- |
can extract some additional information by looking at the difference of the real parts for different values of β .
0
|     |     | λβ0≡β1 | −λβ0≡β2, |     |     |     |
| --- | --- | ------ | -------- | --- | --- | --- |
Indeed, if we consider ∆ λ ≡ we expect that Re(∆ λ ) = 2a(β −β )+O(1/T2), whereas
|      |              | β 0 0 | 0   |     | β 0 | 1 2 |
| ---- | ------------ | ----- | --- | --- | --- | --- |
| Im(∆ | λ )=O(1/T3). |       |     |     |     |     |
β 0
/T2.
We thus fit the difference of the real part for different values of β with a functional form f 2 = b 0 +b 2
As an example, we fit for the Ising model, for the case ∆ λ (β =0.4−β =0.2) we get b =0.51, whereas for
|     |     |     |     | β 0 |     | 0   |
| --- | --- | --- | --- | --- | --- | --- |
∆(β =0.6−β =0.2) we find b =1.02, showing again an excellent agreement with the CFT prediction which
0
| indeed | gives a ratio | of 2 between | the two values. |     |     |     |
| ------ | ------------- | ------------ | --------------- | --- | --- | --- |
At this point, we can also extract the non-universal coefficient a=b /2(β −β )≃1.275 from here, which is
|            |            |              |                  |             | 0 1 | 2   |
| ---------- | ---------- | ------------ | ---------------- | ----------- | --- | --- |
| consistent | with the a | from the fit | of the imaginary | part above. |     |     |
0
E Corrections to the leading finite size scaling for the generalized entropies
TherateatwhichourresultsapproachtheCFTpredictionsofthegeneralizedentropiesisaffectedbythechoice
of initial state for the Loschmidt echo , which can be associated with different boundary CFT states and as a
result with a different operator content of the theory. As an example, we show in Fig. 7 the curves obtained for
IsingandPottsusingfixedBC.TherewecanappreciatethattheconvergencetotheCFTresultsissignificantly
13

slower than the corresponding one for free BC, particularly for approach of the imaginary part to the expected
constant value.
Such large deviations are quite usual in the context of studying the scaling of the entanglement entropy in
different scenarios [68, 82, 83, 84, 85]. Here we show that even for generalized entropies such corrections are
induced by finite size effects and vanish as inverse powers of the IR and UV cutoff. We expect indeed that
both a finite β and T to introduce perturbations to our system. In the following, we will focus on the distance
0
between the numerical results from our TN simulations for the imaginary part (taken at midchain) and the
expected CFT result, ie. ∆S =Im(S ) −Im(S ) even if one could work with the full chain length,
gen CFT gen TN
in order to study the approach of all the points to the CFT results.
In order to investigate the effect of T, we can use the standard analysis described in [68]: In particular, we
expect that the leading corrections should scale as some inverse power of ∆S ∝T−x, with x the lowest neutral
relevant field. This implies that for Ising x=1=∆ , while for Potts we expect x=4/5=∆ .
ϵ ϵ
Our numerical results for Ising, as well as for Potts with fixed BC are indeed consistent with this kind of
prediction, see Fig. 8.
Now we can add the effects of finite β : since β plays the role of a UV cut-off, we now expect corrections
0 0
∆S ∝ βγ with γ the scaling dimension of the most relevant operator coupling to ∆S. Putting everything
0
together we thus expect that the scaling variable for the correction should be βγ/x/T.
0
Performing our fits, we find that for Ising with free BC an exponent γ = 1.5 which is compatible with
γ = 2−∆′ , where ∆′ = 0.5 is the boundary magnetization. For fixed BC on the other hand our best fit
σ σ
suggests γ =0.5. For Potts with fixed BC initial state we find instead γ =2/5, which once more is compatible
with ∆ . The exponents change as expected for the Potts model with free BC: there the best fits give an
ϵ
exponent x≃4/3 (compatible with a neutral para-fermion condensate) and γ ≃8/5, which is compatible with
γ =2−2/5, see Fig. 8
14

Ising fixed BC, γ =0.5,x=1
x=1.0, ?=0.5 x=1.0, ?=0.5
Ising free BC, γ =1.5,x=1
Potts fixed BC, γ =2/5,x=4/5
Potts free BC, γ =8/5,x=4/3
Figure 8: Scaling behavior for ∆S, the difference of the imaginary part (at mid-chain) of the Ising (first rows) and Potts
(bottom rows) entropies with the respect to the CFT predictions, for fixed and free boundary conditions. ∆S for the various
cases are plotted as function of T/βγ/x (left) and of β /Tx/γ (right). The coefficients resulting from the best fits are shown
0 0
in the plots.
15
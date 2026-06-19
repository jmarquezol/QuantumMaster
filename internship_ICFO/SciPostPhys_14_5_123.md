SciPostPhys. 14,123(2023)
Collective Monte Carlo updates through
tensor network renormalization
Miguel Frías-Pérez1,2,3⋆ , Michael Mariën4, David Pérez-García5,6,
Mari Carmen Bañuls1,2 and Sofyan Iblisdir3,5
1Max-Planck-InstitutfürQuantenoptik,Hans-Kopfermann-Str. 1,
D-85748Garching,Germany
2MunichCenterforQuantumScienceandTechnology,
Schellingstr. 4,D-80799München,Germany
3DepartamentdeFísicaQuànticaiAstronomia&InstitutdeCiènciesdelCosmos,
UniversitatdeBarcelona,08028Barcelona,Spain
4KBCBankNV-Havenlaan2-1080Brussels-Belgium
5DepartamentodeAnálisisMatemáticoyMatemáticaAplicada,
UniversidadComplutensedeMadrid,28040Madrid,Spain
6InstitutodeCienciasMatemáticas,CampusdeCantoblanco,28049Madrid,Spain
⋆miguel.frias@mpq.mpg.de
Abstract
We introduce a Metropolis–Hastings Markov chain for Boltzmann distributions of clas-
sical spin systems. It relies on approximate tensor network contractions to propose
correlatedcollectiveupdatesateachstepoftheevolution. Wepresentbenchmarkcom-
putations for a wide variety of instances of the two-dimensional Ising model, including
ferromagnetic,antiferromagnetic,(fully)frustratedandEdwards-Andersonspinglassin-
stances,andweshowthat,withmodestcomputationaleffort,ourMarkovchainachieves
sizeableacceptancerates,eveninthevicinityofcriticalpoints. Ineachofthesituations
we have considered, the Markov chain compares well with other Monte Carlo schemes
suchastheMetropolisorWolff’salgorithm: equilibrationtimesappeartobereducedby
a factor that varies between 40 and 2000, depending on the model and the observable
beingmonitored. Wealsopresentanextensiontothreespatialdimensions,anddemon-
stratethatitexhibitsfastequilibrationforfiniteferro-andantiferromagneticinstances.
Additionally, and although it is originally designed for a square lattice of finite degrees
offreedomwithopenboundaryconditions,theproposedschemecanbeusedassuch,or
with slight modifications, to study triangular lattices, systems with continuous degrees
of freedom, matrix models, a confined gas of hard spheres, or to deal with arbitrary
boundary conditions.
CopyrightM.Frías-Pérezetal. Received15-09-2022
ThisworkislicensedundertheCreativeCommons Accepted28-03-2023
Checkfor
Attribution4.0InternationalLicense. Published22-05-2023 updates
PublishedbytheSciPostFoundation. doi:10.21468/SciPostPhys.14.5.123
Contents
1 Introduction 2
1

SciPostPhys. 14,123(2023)
2 Markov chains and tensor network renormalisation 4
3 Two-dimensional Ising models 8
4 Three-dimensional Ising models 15
5 Other models 18
6 Discussion 27
A MPS renormalisation 28
B Arbitrary boundary conditions 33
References 34
1 Introduction
MarkovChainMonteCarloiscentraltoourunderstandingofstronglycorrelatedsystems[1].
Whenthenumberofdegreesoffreedomistoolargeforexactcomputations,andperturbative
methods are ineffective, Monte Carlo sampling often emerges as the method of choice for
numericalinvestigation. MarkovchainMonteCarlohascontributedsignificantlytothecurrent
state-of-the-art in fields like e.g. high temperature superconductivity [2], ab initio quantum
chemistry[3],or(lattice)quantumchromodynamics[4].
In statistical physics, Monte Carlo sampling has made it possible to chart phase diagrams
of several paradigmatic spin systems [5, 6]. The fundamental problem in this context is to
sample according to the Boltzmann distribution. To achieve this goal, Markov chain Monte
Carlomethodsproduceasamplebysubjectinganinitialconfigurationtoacarefullydesigned
stochastic evolution in the space of configurations. Well-known examples are the Metropolis
algorithmandheatbathdynamicsMarkovchainswhereatmostonespinismodifiedateach
step [6], or the Wolff algorithm, where clusters of spins are flipped at once [7, 8]. The ap-
plications of these algorithms are countless, but there are important circumstances, such as
geometricfrustrationordisorder,wheretheirlimitationsbecomeapparent[6,5].
Over the last two decades, a second notion has been gradually recognised as crucial to
our understanding of strongly correlated systems: tensor networks states [9]. In the realm
ofmany-bodyquantummechanics,the(simple)entanglementpatterns,presentincollections
of identical particles in short range interaction, enables a description that conceptually tran-
scends mean field approximations, but does not demand the exponential cost of exact diago-
nalisation [10, 11]. Tensor networks are also used in many-body classical physics. The first
applications were proposed by Nishino in [12, 13, 14], and significant developments have
beenmadepossiblebytheadvancementsintensornetworkalgorithms. Itwasshownin[15]
that partition functions of all spin systems in nearest neighbour interaction, including inho-
mogeneous and finite ones, could be represented as a tensor network. While the exact con-
traction of the tensor network is in general computationally intractable [16, 17], this idea
has been used in practice to address many physical problems via an approximate contraction
[12,18,19,20,21,22,23,24,25]. Tensornetworkmethodshavebeensuccessfullyappliedto
avarietyofclassicalandquantumtwodimensionalproblems(e.g[20,26,27,28])including
continuousvariables[29,30,31,32],andthreedimensionalclassicalmodels[21,33,34].
2

SciPostPhys. 14,123(2023)
Besides solving concrete problems to very good precision, these contributions have been
insightful: wehaveforexamplelearntthatthenotionofbipartitionSchmidtweights,ordinary
inquantuminformationtheory,isalsorelevanttoclassicalstatisticalphysics. However,unless
an implausible collapse of complexity classes is found, both tensor network and Monte Carlo
methodsareboundtobeultimatelylimited,sincethereexistinstancesoftheIsingmodelfor
which the evaluation of the partition function is #P, even in multiplicative approximation
[35, 36]. The downside of these fundamental obstructions is that a complete understanding
ofthesesystemswill(very)likelyalwaysbeoutofreach. Theupsideisasustainedinterestin
developing new methods to continually push the boundary of what we can learn about these
systems.
Earlier works have looked into particular connections between Monte Carlo and tensor
network methods. One perspective has been using Monte Carlo sampling to approximate
tensor network contractions, either to contract or optimize a quantum state [37, 38, 39], or
to approximate classical partition functions represented by a TN [24]. For the latter task,
focusing on the square lattice O(2) model, a thorough comparison between TN-based and
Markov Chin Monte Carlo techniques was presented in [40, 41]. A different angle has been
theconstructionofstatisticalmixturesofpuretensornetworkstatestorepresentthethermal
ensemble of a quantum system [42, 43, 44]. In the context of classical systems, yet another
possibility has been proposed in [45, 46], namely the static sampling from a tensor network
asawaytoobtainrelevantspinconfigurationsofagivenHamiltonianatsometemperature.
In this work, we present and explore a novel connection between tensor networks and
Monte Carlo methods that goes beyond previous studies. Our primary concern here will be
samplingconfigurationsrepresentativeoftheBoltzmanndistributionofclassicalnearestneigh-
bourHamiltoniansatfinitetemperature. Toachievethisgoal,weintroduceaTensorNetwork
Metropolis-Hastings (TNMH) Markov chain [1, 47], where the asymmetric prior, i.e. the dis-
tribution from which the new candidate configuration is drawn at each step, is an approxi-
mationtothetargetdistribution,obtainedviaaninexpensivetensornetworkrenormalisation
contraction. Our approach does not oppose but genuinely combines TN and Markov Chain
Monte Carlo ideas. In this way, it features concrete advantages with respect to each strategy.
Thesalientpropertiesoftheschemeintroducedherearethefollowing.
(i) It is universal. That is, it works identically for all instances of a given model. This is
in contrast to other Monte Carlo algorithms where a powerful prior choice can only be
built by relying on a deep insight about the target distribution, and thus has limited
applicability beyond the model for which it has specifically been tailored. That is for
instancethecaseofWolff’salgorithm,whichperformsextremelywellforferromagnetic
Isingmodels,butratherpoorlyforantiferromagnetsorfrustratedinstances. Inturn,we
willshowthatourmethodfaresconsistentlywellforavarietyofmodelsthatareallvery
differentfromoneanother.
(ii) The scheme produces collective updates. That is, the state of each degree freedom of
theconsideredsystemissusceptibletochangeateachMonteCarlostep. Wehavefound
that the computational effort scales mildly with increasing acceptance rates in a broad
variety of instances. Presumably as a consequence, we have found that the number of
Monte Carlo steps necessary to reach convergence is between ∼101 and ∼103 shorter
thanthoseofotherwell-establishedMonteCarloalgorithmsforseveralinstancesoftwo
andthreedimensionalmodelsoftheIsingtype.
(iii) As compared to algorithms that purely rely on a tensor network renormalisation of the
partition function, the shift to sampling results in the substitution of systematic errors
withstatisticalerrors,sinceourTNMHschemesatisfiestheclassicalsufficientconditions
3

SciPostPhys. 14,123(2023)
for convergence (see next section). Thus, modest tensor network contraction schemes,
too inaccurate for a direct evaluation of a chosen observable, can be successfully used
in our method, as they still enable collective updates with sufficiently high acceptance
rates.
(iv) The scheme is versatile. As we shall see, a Markov chain designed for Ising models on
asquarelatticewithopenboundaryconditionsisusefulassuchtostudyothersystems,
suchastheλφ4 modelorgasesofhardspheres,otherinteractiongraphssuchastrian-
gularlattices,andarbitraryboundaryconditions.
WehavetestedourMarkovchainsystematicallyinavarietyofinstancesoftheIsingmodel
defined on finite square lattices: ferro- and antiferromagnetic, frustrated, disordered, in two
and three spatial dimensions. One may anticipate that for systems with large (or even di-
verging) correlation length, our scheme will perform increasingly poorly if the bond dimen-
sion(parameterthatcontrolsthecostandaccuracyofthetensornetworkrenormalisation)is
fixed. Ourfindingsareconsistentwiththisexpectation,withdropsinacceptanceratesactually
signaling phase transitions. But we have also observed that for ferro- and antiferromagnetic
instances, acceptance rates remain fairly high for systems of considerable size across a phase
transition, even with a bond dimension as low as D = 2. Equilibration and decorrelation
timesinourTNMHschemehavebeenfoundtobesystematicallylowerthanfortheMetropo-
lisandWolff’salgorithms. Asexpected,frustratedandspin-glassinstanceshaveturnedoutto
bechallenging,notonlyforfundamentalcomplexity-theoreticreasons,butalsobecausetheir
studyiscomplicatedbyill-conditioningissues[48]. However,eveninsuchcases,andwithout
any optimization of our renormalisation procedure, we have observed that acceptance rates
stayhighenoughtobeusabledowntotemperaturesthatcanbeconsideredlowbynowadays
state-of-the-artstandards.
Therestofthispaperisorganisedasfollows. Thenewalgorithmisdescribedinsection2
in general terms. In section 3 we explore its performance for two dimensional models. In
particular, for a broad variety of instances of Ising models, we explore the role of the bond
dimensionintheacceptancerates,alsoinrelationtothepresenceofcriticaltemperatures. We
furtheranalyzeequilibrationandautocorrelationtimes,anddemonstratehowthemethodcan
beusedtoobtainphysicalobservablesandchartphasediagrams. Insection4,wedemonstrate
howthealgorithmisalsousefulforthreedimensionalsystems,andillustrateitforferromag-
netic and antiferromagnetic instances of the Ising model in cubic lattices of up to 163 sites.
Section 5 is a discussion of situations where our findings could find further applications, and
canbeskippedonafirstreading;therewediscusstriangularlattices,modelswithcontinuous
variablesandsystemsofhardspheres. Anoutlookisprovidedinsection6.
2 Markov chains and tensor network renormalisation
In this section, we present a collective Monte Carlo update where, given a current configu-
ration, tensor network renormalisation is used to propose a candidate and to decide whether
it should be accepted or not. For the sake of concreteness, and with a view to the example
computationsthatwillbeconsideredinthenexttwosections,wewillfocusontheIsingmodel
on a square lattice. Generalisation to other nearest neighbour interactions, such as the Potts
model,isimmediate.
Westartbyfixingsomenotation. AlatticewillbedenotedasΛ=(V,E),whereV standsfor
itsvertices,and E foritsedges. Wewillfocusonsystemsmadeoftwo-stateparticles(classical
spins) residing on the vertices. That is, the sample space will be Ω = {−1,+1}|V| . A spin at
location j∈V willbedenotedbyσ .
j
4

|     |     |     |     |     |     | SciPostPhys. | 14,123(2023) |     |
| --- | --- | --- | --- | --- | --- | ------------ | ------------ | --- |
TheIsingHamiltonianassociatedwithaspinconfigurationωisdefinedas
|     |        |     | (cid:88) | (cid:88) |     |     |     |     |
| --- | ------ | --- | -------- | -------- | --- | --- | --- | --- |
|     | H(ω)=− |     | h σ      | −        | J σ | σ . |     | (1) |
|     |        |     | i        | i        | ij  | i j |     |     |
|     |        |     | i∈V      | 〈i,j〉∈E  |     |     |     |     |
OuraimistosampleaccordingtotheBoltzmanndistribution
|     | π(β)(ω)=e |     | −βH(ω)/Z(β), |     | ∀ω∈Ω, |     |     |     |
| --- | --------- | --- | ------------ | --- | ----- | --- | --- | --- |
(2)
whereβ
denotestheinversetemperature,and
(cid:88)
|     |     | Z(β)= |     | −βH(φ) |     |     |     |     |
| --- | --- | ----- | --- | ------ | --- | --- | --- | --- |
|     |     |       |     | e      | ,   |     |     | (3) |
φ∈Ω
isthepartitionfunction.
A Markov chain is a sequence of configurations ω(0),ω(1),... where the probability that
t-thelementofthissequenceisinsomestateωonlydependsonthestateoftheprevious
the
ω(t −1),
element and on some random numbers. That is, a Markov chain is an evolution
withshortmemory. Iftheprocessusedtodeterminethestateateachtimestepsatisfiessome
ω(t)
general requirements reminded below, lim is a state drawn according to the target
t→∞
probability distribution (2). A very simple Markov chain is the celebrated Metropolis algo-
rithm,whereatmostonespinisflippedateachtimestep.
ApowerfulclassofMarkovchainisthatintroducedbyHastingsin[47]. Inthecontextof
Statistical Physics, this class can be described as follows. Given a current spin configuration
ω,acandidateconfigurationω′ isproposedaccordingtosomepriordistribution g (β)(ω′|ω),
fromwhichweareabletodraw. Thiscandidateisnextacceptedasthenewcurrentstatewith
probability
|     |            |     | (cid:26) | (β)(ω|ω′)  |     | π(β)(ω′)(cid:27) |     |     |
| --- | ---------- | --- | -------- | ---------- | --- | ---------------- | --- | --- |
|     | (ω→ω′)=min |     |          | g          |     |                  |     |     |
| P   |            |     | 1,       |            | ×   |                  | .   | (4) |
| acc |            |     |          | g(β)(ω′|ω) |     | π(β)(ω)          |     |     |
As will be shown shortly, this acceptance rule allows to satisfy reversibility (a.k.a. detailed
balance),Eq.(8),oneoftheconditionswhichwhenmetguaranteesconvergencetothetarget
(β)
probabilitydistribution. Iftheprior g issymmetricinitsarguments,theacceptanceproba-
bility(4)reducestothecelebratedMetropolisalgorithmformula. Butinsomesituations,the
generalisation proposed by Hastings allows encoding some information about the target dis-
(β)(ω′|ω),
tribution in the possibly asymmetric prior g in a beneficial way. It can for example
result in a boosted exploration of the sample space along the iterations of the Markov chain.
The Swendsen-Wang and the Wolff cluster algorithms are examples of Metropolis-Hastings
|     |     |     |     |     | (β)(ω′|ω) |     | π(β)(ω′), |     |
| --- | --- | --- | --- | --- | --------- | --- | --------- | --- |
construction [7, 8]. Actually, an ideal prior is one where g = that is, the
π(β)
prior that consists in direct sampling according to the target probability distribution . Of
course, for generic instances of the Ising model, such an ideal prior is unavailable. But an
π(β)
approximation to this ideal prior might be good enough for Monte Carlo. We will be
(cid:101)
concerned with such approximations that can be constructed through tensor network renor-
malization.
Let n = |V| represent the system size, and let {1,2,...,n} denote a certain sequential
labelling of the vertices (see e.g. figure 19 in App. A). Using Bayes formula, the Boltzmann
distributioncanbeexpressedas
n
|     | π(β)(ω)=π | (β)(σ | (cid:89) | π( β)(σ |         |     |     |     |
| --- | --------- | ----- | -------- | ------- | ------- | --- | --- | --- |
|     |           |       | )        |         | |σ ...σ |     | ),  | (5) |
|     |           | 1     | 1        | k       | k 1     | k−1 |     |     |
k=2
whereπ( β) standsforthemarginaldistributionofthefirstspin,andπ( β)(·|σ
|     |     |     |     |     |     |     | ...σ )denotes |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- |
1 1 k−1
k
theconditionaldistributionforthekthspinwhenthespins1throughk−1arefixedtovalues
5

SciPostPhys. 14,123(2023)
|                          | Themarginaldistributionforthefirstspinπ( |     |        |         |     |                   |     | β)(σ |                           |     |
| ------------------------ | ---------------------------------------- | --- | ------ | ------- | --- | ----------------- | --- | ---- | ------------------------- | --- |
| σ ,...σ                  | .                                        |     |        |         |     |                   |     |      | )canbeexpressedastheratio |     |
| 1                        | k−1                                      |     |        |         |     |                   |     | 1 1  |                           |     |
|                          |                                          |     | π(β)(σ | )=Z(β|σ |     | )/Z(β),whereZ(β|σ |     |      | )representsthepartition   |     |
| oftwopartitionfunctions: |                                          |     |        | 1       |     | 1                 |     |      | 1                         |     |
1
functionforasystemwiththesamenearestneighbourHamiltonianasin Z(β)butwherethe
firstspinhasbeenfixedtothevalueσ
. Asmentionedintheintroduction,thepartitionfunc-
1
tion (3) of any nearest neighbour Hamiltonian can be expressed exactly as a tensor network
(TN), whose bond dimension is equal to the number of states accessible by each local degree
of freedom. For the Ising model, this number is equal to two. In general, neither Z(β|σ )
1
Z(β)
nor can be evaluated exactly. But a TN renormalisation scheme yields approximations
| (β|σ       | ),andZ(cid:101) (β)foreachofthesequantities(seeAppendix |     |     |     |     |     |     |                             |     |     |
| ---------- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --------------------------- | --- | --- |
| Z(cid:101) | 1                                                       |     |     |     |     |     |     | A).Withthem,onecanconstruct |     |     |
( β)(σ
anapproximationπ )=Z(cid:101) (β|σ )/Z(cid:101) (β)tothetruemarginaldistributionforthefirstspin.
|     |     | (cid:101) | 1   |     | 1   |     |     |     |     |     |
| --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
1
ThisBernoullidistributionisnextsampled. Lets theoutcomeobtained. Withthisfixedvalue
1
|     |     |     |     |     |     |     |     | (β|s ,σ | )   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- |
for the first spin, one can compute an approximation Z(cid:101) of the partition function for
|     |     |     |     |     |     |     |     | 1 2 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
eachvalueσ
forthesecondspin. Theseapproximationsarethenusedtoconstructanapprox-
2
imationπ ( β)(σ |s )=Z(cid:101) (β|s ,σ )/Z(cid:101) (β|s )tothedistributionforthesecondspin,conditioned
|     | (cid:101) 2 2 1 |     | 1   | 2   | 1   |     |     |     |     |     |
| --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
on the value s for the first spin. This second Bernoulli distribution is then sampled. And
1
π( β)(σ
so on. For all other sites k > 2, the conditional probability distribution |s ...s )
k 1 k−1
k
can be expressed as the ratio of two TN contractions Z(β|s ...s σ )/Z(β|s ...s ), and
|     |     |     |     |     |     |     |     | 1    | k−1 k 1 k−1 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ----------- | --- |
|     |     |     |     |     |     |     |     | (β|s | ) (β|s      | σ ) |
a TN renormalisation scheme provides approximations Z(cid:101) ...s k−1 and Z(cid:101) ...s k−1
|     |     |     |     |     |     |     |     | 1   | 1   | k   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
to Z(β|s ...s ) and Z(β|s ...s σ ) respectively. These approximations are in turn used
|     | 1 k−1 |     | 1   | k−1    | k   |       |      |     |     |     |
| --- | ----- | --- | --- | ------ | --- | ----- | ---- | --- | --- | --- |
|     |       |     |     | ( β)(σ |     | )toπ( | β)(σ |     |     |     |
tocomputeanapproximationπ |s ...s |s ...s ),whichissampledand
|     |     |     |     | (cid:101) | k 1 | k−1 |     | k 1 | k−1 |     |
| --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | k         |     |     | k   |     |     |     |
yields an outcome s . Fig. 1 illustrates the first two steps of this sequential sampling. The
k
|     | (s  | )   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
configuration ,...,s obtained after the whole lattice is swept will have been drawn with
|     | 1   | n   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
probability
n
|     |     | π         | (β)(s | )≡π    | (           | β)(s ) | (cid:89) π ( | β)(s |s | ),  |     |
| --- | --- | --------- | ----- | ------ | ----------- | ------ | ------------ | ------- | --- | --- |
|     |     |           |       | ,...,s |             |        |              | ...s    | k−1 | (6) |
|     |     | (cid:101) | 1     | n      | (cid:101) 1 | 1      | (cid:101) k  | k 1     |     |     |
k=2
whichtheidentity(5)showstobeanapproximationtoπ(β)(s
|     |     |     |     |     |     |     |     | ,...,s | ). Wewillbeinterested |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | --------------------- | --- |
|     |     |     |     |     |     |     |     | 1      | n                     |     |
inschemeswheretheMetropolis-Hastingsprobabilitytoselectacandidateω′
reads:
(β)(ω′|ω)≡π(β)(ω′).
|     |     |     |     | g   |     |     |     |     |     | (7) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:101)
π(β)(ω)
As explained in Appendix A, the approximate probability can be evaluated for any
(cid:101)
configurationω,andtheupdaterule(4)canbeimplemented. Ourconstructionissummarized
inAlgorithm1.1
| Properties | of the | TNMH | Markov |     | chain |     |     |     |     |     |
| ---------- | ------ | ---- | ------ | --- | ----- | --- | --- | --- | --- | --- |
(i) The construction is universal in the sense that it is independent of the magnetic fields
and couplings that define the Ising instance being considered. Yet, it is adaptive in that
thedetailsoftheHamiltonianaretakenintoaccountwhenthetensorsareconstructed.
(ii) Theconstitutionofthecandidateisindependentofthecurrentconfiguration.
(iii) The update (7) is collective and correlated: in principle all spins of the system could be
refreshedinasingleMonteCarlostep,andthespinvaluesproposedatdifferentsitesare
conditionedbythecorrelationspresentinthetensornetwork. Webelievethisfeatureis
the principal cause for the high acceptance rates and fast equilibration reported in the
next section. Whereas a local update rule could have a hard time overcoming energy
1Aftercompletionofourwork,
|     |     |     | weweremadeawarethatasimilarsamplingscheme, |     |     |     |     |     | basedonBayes’chain |     |
| --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | ------------------ | --- |
rule,hasbeenproposedtodirectlysampleanapproximationtotheGibbsdistribution[45].
Butnostudyonhow
touseitinaMetropolis-HastingsMarkovchainwasmadeinthatwork.
6

|     |     |     |     | SciPostPhys. | 14,123(2023) |     |
| --- | --- | --- | --- | ------------ | ------------ | --- |
Figure1: PictorialillustrationofthefirsttwostepsoftheTNMHsequentialsampling.
Whitedotsrefertositeswherethespinvaluehasbeenfixed.
barriers,weexpectouralgorithmtobemorecapableofhoppingbetweendistantregions
of the configuration space and escape local minima in a single iteration of the Markov
chain.
(iv) Thetransitionmatrix,i.e. thesetofprobabilitiestotransitionfromaconfigurationωto
| aconfigurationω′ | (ω→ω′)=π(β)(ω′)×P |           | (ω→ω′),isreversible(i.e. |     |             |     |
| ---------------- | ----------------- | --------- | ------------------------ | --- | ----------- | --- |
|                  | ,                 |           |                          |     | itsatisfies |     |
|                  | T                 | (cid:101) | acc                      |     |             |     |
detailedbalanced):
|     | π(β)(ω) | (ω→ω′)=π(β)(ω′) |     | (ω′→ω). |     | (8) |
| --- | ------- | --------------- | --- | ------- | --- | --- |
|     | T       |                 |     | T       |     |     |
Thatis,theexpression
|     |                    |     | (cid:26) π(β)(ω) | π(β)(ω′)(cid:27) |     |     |
| --- | ------------------ | --- | ---------------- | ---------------- | --- | --- |
|     | π(β)(ω)π(β)(ω′)min |     | (cid:101)        | ×                |     |     |
|     |                    |     | 1,               |                  | ,   |     |
|     | (cid:101)          |     | π(β)(ω′)         | π(β)(ω)          |     |     |
(cid:101)
ismanifestlysymmetricinωandω′
.
Furthermore,whennumericalerrorsaresmallenoughthatalltheconditionedpartition
functions Z(cid:101) (β|σ ...σ )arestrictlypositive(seeAppendix A),theMarkovchainisalso
1 k
irreducible:
|     | (ω→ω′)>0, |     | ∀ω,ω′∈Ω. |     |     |     |
| --- | --------- | --- | -------- | --- | --- | --- |
T
{π(β)
Thus, even if the distributions : k ∈ V} turned out to be poorly approximated by
k
the TN renormalisation scheme used, it is still possible to guarantee that the Markov chain
[49].
will eventually converge to the target probability distribution This last point will be
illustratedwiththree-dimensionalIsingmodels.
Algorithm 1 TNMHMarkovchain
1: Computethetensorsassociatedwiththedistribution(2).
t=0,anddrawsomeinitialconfigurationω(0)accordingtoanydistributionoverΩ.
2: Set
3: If t>t goto8.
max
Usethetensornetworktodrawacandidateconfigurationω′ accordingtoEq.(7).
4:
Evaluatetheprobabilitiesπ(β)(ω(t))andπ(β)(ω′).
5:
|                        | (cid:101)          | (cid:101) | (cid:166) (β)(ω|ω′)×π(β)(ω′)(cid:169) |         |     |     |
| ---------------------- | ------------------ | --------- | ------------------------------------- | ------- | --- | --- |
| Acceptthechangeω(t)←ω′ |                    |           | π                                     |         |     |     |
| 6:                     | withprobabilitymin |           | 1, (cid:101)                          |         | .   |     |
|                        |                    |           | π(β)(ω′|ω) (cid:101)                  | π(β)(ω) |     |     |
7: t←t+1. Goto3.
End.
8:
7

SciPostPhys. 14,123(2023)
′−J
Figure2: DistributionofthecouplingsontheJ model. Blacklinesindicatebonds
′ ,whileredandbluebondshavecouplings−J
withacouplingofJ andJ,respectively.
BeforeturningtoapplicationsofAlgorithm1,therearetwopointswewouldliketostress.
(i) Although the construction of a candidate at each iteration of the Markov chain is actually
independentfromitscurrentconfiguration,thedecisiontoacceptthiscandidatedoesdepend
ω′)
on the current state. That is, the transition probability (ω → is not independent of ω.
T
Thiscanbeseenexplicitly:
|     |     |     | π(β)(ω) | π(β)(ω′) |
| --- | --- | --- | ------- | -------- |
(ω→ω′)=π(β)(ω′)min{1,
|     |     |           | (cid:101) | × }.    |
| --- | --- | --------- | --------- | ------- |
|     |     | (cid:101) | π(β)(ω′)  | π(β)(ω) |
T
(cid:101)
Notice that a similar sampling scheme, based on Bayes’ chain rule, has been proposed to di-
[45].
rectly sample an approximation to the Gibbs distribution In constrast, TNMH is not an
approximate Gibbs sampler: the decision to accept or reject the candidate, depending on the
Metropolis-Hastingsratio,marksanessentialdifference. (ii)Eventhoughthetensornetwork
contractionsusedinTNMHare(inevitably)approximate,thereversibilityconditionisexactly
ThisensurestheasymptoticconvergencetotheGibbsdistribution.2
satisfied.
| 3 Two-dimensional | Ising | models |     |     |
| ----------------- | ----- | ------ | --- | --- |
Inordertoassessthepotentialoftheconstructionpresentedintheprevioussection,wehave
run tests on instances of the two-dimensional Ising models chosen to cover a broad range of
cases(L×L
squarelattice):
| • Ferromagnetic: | J =1∀〈i,j〉, | h =0∀i. |     |     |
| ---------------- | ----------- | ------- | --- | --- |
|                  | ij          | i       |     |     |
=−1∀〈i,j〉,h
| • Antiferromagnetic: | J   |     | constantacrossthewholelattice. |     |
| -------------------- | --- | --- | ------------------------------ | --- |
|                      | ij  |     | i                              |     |
• J ′−J model: In this model h = 0 ∀i, and couplings alternate between even and odd
i
rowsorcolumns(seeFig.2):
′
|     | J      | =J ,   | J =J,  |              |
| --- | ------ | ------ | ------ | ------------ |
|     | 2j−1,k |        | 2j,k   |              |
|     | J      | =J ′ , | J =−J, | j=1,...,L/2. |
|     | k,2j−1 |        | k,2j   |              |
The point J = J ′ , known as the fully frustrated square lattice Ising model (or Villain
model),ischaracterisedbyextensivegroundstatedegeneracyandmaximalfrustration.
• Edwards-Anderson spin glass: this disordered model is such that h = 0 ∀i, and J
i ij
are random couplings sampled from a Gaussian distribution with zero mean and unit
variance[50].
2Wearegratefultoouranonymousrefereesfordiscussionsthatconvincedusthesetwopointsshouldbeem-
phasised.
8

|     |     |     |     | SciPostPhys. | 14,123(2023) |     |
| --- | --- | --- | --- | ------------ | ------------ | --- |
Wewillbeinterestedinthefollowingobservables: theenergyperspin,
1 (cid:88)
| ϵ=  |     | H(ω)e | −βH(ω) |     |     |     |
| --- | --- | ----- | ------ | --- | --- | --- |
,
|V|Z(β)
ω∈Ω
themagnetisationdensity,
(cid:12) (cid:12)
|     | 1 (cid:88)(cid:12)(cid:88) |            | (cid:12)         |     |     |     |
| --- | -------------------------- | ---------- | ---------------- | --- | --- | --- |
| m=  |                            | (cid:12) σ | (cid:12)e −βH(ω) |     |     |     |
,
|     | |V|Z(β)     | (cid:12) i | (cid:12) |     |     |     |
| --- | ----------- | ---------- | -------- | --- | --- | --- |
|     | ω∈Ω(cid:12) | i∈V        | (cid:12) |     |     |     |
thestaggeredmagnetisationdensity,3
|           | (cid:12)                   |          | (cid:12)         |     |     |     |
| --------- | -------------------------- | -------- | ---------------- | --- | --- | --- |
|           | 1 (cid:88)(cid:12)(cid:88) |          | (cid:12)         |     |     |     |
| m =       | (cid:12)                   | sign(i)σ | (cid:12)e −βH(ω) | ,   |     |     |
| s |V|Z(β) | (cid:12)                   |          | i (cid:12)       |     |     |     |
|           | ω∈Ω(cid:12)                | i∈V      | (cid:12)         |     |     |     |
sign(i) ±1
where is equal to in a checkerboard manner. Finally, we will consider also the
magneticsusceptibility,definedforasystemwithauniformmagneticfieldas
χ =∂m/∂h.
Unlessstatedotherwise,wewillbeconsideringopenboundaryconditions.
Role of the bond dimension
A crucial ingredient of the algorithm described in the previous section is the substitution of
| Z(β|σ | ...σ ) |     |     | (β|σ ...σ | )   |     |
| ----- | ------ | --- | --- | --------- | --- | --- |
exact partition functions k−1 with approximations Z(cid:101) k−1 obtained by
|     | 1   |     |     | 1   |     |     |
| --- | --- | --- | --- | --- | --- | --- |
tensornetworkrenormalisation. Amongstallavailablesmethodsforthisrenormalisation,we
haveusedthematrixproductstate(MPS)renormalisationschemedescribedin[15](seealso
Appendix A). It is a choice of simplicity, which turned out to be sufficient for our purposes.
WehoweverwouldliketostressthattheanalogousTNMHMarkovchaincanbedefinedusing
anyothercontractionscheme,andsomemightyieldbetterresultsthanthosepresentedhere.
InMPSrenormalisation,boththeaccuracyoftheapproximationandthecomputationaleffort
increasewithanintegerparameter,thebonddimension,commonlydenotedD. Wethusexpect
thetotalvariationdistancebetweenthetargetandthepriordistribution,
1
| (cid:13) π(β)−π(β)(cid:13) | =              | (cid:88)(cid:12) π(β)(ω)−π(β)(ω) |           | (cid:12)  |     |     |
| -------------------------- | -------------- | -------------------------------- | --------- | --------- | --- | --- |
| (cid:13)                   | (cid:13)       | (cid:12)                         |           | (cid:12), |     | (9) |
|                            | (cid:101) TV 2 |                                  | (cid:101) |           |     |     |
ω∈Ω
to decrease with increasing values of D. As a result, Monte Carlo rejection rates should de-
creaseasthebonddimensiongrowslarge.
Tocharacterizethebehaviourofourmethod,wehaveexploredtheinterplaybetweenthe
bond dimension, the temperature and the rejection rate for the four different models men-
tioned above (Fig. 3). In all cases, we have verified that the rejection rate decreases with
increasing bond dimension, and even modest values of the bond dimension may yield virtu-
ally rejection-free updates. At the same time, for a fixed D, rejection rates increase in the
vicinity of critical points. This can be understood considering that, presumably, the distance
∥π(β)−π(β)∥
forfixed D willincreasewiththetruecorrelationlength.
(cid:101) TV
Fig.3ademonstratesthesefeaturesfortheferromagneticcase. Thismod(cid:112)elexhibits,inthe
limitoflargesystemsizes,asecond-orderphasetransitionatT =2/log(1+ 2)≈2.269from
c
3Whentheexternalmagneticfieldisuniformlynaught,
averagingoverMonteCarlosampleswouldresultin
zero(staggered)magnetisationevenattemperatureswherethesystemisknowntoexhibitafinitespontaneous
magnetization. The absolute values appearing in our definition of the (staggered) magnetisation are meant to
counterthisartefact.
9

|     |     |     |     |     |     |     |     |     | SciPostPhys. |     | 14,123(2023) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------------ |
6
0.4
|     | 0.35 (a) |     |     |     |     |     | (b) |     |     |     |     |
| --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0.30
0.3
0.25
0.20
0.2
0.15
|     | 0.10 |     |     |     |     | 0.1 |     |     |     |     |     |
| --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0.05
|     | 0.00 |     |     |     |     | 0.0 |         |     |         |     |         |
| --- | ---- | --- | --- | --- | --- | --- | ------- | --- | ------- | --- | ------- |
|     | 1.6  | 1.8 | 2.0 | 2.2 | 2.4 | 2.6 | 1.0 1.2 | 1.4 | 1.6 1.8 | 2.0 | 2.2 2.4 |
|     |      |     |     | T   |     |     |         |     | T       |     |         |
|     |      | D=2 | D=3 | D=4 | D=5 |     | D=2     | D=3 | D=4     |     | D=5     |
|     | 1.0  |     |     |     |     | 1.0 |         |     |         |     | (d)     |
(c)
|     | 0.8 |     |     |     |     | 0.8 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 0.6 |     |     |     |     | 0.6 |     |     |     |     |     |
0.4
0.4
|     | 0.2 |     |     |     |     | 0.2 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 0.0 |     |     |     |     | 0.0 |     |     |     |     |     |
0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
|     |     |     |     | T   |      |     |     |     | T   |     |      |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | ---- |
|     |     | D=2 | D=4 | D=8 | D=16 |     | D=2 | D=4 | D=8 |     | D=16 |
FIGF. 2ig.uRreejec3ti:onTraNteMasHarfuenjcetciotnioonf threatteemspaersatuarefufonrcfotuirondi↵oerfentthiensttaenmcespoefrtahteuIsriengfmorodfeol:u(ra)dFiefrfroemreagnntetic,
(b)Antiferromagneticwithaconstantmagneticfield,(c)Fullyfrustrated,(d)Gaussianspinglass. Thegeometryisthatof
a32ins3t2asnqucaerseloatftitcheewiIthsionpgenmbooudndealryactoandfiitixoensdinsyalsltfeoumrcsasiezse.:Fo(rae)acFhemrroodeml,tahgenreejetcitcio,n(bra)teAwnastiofbetrarinoed-
by
aver agi ⇥ ng ov er 4 0 ind e pen d en t ch a i n s, ea ch r un f o r a m o d e l-d ep=end en t n u m b er o fs t ep s . F o r t h ef u ll y fru st ra t ed a n ds p in glass
m a gn e ti c w it h a c o n s t a n t m a g n e t i c fi e l d h 2 , ( c ) F u ll y f r u s t r a t e d , ( d ) G a u s si a n s p in
cases,thepointsattemperatureswherewebelieveourmethodstartstosu↵erfromill-conditioningissuesareindicatedwithout
hFoertgheeofemrroe-tarnydiasnttihferartomoafganet3ic2in×st3an2cess,qtuheavreertliacatltliicneeswindiitchateocpreitnicablitoyuinntdhearthyercmoondydniatmioicnlsimit.
markgelrafislsli.ngT.
a
a ** i * n W a e l h l a f v o e u th r e. c . a .r s a e nd s o . m F g o en r er e at a or c s h for m al o lc d om e p l, ut t a h tio e ns r p e r j e e se c nt t e i d o i n nt r h a is t p e ap w er. as obtained by averaging over
40 independent chains, each run for a model-dependent number of steps. For the
valufeu.llWyefrhuavsetriamtpeledmaenntdedstphiins hgeluarsissticcatsesetso,fthcoen-pointsattempera tureswh0e.2r5e0w.15e0b.0e5li0e.0v2e5our
vergmenecethfoordthsetTarNtMsHtoMsaurkfofevrchfarionm. Oiulrl-ficnodnindgistaiorening b2y21e2m22pt2y23m2a2r4kers.
|                |     |                                |     |     |     |     | issues | are inPdTicated |     |     |     |
| -------------- | --- | ------------------------------ | --- | --- | --- | --- | ------ | --------------- | --- | --- | --- |
| reportedonFig. |     | 7andonTableI,wherewecompareour |     |     |     |     |        |                 |     | 213 | 214 |
timeFsowritthhestfaeter-roof--thaen-adrtamnettihfoedrsroonmtahgisniesstuiec:inpasrt-ances,theverticallinesindicatecriticalityin PT+ICM - -
|                        |     |               |                           |     |          |     | TNMH+Metropolis |     | 4   | 5 6 | 8   |
| ---------------------- | --- | ------------- | ------------------------- | --- | -------- | --- | --------------- | --- | --- | --- | --- |
| allelthteemtphereinrmg |     | (oPdTy)naanmd | ipcarlailmlelitt.empering |     | combined |     |                 |     |     |     |     |
withisoenergeticclustermoves(PT+ICM)[47,48]. The TABLEI. Firstrow: targetvalueof ,asdefinedbyEq.(9).
comparisoninTableIclearlyshowsthatTNMHoutper-
|                                       |     |     |     |     |           | Secondandthirdrow: |     |     | eachentryrepresentsalowerbound |     |     |
| ------------------------------------- | --- | --- | --- | --- | --------- | ------------------ | --- | --- | ------------------------------ | --- | --- |
| formsthesemethodsbyordersofmagnitude. |     |     |     |     | Tobefair, |                    |     |     |                                |     |     |
onthenumberofMonteCarlosweepsnecessarytodecrease
oursimulationsetuponlydi↵ersabitfromthatofthese   below the value indicated in the same column for paral-
r efe r en c es : w h e rea s p e ri o d icb oun da r y c o nd it io n s a n d a l el[te m p]e rin g ( P T ) a n d p ar all el t e m p erin g p lus i so e n e rg et i c
a m a g n e t ic a l ly o r d e r e d to a p a r a m a g n e t i c p has e 5 2 . S t i ll , f o r th e s y s t e m s iz e c o n s i d e r e d,
temperature T = 0.212 were considered in these refer- cluster moves (PT+ICM) (data read o↵ Fig. **** of ref
32en × ces3,2w,ehreavjeecotpitoednforratoepsenrbeomunadianryrceomndaitrioknasbalnydlow***(*b).elFoowurth0r.o4w): aCcorrroesspsontdhineg wnuhmobelreoftestmeppsenereadteudre
| rawnogrekwtihthatassluigrhrtolyucnolddesrtshysetemcrTiti=ca0l.2.teTmhepneurmabtuerre. |     |     |     |     |     | by  | T N M H .  |          |                  |     |           |
| ------------------------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | ---------- | -------- | ---------------- | --- | --------- |
|                                                                                       |     |     |     |     |     |     | A c tu a l | ly, even | a bond dimension |     | as low as |
ofMarkovchainusedforthethermalaveragewas32for
=
Dth e(2PTa+pICpMea)rcsomtpoutaaltrioenasdayndb3e0insuouffirccaiseen,wthteoreaaschieve our goal of producing collective updates
|     |     |     |     |     |     | te  | r e q uil ib | r a t i o n is | re a c he d , th e | ti m e ne | ed ed b e t w een |
| --- | --- | --- | --- | --- | --- | --- | ------------ | -------------- | ------------------ | --------- | ----------------- |
frethqeuneunmtblyer.oMfdoi↵reerendteitnastialsncreesgusaerddfionrgthethdeisovrdiceriendity o f t h e c r i t i c a l po i n t a r e p ro v i de d o n F i g . 4a,
|         |     | 104      |             |     | 103        | twosampleextractionstoguarantee(su cient)indepen- |     |     |     |     |     |
| ------- | --- | -------- | ----------- | --- | ---------- | ------------------------------------------------- | --- | --- | --- | --- | --- |
| average | is  | in those | references, | and | here. How- |                                                   |     |     |     |     |     |
wheveerr,eitwweouhldabveeveprlyostutrepdristinhgeifroeujreficntidoinngsrwaetreesaigs-a fudenncceti.oGnivoefntahneobssyesrtveabmlesXiz,ewefostruddyiftfheereqnuatnbtiotynd
dinmifiecnanstiloynaslt.ereEdvbeynbyfocronssyidsetreinmgsconadsitlioanrsgiedeantsic2al56×256,
|     |     |     |     |     |     |     |     | acceptance | rates of | about | 0.4 can be |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | ----- | ---------- |
tothoseof[47,48].
|     |     |     |     |     | D = |     |     | eX (at 0p) pX r( | t + tt )e Xr | (t ) X | e( t n+ ste) |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | ------------ | ------ | ------------ |
o b t a i n e d u s in g o n l y a b o n d d i m e n s i o n 4 . A C s c ( a t ) n = b h e 0c i a i d   h f o m 0 i t h h 0i i t . o ( f 10 t ) h i s
W e n e xt m ov e t o a u to c o rr e la ti o n t im e s , t ha t i s, a f- X X 2 ( t ) X ( t ) 2
fi g u r e , o u r d a t a s u g g e s t t h a t th e b o n d d i m e n s i on o n l y n e e d s t o h g r o w 0 il o g h a r i t0 h m i i c a l ly w i t h t h e
systemsizeinordertomaintaintheacceptancerateaboveathresholdvalue. Ourobservations
fortheantiferromagneticcase(Fig.3b)aresimilar.
For the fully frustrated case of the J ′ − J model (Fig. 3c) we obtain lower acceptance
rates, as compared to the two previous cases, but still high enough that the Markov chain is
|     |     |     | = O(10 | −1). |     |     |     |     |     |     |     |
| --- | --- | --- | ------ | ---- | --- | --- | --- | --- | --- | --- | --- |
usable down to T As expected, the rejection rate increases when approaching
the T = 0 critical point. Still, the minimal cost curve D = 2 is sufficient to obtain decent
acceptanceratesdowntoatleastT =0.2,andincreasingthebonddimensionagainsuppresses
rejectionevents. Atverylowtemperatures,acceptanceratesdropdramaticallyandnumerical
[48]
instabilities typical of frustrated systems pointed out in set in. Some strategies exist to
mitigatetheseeffects,buttheirdiscussionisbeyondthescopeofthepresentwork,andwillbe
10

SciPostPhys. 14,123(2023)
Figure 4: (a) TNMH rejection rates near the ferromagnetic Ising phase transition
(T ≃ 2.27) as a function of the system size for different bond dimensions. Dashed
linesaresimplyaguidetotheeye. Inset: Bonddimensionneededtomaintainafixed
rejectionrate(inthiscase0.25,althoughthebehaviourseemstobeindependentof
thevaluechosen)asafunctionofthesystemsize. Thefitshowsthattheincreasein
the bond dimension seems to be only logarithmic. (b) TNMH rejection rates for the
fully frustrated Ising model at T =0.4 as a function of the system size for different
bonddimensions.
thesubjectofaseparatestudy[53]. Again,wehavelookedattherejectionrateasafunctionof
thesystemsizefordifferentbonddimensions(seeFig.4b). Eventhoughthisinstanceismore
challenging,theexampleinthefiguredemonstratesthatatT =0.4perfectlyusableacceptance
ratesofabout0.2orhighercanbeobtainedforsystemsofsizeupto128×128usingabond
dimension, D=6,forwhichcomputationsarenottoodemanding. Fig.5providesadditional
dataregardingtheJ ′−J model,beyondthefullyfrustratedpointJ =J ′ . Itisremarkablethat
the observed maxima of rejection rates are consistent with the predicted critical lines of this
model.
OurfindingsfortheEdwards-Andersonspinglass,Fig.3d,arequalitativelysimilartothose
forthefullyfrustratedcase,presumablybecausethisspinglassisalsocriticalat T =0[50].
Improved approximations of the contraction will generally result in a higher acceptance
rate. Butactually,asfarasthisacceptanceratedoesnotvanishandscaleswellwiththesystem
size,theTNMHschemeshouldbeapplicable.
Equilibration and decorrelation
Equilibration and auto-correlation times are the two crucial time scales in Monte Carlo sim-
ulations. The former controls the number of steps needed by the Markov chain to decouple
from the initial distribution (that is, the distribution from which the first configuration of the
chain is sampled) and reach the desired equilibrium distribution. The latter determines the
minimaltimebetweentwoconsecutivesampleextractionsinordertoguaranteestatisticalin-
dependence. Since these times are typically difficult to bound, let alone calculate, rigorously,
heuristicdiagonosticsarecommonlyusedtoestimatethem. Wehaveusedtwosuchheuristics
to provide evidence that these two time scales are relatively short for the TNMH scheme of
Section2.
Astandardtechniquetodecidethatequilibrationhasoccurredistomonitoranobservable
fromitsvalueatthebeginningoftheMarkovchainuntilitappearstoplateauatanequilibrium
valuearoundwhichitfluctuates[55]. Fig.6aillustratestheevolutionoftheexpectationvalue
forthemagnetisationofaferromagnet,andindependentMarkovchainsevolvedaccordingto
eithertheTNMHalgorithm1(blue),asimplespinflipMetropolisalgorithm(green)orWolff’s
11

|     |     |     |     |     |     |     | SciPostPhys. | 14,123(2023) |     |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------------ | --- |
2.50
1
2.25
2.00
1.75
|     |     |      |     |     |     |     | 10 2 |     |     |
| --- | --- | ---- | --- | --- | --- | --- | ---- | --- | --- |
|     |     | 1.50 |     |     |     |     | −    |     |     |
T
1.25
|     |     | 1.00 |     |     |     |     | 10 4 |     |     |
| --- | --- | ---- | --- | --- | --- | --- | ---- | --- | --- |
−
0.75
0.50
|     |     | 0.0 | 0.2 0.4 | 0.6 | 0.8 1.0 | 1.2 | 1.4 |     |     |
| --- | --- | --- | ------- | --- | ------- | --- | --- | --- | --- |
J
0
Figure5: TNMHrejectionrateforthe J ′−J modelasafunctionofthetemperature
andthevalueofoneofthecouplings(theotherhasbeensettounity). Computations
=
made with a bond dimension D 4. Rejection rates obtained as averages over 40
32×32.
independent Markov chains, each run for 200 steps. Lattice dimensions:
Thephaseseparatrixpredictedin[54]isshowninblack.
clusteralgorithm(orange)–whichisknowntoperformbestforferromagneticinstances. We
see in this numerical experiment that the number of time steps required for TNMH to equili-
brateisabout1/80to1/40thenumberofstepsrequiredforWolffalgorithm,andabout1/103
thenumberofsweepsrequiredbythesinglespinflipMetropolisalgorithm.
More sophisticated equilibration diagnostics can be devised for specific problems. In par-
ticular, for the Edwards-Anderson spin glass, we have run the following test, discussed in
| [56, 57]. | 〈X〉 |     |     |     |     |     |     | [x] |     |
| --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Let stand for the thermal average of an observable X, and denote the
av
disorder average of a quantity x that might depend on the coupling constants {J }. We have
ij
consideredthedisorderaveragedenergy.
(cid:90)
|     |     |          |     | (cid:89) dJ |             | (cid:88) |         |     |      |
| --- | --- | -------- | --- | ----------- | ----------- | -------- | ------- | --- | ---- |
|     |     | [〈H〉] =− |     | (cid:112) a | b e −J 2 /2 | J        | 〈σ σ 〉. |     | (10) |
|     |     | av       |     |             | a b         |          | ij i j  |     |      |
|     |     |          |     | 2 π         |             |          |         |     |      |
|     |     |          |     | 〈ab〉        |             | 〈ij〉     |         |     |      |
Atequilibrium,integratingbypartsallowstoprovethat
|     |     |          |     | (cid:32) |          |       | (cid:33) |     |      |
| --- | --- | -------- | --- | -------- | -------- | ----- | -------- | --- | ---- |
|     |     | 1        |     | 1        | (cid:88) |       |          |     |      |
|     |     | ∆≡ [〈H〉] | +   | β |E|−   |          | [〈σ σ | 〉2] =0,  |     | (11) |
|     |     | |V|      | av  | |V|      |          | i     | j av     |     |      |
〈ij〉
(cid:80)
where [〈σ σ 〉2] is a quantity known as the link overlap. Starting configurations of
〈ij〉 i j av
Markovchains,drawnaccordingtoaneasydistribution,typicallyhavehighenergyandsmall
link overlap. As a result, ∆ typically has a non-zero value when the Markov chain is started.
As Monte Carlo steps are taken, this value decreases in magnitude. It is a common heuristic
todecideequilibrationhasoccurredwhen∆isbelowagiventhresholdvalue.
104
We have implemented this heuristic test of convergence with disorder realisations,
usingtheTNMHMarkovchain. Wehaveobservedthatafterafewtimesteps,∆dropsdrasti-
cally from its initial value before stagnating at a small but finite value. For the vast majority
of disorder realisations, the Markov chain behaved well. That is, it shows frequent jumps be-
tween different configurations from one time step to the next. However, for some disorder
realizations, a few Markov chains (typically less than a sixth of the chains we simulate for
a given disorder realization) remained stuck in their original configurations. This inertia is
∆
what causes to stagnate. Increasing the bond dimension did not help. Actually, we believe
theissueisratherrelatedtoanill-conditioningofthetensornetworkcontraction,inducedby
frustration,aneffectpreviouslyreportedinRef.[48]. Asaresult,theapproximateprobability
12

SciPostPhys. 14,123(2023)
Table 1: First row: target value of ∆, as defined by Eq.(11). Next rows: each entry
represents the number of lattice sweeps necessary to decrease ∆ below the value
indicated in the same column for the given algorithm. An entry of the form ’> a’
indicatesthatmorethanaiterationsareneededtoreachthedesiredvalueof∆. The
systemconsideredisthesameasinFig.6.
∆ 0.25 0.15 0.05 0.025
Metropolis >105 >105 >105 >105
PT >105 >105 >105 >105
PT+ICM 1.1·103 1.6·103 3.2·103 4.3·103
TNMH >102 >102 >102 >102
TNMH+Metropolis 3 3 5 5
weights can be off their true value by orders of magnitude.4 As can be seen from Eq.(4), this
mismatchaffectstheacceptancerates. Thatis,whentheMarkovchainhitsaconfigurationω
suchthatπ(β)(ω)/π(β)(ω)≪1,itmayremainstuckforalongtime,aswehaveobserved.
(cid:101)
Variousstrategiesarepossibletomitigatetheeffectofill-conditioning. Oneistoworkwith
greatermachineprecision,usingforexamplethetechniquesdescribedinRef.[58]. Anotheris
toconsideravariationoftheTNMHAlgorithm1. Averysimplesuchvariationconsistsininter-
spersingspinflipMetropolissweepsinbetweenTNMHmoves. Wehavetestedthispossibility.
OurfindingsarereportedonFig.6bandonTable1,wherewecompareourtimeswithstate-of-
the-art methods: parallel tempering (PT) and parallel tempering combined with isoenergetic
cluster moves (PT + ICM) [56, 57]. This comparison clearly shows that the combination of
TNMHwithsingleflipMCsweepsallowsustooutperformthesemethodsbyordersofmagni-
tude. Thisresultisinteresting: whereasbothTNMHandtheMetropolisalgorithmshowpoor
performanceindividually(fordifferentreasons;ill-conditioninginthecaseofTNMH,locality
inthecaseofsingleflipupdates),theiralternatinguseisdrasticallymoreefficientthaneither
ofthem.
So far, we have counted equilibration times in steps of the Markov chain, which has con-
ceptualrelevance. Fromapracticalpointofviewthough,itisalsointerestingtoknowhowthe
TNMHcomparestoothermethodswhenlookingatprogramexecutiontimes. Tomakesucha
comparisonfairlyisadelicateissuebecausewehavenotsoughttooptimiseourcodeatall: a
detailed comparison with e.g. Metropolis sweeps, which is simpler to optimise is a project in
itself. WecanhoweverprovideindicativetimesrelatedtothedatapresentedonTable 1. With
oursetup,thetimestogetto∆<0.025are>3.93×107sec,1.76×106sec,and6.75×104sec
for the parallel tempering method, the parallel tempering method supplemented with isoen-
ergeticclustermoves,andTNMHrespectively. Webelievethattheseestimatescrediblysignal
thepracticalpotentialoftheTNMHMarkovchainintroducedhere.
We next move to autocorrelation times, that is, after equilibration is reached, the time
needed between two sample extractions to guarantee (sufficient) independence. Given an
observable X,westudythetimecorrelationfunction
〈X(t )X(t +t)〉−〈X(t )〉〈X(t +t)〉
C (t)= 0 0 0 0 , (12)
X 〈X2(t )〉−〈X(t )〉2
0 0
where t is assumed larger than the equilibration time. As discussed in [55], at large t, we
0
expect C (t) to decay exponentially, with a time scale set by the decorrelation time. As the
X
4Interestingly, it is not clear that such configurations actually correspond to local minima or maxima of the
energy.
13

SciPostPhys. 14,123(2023)
1.0
0.8
0.6
0.4
0.2
0.0
100 101 102 103 104
t
m
1.0
0.8
0.6
0.4
0.2
0.0
0.2
− 100 101 102 103 104 105
t
(a)
∆
TNMH+Metropolis
PT+ICM
PT
(b)
Figure 6: (a) Absolute value of the magnetization per site along different chains at
T =1.5, for three algorithms, D =4 TNMH (blue), Wolff’s algorithm (orange) and
a simple spin flip (green) for a ferromagnetic 64×64 lattice. Each line represents
an independent run. Time t is measured in Markov chain iterations for TNMH, in
sweepsforthesimplespinflipMetropolis,andinclusterupdatesfortheWolffalgo-
rithm. (b)Differencebetweenthe(disorderedaveraged)energyperspincomputed
from the Hamiltonian and computed from the link overlap, Eq. (11), as a function
of the number of iterations for three algorithms: TNMH + Metropolis (blue), (PT
+ ICM) (orange) and PT (green). Bond dimension for the TNMH moves: D = 16.
Lattice dimensions: 32×32 (open boundary conditions). The symbol t represents
the number of iterations for TNMH, and the number of lattice sweeps for PT and
PT+ICM.Theerrorbars,smallerthanthesymbols,havebeencomputedbyestimat-
ingthevarianceofthedisorder. ThenumberofMarkovchainsusedforthethermal
average is 30 in all cases, the number of different instances used for the disordered
averageis104 andthetemperaturehasbeensetto T =0.212. (b)
exponential tail has a fixed amount of noise, controlled by the number of samples, a useful
measuretodeterminethattimescaleistheintegratedcorrelationtime
t
(cid:88)
τint(t)≡1+2 C (t ′). (13)
X X
t′=1
Itcanalsobeshownthatitisapproximatelythefactorthatenhancesthevariancewhenaver-
agingoversamplesthatarenotsufficientlydecorrelated[55]. Thetwoquantitiesareplotted
onFig.7aforthemagnetizationofaFullyFrustratedIsingmodelona32×32latticeatT =1.
The motivation for choosing this observable is that often the energy is a poor choice to mea-
sure the decorrelation of samples in a Markov chain. At low temperatures, it is impossible
to distinguish global changes in a configuration of a Markov chain from local motion around
a local minima just by tracking the energy. In this particular case, choosing an observable
that breaks the global spin flip symmetry of the model in consideration allows to assess the
ergodicityofthescheme,sinceforanyconfigurationwithenergyEandmagnetizationmthere
existsanotherwithmagnetization−mandsameenergy. ThedatashowninFig.7ashowsthat
TNMH outperforms a local algorithm by almost two orders of magnitude. Furthermore, we
expect that the difference in performance can only increase as the temperatures are lowered
orthesystemsizeisincreased.
Tofurtherillustratesample-to-sampledecorrelationinouralgorithm,wehaveconsidered
the fully frustrated Ising model and represented in Fig. 7b snapshots at different times for
TNMHandforMetropolissweepsstartingfromasameconfiguration. Thedifferencebetween
bothtechniquesisstriking: whiletheconfigurationsappearinginourtechniqueseemtobear
noresemblancetooneanotherfromoneacceptancetothenext,memoryoftheinitialconfig-
14

SciPostPhys. 14,123(2023)
102
101
100
0 25 50 75 100 125 150 175 200
t
tniτ m
100
10− 1
10− 2
10− 3
10− 4
0 10 20 30 40 50
t
D=2 D=4 D=6 D=8 Spinflip
)t(mC
TNMH
Spin
flip
(a) (b)
Figure 7: (a) Integrated correlation time (13) and (inset) decay of the autocorre-
lation function (12) of the magnetization as a function of time, for different bond
dimensionsonaFullyFrustratedIsingmodelona32×32latticewithopenbound-
ary conditions at T = 1. The error bars have been computed by estimating the
variance of the observables using the same samples. (b) Snapshots of the evolution
of two different Markov chains starting from the same configuration, for the fully
frustratedIsingmodelona64×64latticeat T =0.5. Thetopplotsdisplaythecon-
figuration obtained after three consecutive steps of the TNMH method (with bond
dimension D = 8), while those below show configurations after Metropolis sweeps
attimes t =0,1,10.
urationcanstillbeappreciatedvisuallyaftertenMetropolissweeps.
Observables
WenowturntotheestimationofobservablesfromthesamplesoutputbytheTNMHMarkov
chain. Wehavefocusedontheferromagneticcaseandtheantiferromagneticonewithanex-
ternal field. The absolute value of the magnetisation for the ferromagnetic case is plotted on
Fig.8a(inset)andisingoodagreementwiththetheory[52]. Fig.8aalsoshowsestimatesfor
the fourth order Binder cumulant [6], g =(3−〈m4〉/〈m2〉2)/2. One can appreciate that the
phasetransitionpointiscorrectlysignalledbythelocuswherealldatasetsmeet,asexpected.
On Fig. 8b, we have represented the staggered susceptibility as a function of the tempera-
tureandtheexternalmagneticfieldforanantiferromagnet. Ourfindingsseemtobeingood
agreement with previous studies of this model [59, 60].(cid:112)When the external field is naught,
theferromagneticphasetransitionaround T =2/ln(1+ 2)isrecovered,asexpected,since
c
for a square lattice, a local change of variables allows a mapping between antiferro and fer-
romagnetic instances of the Ising model. As the field increases, the temperature at which the
phase transition takes place decreases. The intuition for this fact is as follows: at h = 0 and
belowthecriticaltemperatureonehasantiferromagneticorder. Inthelargehlimitatthesame
temperature all spins would align with the external field and one would have ferromagnetic
order. Thus,somephaseboundarymustbeencounteredwhengoingfromonetotheother.
4 Three-dimensional Ising models
Just as for planar systems, the partition function of a three-dimensional Ising model can be
expressedasatensornetwork. Asaconsequence,ourTNMHalgorithmimmediatelyextendsto
threedimensions. Theapproximatecontractionofathree-dimensionalTNishoweveramore
15

|     |     |     |     | SciPostPhys. | 14,123(2023) |     |
| --- | --- | --- | --- | ------------ | ------------ | --- |
1.0
| 0.9 | 0.8 |     | 0.0 |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
0.6
| 0.8 | m 0.4 |     | 0.2 |     |     |     |
| --- | ----- | --- | --- | --- | --- | --- |
|     | 0.2   |     | 0.4 |     |     | 30  |
0.7
|       | 1.8 | 2.0 2.2 2.4 2.6 2.8 | 0.6 |     |     |     |
| ----- | --- | ------------------- | --- | --- | --- | --- |
| g 0.6 |     | T                   |     |     |     |     |
|       |     |                     | 0.8 |     |     | 20  |
0.5
|     |     |     | h 1.0 |     |     | χ   |
| --- | --- | --- | ----- | --- | --- | --- |
0.4
1.2
| 0.3            |           |                |     |     |     | 10  |
| -------------- | --------- | -------------- | --- | --- | --- | --- |
| 0.2            |           |                | 1.4 |     |     |     |
| 2.20 2.22 2.24 | 2.26 2.28 | 2.30 2.32 2.34 |     |     |     |     |
1.6
T
1.8
|        |          |               | 1.0 1.2 1.4 | 1.6 1.8 2.0 | 2.2 2.4 |     |
| ------ | -------- | ------------- | ----------- | ----------- | ------- | --- |
| 8 8 16 | 16 32 32 | 64 64 128 128 |             | T           |         |     |
| ×      | × ×      | × ×           |             |             |         |     |
|        | (a)      |               |             | (b)         |         |     |
Figure 8: (a) Binder ratio as a function of temperature for the two dimensional fer-
D=6.
romagneticIsingmodelobtainedthroughTNMHwithabonddimension The
dataapproximatelycrossatonepoint,signallingaphasetransition,ingreatconcor-
dance with the theoretical result T ≈ 2.269. Inset: Magnetisation of the ferromag-
64×64
netic Ising model on a square lattice with open boundary conditions. The
errorbarshavebeencomputedviaajackknifeanalysisfortheBindercumulantand
byestimatingthevarianceforthemagnetization,andaresmallerthanthesymbols.
(b)Susceptibilityχ foranantiferromagneticIsingmodelwithanexternalfield,ona
| 64×64squarelatticeobtainedthroughTNMHwithabonddimension |     |     |     |     | D=6. |     |
| ------------------------------------------------------- | --- | --- | --- | --- | ---- | --- |
Black:
theoreticalpredictionofthecriticallineinthethermodynamicallimit.
demanding problem than its two-dimensional analogue. Still, TN renormalisation schemes
can be applied to find an approximation to the contraction [21, 33, 34]. We have chosen an
unsophisticatedrenormalisationschemeinvolvingprojectedentangledpairstates(PEPS).Two
cutoffparametersnowgoverntheeffortputintheTNMHforthisimplementation: aboundary
PEPSbonddimensionD,andaboundaryMPSbonddimensionχ (seeAppendixAfordetails).
We have considered two instances of the Ising model: ferromagnetic, and antiferromagnetic
with an external magnetic field. The upshot is that our TNMH performs very well, even with
| ratherlowvaluesfor | D andχ. |     |     |     |     |     |
| ------------------ | ------- | --- | --- | --- | --- | --- |
Fig. 9 shows the interplay between the two parameters D and χ, the temperature, and
therejectionrate. Again,thepeakintherejectionratesignalsthepresenceofacriticalpoint
(displacedduetofinitesizeeffects). Asthiscriticalpointisapproached,rejectionratesincrease
muchfasterthanintwodimensions,andputtinginmorecomputationaleffortbyincreasingD
andχ nowproducesmilderdropsinrejectionrates. Weattributethissituationtoanincrease
of correlations in the system due to a higher coordination number for each spin. Still, these
preliminaryresultsareveryencouraging,sinceusinganon-optimizedcontractionscheme,and
modestvaluesfortheparametersDandχ,usableacceptancerates(>0.12and>0.05forthe
ferro-andantiferromagneticcaserespectively)havebeenfoundacrossthewholetemperature
rangeconsidered,forsystemsaslargeas163=4096spins.
AnalogoustoFig.6a, whichexploredequilibrationinthetwodimensionalcase, weshow
the energy of a 163 ferromagnetic Ising model as a function of time on Fig. 10, both for the
TNMH Markov chain and for the three-dimensional Wolff algorithm. As in two dimensions,
the former appears to necessitate a lower number of steps than the latter. The magnetisation
of the ferromagnetic Ising model has also been plotted in Fig. 11 and shows good agreement
withpreviousstudies[61,62,63].
Since observables can also be expressed as a TN, it is possible to estimate them using a
directcontraction[15],andonemightthenwonderifthesamplingprocedure,whichinitself
requiresaTNcontraction,providesanadvantagewithrespecttosuchadirectcalculation. But,
16

SciPostPhys. 14,123(2023)
Figure9: TNMHrejectionratesforthethree-dimensionalIsingmodelasafunctionof
thetemperature. Plot(a)correspondstoauniformferromagnetand(b)toauniform
antiferromagnetinafieldh=3. DdenotesthePEPSbonddimension,whileχ stands
for the boundary bond dimension used when compressing the PEPS associated to a
plane of the lattice. A lattice of size 16×16×16 was used, with open boundary
conditions, and for each bond dimension and temperature, 50 chains were run for
150 steps each. The critical temperature is T ≈ 4.512 [61] for the ferromagnetic
c
Ising model, and T ≈4 [63] for the antiferromagnetic with this field (We attribute
c
theoffsetwithrespecttothisvaluetofinitesizeeffects.).
0.8
0.6
0.4
0.2
0.0
0 5 10 15 20 25 30
t
m
Figure10: SinglesitemagnetizationalongdifferentMarkovchainsat T =3fortwo
algorithms, Wolff’s (orange) and TNMH (blue) (D = 2,χ = 2) for a ferromagnetic
16×16×16lattice. t representsthenumberofclustermovesintheformercaseand
thenumberofTNMHiterationsinthelatter.
whiletheTNMHalgorithmcansucceedwithaveryundemandingapproximateTNcontraction
(i.e. usingverylowbonddimensions),achievingaresultofcomparablequalitybydirectcon-
traction generally requires more computational effort. To make this point more concrete, we
have compared the value of the average energy in the three-dimensional ferromagnetic case,
as obtained with the TNMH scheme and with direct TN contractions with different values of
thebonddimensions(Fig.12). Weobservethat,attemperatureswherethedirectcontraction
with up to (D,χ) = (8,16) was not sufficient to obtain an accurate estimate of the energy,
theTNMHwith(D,χ)=(2,2)wassuccessful,sinceitproduceddecentacceptancerates,and
eventuallyprovidedgoodsamplesthankstoirreducibilityandreversibility.
17

SciPostPhys. 14,123(2023)
Figure 11: Magnetisation (blue) and energy (orange) per spin of the ferromagnetic
Ising model (a) and staggered magnetisation and energy per spin of the antiferro-
magneticIsingmodelinanexternalfield(b)onacubic16×16×16latticewithopen
boundaryconditions. Theerrorbarsaresmallerthanthesymbols. Thelargestbond
dimensionsusedtoobtainthecurveswere D=4,χ =8.
5 Other models
IntheprevioussectionswehavepresentedtheTNMHalgorithmindetailandbenchmarkedit
fortheIsingmodelontwo-andthree-dimensionalsquarelattices. However,theschemeoffers
great versatility. In this section, we summarize a number of possibilities to apply and extend
the algorithm for more general problems, which will be explored in further detail elsewhere.
WeshowhowtodealwitharbitraryboundaryconditionsinAppendixB.Thereaderinterested
onlyinthebasicalgorithmcansafelyjumptosection6.
0.25
0.20
0.15
0.10
0.05
0.00
3.50 3.75 4.00 4.25 4.50 4.75 5.00 5.25 5.50
T
(cid:15)
(D,χ)=(2,4) (D,χ)=(6,12) TNMH
(D,χ)=(4,8) (D,χ)=(8,16)
Figure 12: Relative error ε in the average energy per spin computed via differ-
ent techniques for different temperatures (near the phase transition) for the three-
dimensional ferromagnetic Ising model. The reference values are obtained using
Wolff’s algorithm, purple rhombi are obtained using samples from our algorithm
with(D,χ)=(2,2),andtheotherresultsareobtainedtakingderivativesofthelog-
arithmoftheapproximatecontractionoftheTNrepresentingthepartitionfunction.
18

SciPostPhys. 14,123(2023)
The XY model
In absence of a vector potential, the XY model describes a lattice of planar spins, interacting
throughtheHamiltonian
|     |     |     |     | =−  | (cid:88) cos(θ | −θ  | )v, |     |      |
| --- | --- | --- | --- | --- | -------------- | --- | --- | --- | ---- |
|     |     |     | H   |     |                |     |     |     | (14) |
|     |     |     |     | XY  |                | i   | j   |     |      |
〈i,j〉
where the local variables are the angles {0 ≤ θ < 2π : i ∈ V}. Although these variables
i
are continuous, this model can be mapped into a system that allows to use a variation of
the sampling method used for the Ising model. First, a duality transformation establishes an
equivalencebetween(14)andasystemofintegervariablesresidingonthe(oriented)linksof
the lattice involved in four-body interactions [64, 65, 66, 67]. That is, the partition function
takestheform
|     |     |       |     |          | (cid:32) | (cid:33) |            |         |      |
| --- | --- | ----- | --- | -------- | -------- | -------- | ---------- | ------- | ---- |
|     |     |       |     |          | N        |          | (          | i) ( i) |      |
|     |     |       |     | (cid:89) | (cid:88) |          | (cid:89) n | ,n      |      |
|     |     | Z(β)= | lim |          | I        | (β)      | F 3        | 4 ,     | (15) |
|     |     |       |     |          |          | n        | (i)        | (i)     |      |
|     |     |       | N→∞ |          |          | l        | n          | ,n      |      |
|     |     |       |     | l∈E      | n =−N    |          | i∈V 1      | 2       |      |
l
| (i) (i) | (i) | (i) |     |     |     |     |     | (β) |     |
| ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
where n ,n ,n ,n are the values for the links meeting at site i. I are the modified
| 1 2 | 3   | 4   |     |     |     |     |     | n l |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Besselfunctionsofthefirstkind,and
2π
|     |     |       | (cid:90) dθ |          |           |     |     |          |     |
| --- | --- | ----- | ----------- | -------- | --------- | --- | --- | -------- | --- |
|     | n   | ,n    |             | eiθ(n +n | −n −n )=δ |     |     |          |     |
|     | F   | 3 4 = |             | 1        | 2 3 4     | (n  | +n  | −n −n ), |     |
|     | n   | ,n    | 2π          |          |           | K   | 1 2 | 3 4      |     |
1 2
0
whereδ denotestheKroneckerdeltafunction.
K
At fixed β, I (β) decays fast and truncating the sum in Eq. (15) is a sensible approxima-
n
l
tion. The partition function of the XY model can thus be approximated by a tensor network
where the degree of freedom at each bond takes value in a finite set. In the language of
| AppendixA,thetensorateachsite |     |     |     | i wouldnowbe |     |     |     |     |     |
| ----------------------------- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- |
(cid:140)1/2
(cid:130) 4
(cid:89)
|     |     |     | (i)   | =         | (β) |     | n 3 ,n 4, |     |     |
| --- | --- | --- | ----- | --------- | --- | --- | --------- | --- | --- |
|     |     |     | A     |           | I n |     | F n ,n    |     |     |
|     |     |     | n 2 n | 4 n 1 n 3 | k   |     | 1 2       |     |     |
k=1
(β)to
andthecontractionoftheTNmadeupofthesetensorswouldgiveanapproximation Z(cid:101)
the partition function Z(β). Similarly, the marginal probability density of the spin at a site i,
π(β)(θ
| ),canbeapproximatedbyreplacingthetensoratsite |     |     |     |     |     |     | i with |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | ------ | --- | --- |
(cid:101) i
(cid:140)1/2
|     |     |       |           | (cid:130) 4 |       | eiθ(n | +n −n | −n ) |     |
| --- | --- | ----- | --------- | ----------- | ----- | ----- | ----- | ---- | --- |
|     |     |       |           | (cid:89)    |       |       | 1 2   | 3 4  |     |
|     |     | A (i) | (θ        | )=          | I (β) |       |       | ,    |     |
|     |     |       | i         |             | n     |       | 2π    |      |     |
|     |     | n 2 n | 4 n 1 n 3 |             | k     |       |       |      |     |
k=1
and normalizing the contraction to the approximate partition function previously obtained.
Using renormalisation to approximately contract tensor networks, and the inverse sampling
|     |     |     |     | ω′ ={θ′ | i ∈ | V}  |     |     |     |
| --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- |
method, a candidate configuration : can be drawn and accepted or rejected,
i
as we did for Ising models with Algorithm 1. A vector potential could be included [65, 66],
andothercontinuousvariablesystemsadmitasimilarconstruction[68].
On top of the bond dimension used for the renormalisation, the number of terms kept in
the series expansion of the transfer matrix in Eq. (15) is another parameter that governs the
accuracyofthecontraction. Asforthe3DIsingmodeldiscussedabove,atensornetworkwith
alowvalueforthisparametermaybeaccurateenoughtosamplefromandproposemovesfor
aMarkovchain,butnotpreciseenoughtocomputetheobservableswithasinglecontraction.
A detailed study of the XY model is beyond the scope of this paper. But we have made
preliminary computations that show acceptance rates comparable to those of the ferromag-
neticIsingmodel. Inordertoseehowcorrelatedtheproposedcollectivemovesare,wehave
19

|     |     |     |     | SciPostPhys. | 14,123(2023) |
| --- | --- | --- | --- | ------------ | ------------ |
22
| 0123456789101112131415 |     | 0123456789101112131415 |     |     |     |
| ---------------------- | --- | ---------------------- | --- | --- | --- |
| 0                      |     | 0                      |     |     |     |
| 1                      |     | 1                      |     |     |     |
| 2                      |     | 2                      |     |     |     |
| 3                      |     | 3                      |     |     | 1   |
| 4                      |     | 4                      |     |     |     |
| 5                      |     | 5                      |     |     |     |
| 6                      |     | 6                      |     |     |     |
| 7                      |     | 7                      |     |     | 2 2 |
−
| 8   |     | 8   |     |     |     |
| --- | --- | --- | --- | --- | --- |
| 9   |     | 9   |     |     |     |
| 10  |     | 10  |     |     |     |
| 11  |     | 11  |     |     |     |
4
| 12  |     | 12  |     |     | 2 − |
| --- | --- | --- | --- | --- | --- |
| 13  |     | 13  |     |     |     |
| 14  |     | 14  |     |     |     |
| 15  |     | 15  |     |     |     |
2 6
−
Figure 13: Mutual information in bits between the updates at different sites,
that is, in the changes of the angles after a sweep of the Markov chain,
| I(θ (t +1)−θ | (t) θ (t +1)−θ | (t)) |              |               |           |
| ------------ | -------------- | ---- | ------------ | ------------- | --------- |
|              | :              | for  | two schemes. | Left: a local | algorithm |
| i            | i j            | j    |              |               |           |
(a Metropolis single spin flip where the proposed local updates were chosen from
U(0,2π)). Right: TNMH. The numerical experiment was conducted on a homoge-
neousXYmodelwithnoexternalfieldona16×16latticeatatemperatureofT =0.5.
=
The bond dimension used in TNMH was D 20, and the number of terms kept in
theseriesofEq. (15)was N =4,whichgaveaTNwithalocaldimension d =9.
computed the mutual information between the updates at different sites of the TNMH algo-
rithmandcomparedittothatobtainedfromalocalalgorithm,Figure13. Theinstancechosen
16×16
for the comparison is the zero-field uniform XY model on a lattice at a temperature
T =0.5. The difference in the results is noteworthy, and demonstrates that the TNMH algo-
rithmisindeedcapableofproducingglobalcorrelatedupdates.
| Drawing configuration | differences |     |     |     |     |
| --------------------- | ----------- | --- | --- | --- | --- |
We now show that a TNMH scheme for the Ising model can be extended to deal with other
Forthesakeofconcreteness,wewillfocusontheλφ4model,
nearestneighbourhamiltonians.
definedonatwo-dimensionalsquarelatticeΛ=(V,E)bytheenergyfunction
λ
|     |          | (cid:88)  | (cid:88) 1 |      |     |
| --- | -------- | --------- | ---------- | ---- | --- |
|     | H({φ })= | (φ −φ )2+ | ( m2φ2+    | φ4), |     |
|     | i        | i j       |            | i i  |     |
|     |          |           | 2          | 4!   |     |
|     | 〈i,j〉∈E  |           | i∈V        |      |     |
where each local variable φ takes value in (cid:82). (See also Refs. [32, 69] for the use of tensor
i
networks in lattice field theories.) As usual, we are interested in sampling according to the
Boltzmanndistributionforsomefixedvalueβ. Wewillusethefollowingsimplelemma.
Lemma 1 Anyrealfunctionoftwobinaryvariables,B,canbeexpressedasanIsingmodelenergy
plussomeconstant:
|     | B(σ,σ′)=Jσσ′+hσ+h |     | ′σ′+K. |     |     |
| --- | ----------------- | --- | ------ | --- | --- |
(16)
′
Proof: (16)definesasystemoffourlinearequationsforthefourunknownsJ,h,h ,K,onefor
each assignment (σ,σ′). The determinant of the matrix of this system of equations does not
{B(σ,σ′)}
vanish but is equal to 16; a solution to (16) therefore exists for any 4-uple and is
unique.
| ω = {φ | ∈ Ω} |     |     |     |     |
| ------ | ---- | --- | --- | --- | --- |
Let i : i denote the current configuration. A Markov chain with collective
updates can be constructed using the TNMH presented for the Ising model in Section 2 if we
draw configuration changes. We proceed as follows. An integer m is drawn uniformly and
20

SciPostPhys. 14,123(2023)
randomly in {0,1,...,m }, where m is equal to 9, say. ∀i ∈V, we draw γ according to
max max i
aGaussiandistributionwithzeromeanandvarianceequalto10 −m. WithΓ ={γ :i∈V},we
i
constructthefunction:
(cid:88)
H ({σ }|ω,Γ)= (ψ (σ |φ ,γ )−ψ (σ |φ ,γ ))2
I i i i i i j j j j
〈i,j〉∈E
1(cid:88) λ (cid:88)
+ m2ψ (σ |φ,γ )2+ ψ (σ |φ ,γ )4,
i i i i i i i
2 4!
i∈V i∈V
where
1−σ 1+σ
ψ (σ |φ ,γ )= iφ + i(φ +γ ),
i i i i i i i
2 2
withσ ∈{−1,+1}∀i∈V. Bylemma1, H ({σ }|ω,Γ)canbeexpressedasanIsingHamilto-
i I i
nianforthevariables{σ }(plussomeirrelevantglobalconstant):
i
(cid:88) (cid:88)
H ({σ }|ω,Γ)=− h (ω,Γ)σ − J (ω,Γ)σ σ .
I i i i i,j i j
i∈V 〈i,j〉∈Γ
TheBoltzmanndistributionoftheIsingmodel H ,
I
π(β)({σ }|{θ },Γ)= e
−βH
I
({σ
i
}|{θ
i
},Γ)
,
I i i (cid:80)
e
−βH
I
({σ
j
}|{θ
i
},Γ)
{σ }
j
can generically not be sampled directly. But we can construct a tensor network approxima-
tion π(β)(·|ω,Γ) for it, as described in Section 2. Given Γ as defined above, let us define
(cid:101)
τ(Γ)={−γ :i∈V}. ThesequenceofinstructionslistedinAlgorithm2definesanirreducible
i
andreversibleMetropolis-HastingsMarkovchainthatachievescollectiveupdatesfortheλφ4
model.
Algorithm 2 Configurationdifferencecollectiveupdate
1: Drawanintegermu.a.r. in{0,...m }.
max
2: Draw|V|i.i.d. Gaussianswithzeromeanandvarianceequalto10 −m: Γ ={γ :i∈V}.
i
3: Draw{σ :i∈V}accordingtoπ(β)(·|ω,Γ).
i (cid:101)
4: Acceptthemove{φ :i∈V}→{φ +1+σ iγ}withprobability
i i 2 i
(cid:168) π(β)({σ}|ω,Γ) π(β)(ω′) (cid:171)
min 1, (cid:101)I i × .
π(β)({σ}|ω′,τ(Γ)) π(β)(ω)
(cid:101)I i
The idea of making configuration difference updates appeared in the study of the ferro-
magnetic XY model, for which the Wolff algorithm for the ferromagnetic Ising model can be
recycled[6]. Inprinciple,Algorithm2couldbeappliedtofrustratedsystems.
Aclassofsystemsforwhichwebelieveitcouldbeusefultodrawdifferencesofconfigura-
tionsasdescribedherearematrixmodels,suchasSU(d)latticegaugetheories[70]. Theaux-
iliaryHamiltonianrepresentingthepossiblechoicesforamovewouldnolongerbetwo-body
Ising. Still, it is not difficult to construct a tensor network representation for its Boltzmann
distribution,aswehavedonewhenstudyingtheXYmodel.
Triangular lattices
We now show how the construction presented in Section 2, specific to square lattices, can be
used as such to deal with a triangular lattice. Let us assume that we are interested in some
21

blabla
|     | SofyanIblisdir1 | andDavidP´erezGarc´ıa2,3 |     |     |     |     |     |
| --- | --------------- | ------------------------ | --- | --- | --- | --- | --- |
1Dpt.
F´ısicaQu`anticaiAstrof´ısica&InstitutdeCi`enciesdelCosmos,
UniversitatdeBarcelona,08028Barcelona,Spain
2Dpto. Ana´lisisMatema´ticoyMatema´ticaAplicada,
UniversidadComplutensedeMadrid,28040Madrid,Spain
3InstitutodeCienciasMatema´ticas,UniversidadAuto´nomadeMadrid,
28049Madrid,Spain
March5,2021
Abstract
blabla
|     |     |     |     |     |     | SciPostPhys. 14,123(2023) |     |
| --- | --- | --- | --- | --- | --- | ------------------------- | --- |
1 blabla
Figure1: Solidlines: interactiongraphofatriangularlattice(periodicbound-
Figure14:aLryecfotn:diItniotnesraasscumtieodnfogrsriampplhicitoyf).aGrteryiadnotgsu: elxatrraldaetgtrieceseosfyfrseetdeomm.ofCentre: Sameinter-
theextendedmodel.
actiongraphdecoratedwithextradegreesoffreedomlocatedonthediagonals(grey
dots). Right: SquarelatticeonwhichaHamiltonian H□ associatedwiththeoriginal
systemisdefined.
particularobservable X. Thatis,wewishtoestimate
1
1
|     | 〈X〉= |     | (cid:88) X(ω)e | −βH(ω) |     |     |     |
| --- | ---- | --- | -------------- | ------ | --- | --- | --- |
.
Z(β)
ω∈Ω
To this end, we construct an extended model, obtained by decorating the original lattice
with extra spins living on each diagonal link as shown on Fig. 14 (a) and (b). With each
particleoftheoriginalmodel,wewillassociatethenewextraspinlocatedsoutheasttoit. Let
→V
p:V denote the function that realises this association, where V denotes the set of
new new
newvertices. TheHamiltonianoftheextendedmodelreads
(cid:88)
|     | (ω  | |γ)=H(ω)−γ |     |     | σ σ   |     |      |
| --- | --- | ---------- | --- | --- | ----- | --- | ---- |
|     | H   |            |     |     | p(j), |     | (17) |
|     | ext | ext        |     |     | j     |     |      |
j∈V
forγ>0,whereω ∈Ω ={−1,+1}|V∪V | (β|γ)willdenoteitspartitionfunction.
|     |     |     | new . Z |     |     |     |     |
| --- | --- | --- | ------- | --- | --- | --- | --- |
| ext | ext |     |         | ext |     |     |     |
Proposition 1
|     |          |     | 1 (cid:88) | −βH   | (ω  | |γ)   |      |
| --- | -------- | --- | ---------- | ----- | --- | ----- | ---- |
|     | 〈X〉= lim |     |            | X(ω)e | ext | ext , | (18) |
|     | γ→∞      |     | (β|γ)      |       |     |       |      |
Z
ext ω
ext
wheneverβ and|V|arebothfinite.
Ω(0)∪Ω(1)∪...
Proof: The extended configuration space Ω can be decomposed as Ω =
|     |     |     | ext |     |     | ext ext | ext |
| --- | --- | --- | --- | --- | --- | ------- | --- |
∪Ω(|V|) ,whereΩ(m)
denotesthesubsetofallconfigurationssuchthatthereareexactlymsites
ext ext
∈V σ ̸=σ
j where p(j). This decompositioninduces another forthe partition functionof the
j
extendedmodelas
|V|
|     |        | (cid:88)        |     | (cid:88) |                |     |     |
| --- | ------ | --------------- | --- | -------- | -------------- | --- | --- |
| Z   | (β|γ)= | e −βH(ω)+βγ|V|+ |     |          | ζ e βγ(|V|−2m) | ,   |     |
| ext |        |                 |     |          | m              |     |     |
|     |        | ω∈Ω             |     | m=1      |                |     |     |
where the coefficients ζ are all finite and independentof γ. Similarly, the sum appearing in
m
ther.h.s. of(18)canbeexpressedas
|V|
|     | (cid:88) |               |     | (cid:88) |              |     |     |
| --- | -------- | ------------- | --- | -------- | ------------ | --- | --- |
|     | X(ω)e    | −βH(ω)+βγ|V|+ |     | ξ        | e βγ(|V|−2m) | ,   |     |
m
|     | ω∈Ω |     |     | m=1 |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
where the coefficients ξ are also finite and independent of γ. Finally, it is obvious that the
m
ratio
|     | (cid:80) X(ω)e | −βH(ω)+βγ|V|+(cid:80)|V| |     | ξ   | βγ(|V|−2m) |     |     |
| --- | -------------- | ------------------------ | --- | --- | ---------- | --- | --- |
|     |                |                          |     | m=1 | e          |     |     |
m
ω∈Ω
,
|     | (cid:80) e−βH(ω)+βγ|V|+(cid:80)|V| |     |     | ζ eβγ(|V|−2m) |     |     |     |
| --- | ---------------------------------- | --- | --- | ------------- | --- | --- | --- |
|     |                                    |     | m=1 | m             |     |     |     |
ω∈Ω
22

|     |     |     |     |     |     |     |     |     | SciPostPhys. | 14,123(2023) |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ------------ | --- |
tendsto〈X〉inthelimitwhereγtendstoinfinity.
A similar argument provides the following identity between the Boltzmann weight for a
configurationoftheextendedspaceω andtheBoltzmannweightofitsrestrictiontoΩ,ω:
ext
|     |     |     |       | e −βH (ω | |γ)   |            |       | e    | −βH(ω) |     |      |
| --- | --- | --- | ----- | -------- | ----- | ---------- | ----- | ---- | ------ | --- | ---- |
|     |     |     |       | ext      | ext = | (cid:89) δ | (σ ,σ | )    |        |     |      |
|     |     |     | l im  |          |       |            |       |      | .      |     | (19) |
|     |     |     | γ → ∞ | Z (β|γ)  |       |            | K j   | p(j) | Z(β)   |     |      |
ext
j∈V
Λ
The contribution of any site j of the original lattice to the numerator of the r.h.s. of (19)
reads
|     |     |     |      |         | (cid:0)β | (cid:0) | (cid:88) |     | (cid:1)(cid:1) |     |      |
| --- | --- | --- | ---- | ------- | -------- | ------- | -------- | --- | -------------- | --- | ---- |
|     |     |     | δ (σ | ,σ )exp |          | h σ     | +        | J σ | σ ,            |     | (20) |
|     |     |     | K    | j p(j)  |          | j       | j        | jk  | j k            |     |      |
k∈N(j)
N(j)
where denotes the neighbourhood of j. Because of the Kronecker delta, for any bipar-
tition of this neighbourhood N(j) = N ′(j)∪N ′′(j), (20) remains invariant if the sum in the
exponentialissubstitutedwith
|     |     |     |     | (cid:88) |         |          | (cid:88) |       |     |     |      |
| --- | --- | --- | --- | -------- | ------- | -------- | -------- | ----- | --- | --- | ---- |
|     |     |     |     | J        | σ       | σ +      |          | J σ σ | .   |     | (21) |
|     |     |     |     |          | jk p(j) | k        |          | jk j  | k   |     |      |
|     |     |     |     | k∈N′(j)  |         | k∈N′′(j) |          |       |     |     |      |
Assuming w.l.o.g. the boundary conditions represented on Fig. 14-left, we choose, for every
|     | ′(j) |     |     |     |     |     |     |     |     | ∀j ∈ Λ. |     |
| --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- |
site j, N to consist in the sites located east, south, and south east of j, (Edge
′(j),
and corner sites might require different choices of subsets N depending on the boundary
conditions.) ThischoiceresultsinasquarelatticehamiltonianH□ whosecouplingsareshown
onFig.15,andwhoseinteractiongraphisdisplayedonFig.14(c).
Letπ
(cid:101)□ denoteaprobabilitydistributionapproximatingtheBoltzmanndistributionassoci-
atedwith H□ throughtensornetworkrenormalisation. Todealwithatriangularlatticeusing
aTNMHcodeforasquarelattice,apossibilityisaMarkovchainwhere,ateachstep,acandi-
| dateconfigurationω′ |     |     | isdrawnaccordingtoπ |     |     |     |     |     |     |     |     |
| ------------------- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:101)□,andthemovefromthecurrentconfiguration
ext
ω
ext tothiscandidateisacceptedwithMetropolis-Hastingsprobability:
|     |     |     |     | (cid:26) | −βH(ω′) |     | π (ω       | )(cid:27) |     |     |     |
| --- | --- | --- | --- | -------- | ------- | --- | ---------- | --------- | --- | --- | --- |
|     |     |     |     |          | e       |     | (cid:101)□ | ext       |     |     |     |
|     |     |     |     | min      | 1,      | ×   |            | ,         |     |     |     |
|     |     |     |     |          | e−βH(ω) |     | π (ω′      | )         |     |     |     |
(cid:101)□
ext
| whereω(resp. |     | ω′ )denotestherestrictionofω |     |     |     |     | (resp. | ω′ )toΩ. |     |     |     |
| ------------ | --- | ---------------------------- | --- | --- | --- | --- | ------ | -------- | --- | --- | --- |
ext
ext
This mapping from a triangular lattice to a square lattice doubles the number of sites but
we stress that the bond dimension of the (square) tensor network associated is unchanged
=
and equal to that of the local degrees of freedom (d 2 for the Ising model). It would be
very interesting to see whether the argument can be extended to three dimensions, and for
examplemapabodycentredcubiclatticemodeltoasimplecubiclatticemodel.
A quantum analogue of the mapping exists: square PEPS can be used for a tri-
angular quantum spin Hamiltonian. The extended Hamiltonian (operator) now reads
|     | −γ(cid:80) |     |     |     |     |     |     | (cid:80) |     | −βH (ω |γ) |     |
| --- | ---------- | --- | --- | --- | --- | --- | --- | -------- | --- | ---------- | --- |
H = H σz σz . Proposition 1 still holds true if X(ω) e ext ext is sub-
| ext |     | j∈V | j p(j) |     |     |     |     |     | ω   |     |     |
| --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
ext
stituted with Tr Xe −βH ext. Expressing the trace in the basis of eigenstates of {σz} operators,
j
an analogue of the substitutions (20,21) holds true too. If for example, one wants a TNS
approximation of the ground state, one could alternate Trotter steps with applications of the
| projector|00〉〈00| |     |     | +|11〉〈11| |     |     |     |     |     |     |     |     |
| ----------------- | --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
z z oneachparticleoftheoriginallatticeanditspartner. Actually,
afurtherreductioncanbemade: onereadilychecksthattheinteractiongraphtransformation
shownonFig.14producesahexagonallatticewhenappliedtoasquarelattice. Therefore,in
principle,itshouldevenbepossibletostudytriangularlatticeswithhexagonalPEPS.
23

SciPostPhys. 14,123(2023)
J
h
J
∞ h
J v J d J v J d
Figure15: Couplingsinandaroundaplaquetteintheoriginalandextendedmodels
(left and right respectively). The new couplings produce a square lattice rotated by
aπ/4anglewithrespecttotheoriginallattice.
Figure16: Exampleofaconfigurationofharddisksinadiscretisedvolume. (Periodic
boundaryconditionsassumed.)
Hard spheres
Toclosethissection,weshowhowtensornetworkcontractionscanalsobeusedtoimplement
collective Monte Carlo updates for systems of hard spheres (or disks in two dimensions) [5].
We will combine three ideas for that purpose. The first is a discretisation of the domain that
contains the spheres. The second is a shift of perspective where a configuration will not so
much be regarded as a collection of locations for the spheres, but rather as the specification
for the state of each cell of the volume that contains them (occupied or empty). The third is
touseatensornetworktoencodepossiblechangesforeachcell.
Weconsiderasystemof N harddisksintwodimensionsconfinedinasquareareadiscre-
tised with a square lattice (M cells). Although this is not essential, we will assume periodic
boundaryconditionsinordertokeepthepresentationsimple. N isfixed,aswellasthelattice
spacing ε. All disks have identical radius. A configuration is said to be valid if (i) the centre
of each disk is pinned on the intersection of a vertical and a horizontal line of the lattice, (ii)
no cell contains bits of matter belonging to different disks. Fig. 16 is an example of a valid
configuration.
Ourgoalistosampleuniformlyamongstallvalidconfigurations. Forthat,wewilldesign
aMarkovchainofcollectiveupdateswhereeachdiskeitherstandsstillorismovedvertically
orhorizontallybyonelatticespacing. Aconfigurationchangemustcomplywiththefollowing
rules:
1. Adiskcannotbesplit.
2. Adiskcannotbecompressed.
3. Diskscannotoverlap,notevencompletely(conservationofparticlenumber).
We will assume the disks are distinguishable and we will associate a label {1,2,...,N} to
each of them, which is why each disk appears with a different colour in the illustration of
24

SciPostPhys. 14,123(2023)
′
ns
σ
e ′ P e
w j w ′
′s
n
Figure17: DiagrammaticrepresentationofthePEPStensorassociatedwitheachcell
j ofthelattice.
Fig. 16. We will denote S
0
the set of empty cells, and Sα the set of cells occupied by disk α,
1≤α≤N.
Given a valid configuration ω, we associate a tensor P with each cell j of the lattice, see
j
Fig. 17. The index σ of this tensor encodes the move that a bit of matter located at cell j
wouldundergo: ≡{0,−1,+1,−2,+2}for{stillness,displacementtotheleft,displacement
M
to the right, downwards displacement, upwards displacement} respectively. The role of the
w,e,n,s degrees of freedom of P is to communicate the chosen move at j to its neighbour
j
′ ′ ′ ′
cells; the indices w ,e ,n ,s provide the information about the moves made in neighbouring
cells to cell j. We want to assign values to these tensors {P } that guarantee moves can only
j
occurbetweenvalidconfigurations.
A.Initialisation. Foreachcell j, P (σ)w′,e′,n′,s′ =1∀σ,w ′ ,e ′ ,n ′ ,s ′ ,w,e,n,s∈ .
j w,e,n,s M
B. Empty cells. ∀j ∈ S , since there is no matter to be moved, we decree that
0
P (σ)w′,e′,n′,s′ =0, ∀w ′ ,e ′ ,n ′ ,s ′ ,w,e,n,s ifσ̸=0(holesdonotmove).
j w,e,n,s
C.Faithful move communication. ∀j, P (σ)w′,e′,n′,s′ =0unless w=e=n=s=σ.
j w,e,n,s
D. Rigidity. Let j,k denote two neighbouring cells covered by a same disk Sα,α̸=0. Let
us assume, say, that j is located left to k. We impose that P (σ)w′,e′,n′,s′ =0 if e̸=w ′ . Similar
j w,e,n,s
constraints are imposed on all other pairs of cells j,k covered by a same disk and such that
|j−k|=1.
E. Prevention of collisions. By definition, a collision has occurred between two disks α
andα′ ifandonlyiftwobitsofmatterbelongingtoαandα′
respectivelyarefoundinasame
cell. Therefore, it is necessary and sufficient to forbid all such events in order to prevent a
collision.
Ifthereisacollision,eitheronediskisimmobile,sayα,andα′
movesbyonecellto
overlapwithα(caseA),orbothαandα′
movetocausetheoverlap(caseB).
′
CaseAoccursifandonlyiftherearepairsofadjacentcellscandc inSαandSα′respectively
which content will occupy a same cell. To prevent the collision, it is sufficient to impose that
foreachsuchpair(c,c ′),thebitofmattercontainedinc ′ cannothopinc. Therearefoursuch
movestoprohibit;theyarerepresentedbythefourleftmostdrawingsofFig.18.
In case B, α and α′ either move along a same direction (case BI) or along perpendicular
′
directions (case BII). Case BI occurs if and only if there are pairs of cells c and c , separated
byonecell,inSα andSα′ respectively,whichcontentsaremovedclosertoeachotheralonga
commonline. Itisthusenoughtopreventtheeventsrepresentedbytherightmostdrawingsof
Fig.18. CaseBIIisdealtwithsimilarly,andresultsintheprohibitionoftheeventsrepresented
bythefourremainingdiagramsofFig.18.
Collisions where a bit of matter contained in a cell k ∈ Sα′ moves to its left, and lands
in a cell j ∈ Sα already occupied by a bit of matter that does not change its position, can be
prevented by imposing P (0)−1,e′,n′,s′ = 0 ∀e ′ ,n ′ ,s ′ . The other A prohibitions admit similar
j 0000
translations into constraints on the tensors, and the six B prohibitions can be enforced like-
wise. For example the prohibition of the move depicted on the diagram located rightmost
top of Fig. 18 translates into P (σ)+1,−1,n′,s′ = 0 ∀w,e,n,s,n ′ ,s ′ whenever the left and right
j w,e,n,s
25

-dnuob fo
modeerf
cidoirep(
fo
seerged
ecittal
|     |     |     |     |     |     | SciPostPhys. | 14,123(2023) |     |
| --- | --- | --- | --- | --- | --- | ------------ | ------------ | --- |
artxe
ralugnairt
|     |     |     | "   |     | #   | # " | :   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
stod
| • ? | • " |     | ⌃ ? |     | ? ⌃ | ? ⌃ ? |        |     |
| --- | --- | --- | --- | --- | --- | ----- | ------ | --- |
| ! ? | ? ? | !   | ?   |     | ? ! |       | a yerG |     |
fo

|     |     |     |     |     |     | ?   | hparg | 2   |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- |
.)yticilpmis
|     |     |     |     |     |     |     |             |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- |
| ?   | # • |     | ?   |     | ?   | ⌃   |             |     |
| • ? | ? ? |     | ? ⌃ |     | ⌃ ? | ? ! | noitcaretni |     |
|     |     |     | #   |     | "   |     |             |     |
rof
Figure 18: Forbidden moves in the discretised hard disks model. An asterisk in a
demussa
cellindicatepresenceofmatter,therhombussymbolstandsforacellthatcaneither
:
senil .ledom
be empty or filled. A dot on the side of a cell indicates no move, whereas an arrow
indicatesamovebyonelatticespacinganditsdirection.
diloS snoitidnoc
dednetxe
:2
erugiF
neighboursofcell j areoccupiedbydifferentdisks.
TheexactcontractionofalltensorsyieldsafunctionQ(σ ,...,σ |ω),whichvalueisequal yra eht
|     |     |     |     |     | 1   | M   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
to0ifthemove{σ ,...,σ }isforbiddenfromconfigurationω,and1otherwise. Wenotethat
| 1   | M   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
for a fixed assignment {σ ,...,σ }, Q(σ ,...,σ |ω) can be evaluated exactly. Ideally, we
|     | 1   | M   | 1   | M   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
wouldconstructaMetropolis-HastingsMarkovchainwherethemovesaresampledaccording
totheprior
|     |             |      |          | Q(σ ,...,σ | |ω)    |     |     |     |
| --- | ----------- | ---- | -------- | ---------- | ------ | --- | --- | --- |
|     | π (σ ,...,σ | |ω)= |          | 1          | M      | .   |     |     |
|     | id 1        | M    | (cid:80) | Q(τ        | ,...,τ | |ω) |     |     |
|     |             |      |          | {τ}        | 1 M    |     |     |     |
Aswedon’texpectthistobepossible,weproposetoapproximateπ
throughtensornetwork
id
renormalisation, as we did for Ising models. At fixed volume M, the computational cost for
constructingthetensorsscalesas1/ε2. Thebonddimensionofthetensornetworkisindepen-
dentofε. AsforIsingmodels,acceptanceratesshouldincreasewiththebonddimensionused
inthetensornetworkrenormalisation. Ifnecessary,acomplementarystrategytoincreaseac-
ceptance rates is to select a region at each Markov step, and impose that all disks outside of
it or touching its boundary remain fixed; such a region would vary from one time step to the
nextandmayevenbedisconnected.
Intwodimensions,thehardspheremodelisknowntoexhibitafluid-solidphasetransition
for a filling fraction η=πa2N/A≃0.7, where a is the radius of the disks, and Adenotes the
area of the domain that contains them [5] (A= Mε2 here). It would be very interesting to
seehowafinitevalueofεaffectsthisphasetransition.
Actually,becauseofthediscretisation,
the model considered here is, strictly speaking, not the hard sphere model discussed in [5],
for which the disks could in principle occupy any position in Euclidean space. It might be
ε→0
that the phase transition in the limit does not correspond to the transition point of the
hardspheremodeldefinedinEuclideanspace. Butitjustmightifadifferentlatticegeometry
is used. A similar phenomenon occurs in the study of fluids with cellular automata: square
latticesdonotrelatetotheNavier-Stokesequationwhereastriangularlatticesdo[71].
A construction similar to Fig. 17 should hold for hard spheres in three dimensions, and
webelievethatananaloguealsoexistsfordimer(anddimer-monomer)models. Inthislatter
case, the possibility to rotate dimers by a π/2 angle produces additional constraints on the
tensors.
26

SciPostPhys. 14,123(2023)
6 Discussion
The interplay between Monte Carlo and tensor network methods is a rich and vastly unex-
plored subject. While various previous works have reported on using Monte Carlo sampling
for tensor network contractions, we have here presented an analysis of the converse: a new
class of Markov chain Monte Carlo algorithms for many-body classical systems based on ten-
sornetworkrenormalisation. ThisclassbelongsinthefamilyofMetropolis-Hastingsschemes.
Our construction produces collective updates. It is also irreducible and reversible; as such,
asymptoticconvergencetowardsthetargetprobabilitydistributionisguaranteed. Weempha-
sizeitsuniversalnature: itworksthesameforanynearestneighbourHamiltonianwithfinite
localdegreesoffreedom.
We have benchmarked our scheme for a variety of instances of the two-dimensional Ising
model defined on a square lattice. For ferromagnets and antiferromagnets, very high accep-
tance rates have been observed for larger systems, even with modest values of the bond di-
mension. Besides, drops in acceptance rates have been shown to signal criticality. Looking at
equilibration and decorrelation times, the scheme compares extremely well with single spin
flip updates and Wolff algorithm. As expected, the scheme’s performance is lower for frus-
tratedanddisorderedinstancesthanfortheferro-andantiferromagnets. Still,ourresultsare
veryencouraging. Inparticular,fordisorderedinstances,equilibrationappearstobeoccurring
ordersofmagnitudefasterthanforstate-of-the-arttechniquessuchasparalleltemperingsup-
plemented with isoenergetic cluster moves, both when time is counted in Monte Carlo steps
andinseconds.
We have also demonstrated the potential of the method for three dimensional systems,
by testing it on ferromagnetic and antiferromagnetic instances. Also in this case, we have
observed faster equilibration as compared to Wolff algorithm and, remarkably, even with a
simple contraction strategy and small bond dimension, the scheme can be shown to remain
usableatnearcriticaltemperatures,whereasamuchmorecostlydirectTNcontractionresults
inconsiderableerrors.
We have used simple procedures to implement tensor network renormalisation, and we
have made no particular effort to write an efficient code. For these reasons, we believe the
results presented here could be substantially improved. It would also be very interesting to
studywhatcanbegainedbyusingotherrenormalisationschemesforapproximatecontractions
of tensor networks [19]. For example, schemes involving disentanglers would be a natural
option in this regard [22]. Also for future work is the study of how TNMH Markov chains
combinewithparalleltempering[72].
Amajoradvantageofourconstructionisitsversatility. Wehaveseenthatwithlittleextra
effort, a code valid for the Ising model on a square lattice can be used as such to construct a
collectiveupdateMarkovchaininothersettingssuchastheXYmodel,oratriangularlattice,
andthatTNMHcouldalsobeusedtostudygasesofhardspheres. Inprinciplelatticesystems
with long range interactions could also be considered. For instance, given an Ising Hamilto-
nian H where the interactions decay with the distance as a power law, one can associate an
auxiliary Hamiltonian Hϱ where all interactions within some range ϱ are identical to H, and
allinteractionsbeyondϱ havebeentruncated. Onecannextconstructatensornetworkprior
from this Hamiltonian Hϱ. Two parameters would now govern the Markov chain: the bond
dimensionandtherangeϱ. Wehavealsorestrictedourselvestoscalardegreesoffreedomin
thiswork. ButthediscussionheldinSection5showsthatTNMHsamplingshouldalsoapply
tomatrixmodels,inparticularlatticegaugetheories.
A natural variation of our work would be to depart from tensor network representations
and use a quantum device to prepare Gibbs states and estimate the probability to draw a
given configuration [73, 74]. Such a device would be called as an external subroutine in
27

SciPostPhys. 14,123(2023)
(classical) Metropolis-Hastings iterations. Just as our 3D computations have revealed that
inaccuratecontractionschemescouldstillbeusefulforsampling,itwouldbeveryinteresting
to investigate how much computational power such quantum devices retain when imperfect.
Theseideaswillbestudiedelsewhere.
Finally,itwouldbeinstructivetodevelopamathematicalperspectiveontheschemespre-
sented here. In particular, we believe it would be meaningful to identify a non-trivial model
for which the mixing time associated with our TNMH scheme could be upper bounded, e.g.
usingalog-Sobolevinequality[75]. Itwouldbeinsightfultoestablishthedependenceofthe
logSobolevconstantwiththebonddimension.
Acknowledgements
We thank J.I. Cirac and F. Verstraete for fruitful discussions. This work was partly supported
by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Ger-
many’s Excellence Strategy – EXC-2111 – 390814868, and by the European Union through
the ERC grant GAPS (Grant no. 648913), by Ministerio de Ciencia, Innovación y Universi-
dades (Spain) (grant no. PGC2018-095862-B-C21, ‘Tecnologías cuánticas teóricas’, grant no.
PID2020-113523GB-I00, ‘Análisis Matemático y Teoría de Información Cuántica’, grant no.
MTM2017-88385-P,grantno. SEV-2015-0554,andgrantno. CEX2019-000918-M,‘Maríaade
Maeztu’), by Generalitat de Catalunya (Spain), SGR 1761, and from the European Union Re-
gionalDevelopmentFundwithintheERDFOperationalProgramofCatalunya(Spain)(project
QUASICAT/QuantumCat, ref. 001- P-001644) and by Comunidad de Madrid (Spain) (grant
QUITEMAD-CM,ref. S2018/TCS-4342).
A MPS renormalisation
We here review the relation between tensor networks and partition function [13, 14, 15, 19,
21]. The setup is a slight generalization of that of Section 2. That is, we consider a nearest
neighbourclassicalHamiltonian
(cid:88)
H(ω)= ϕ (σ ,σ ),
ij i j
〈i,j〉
on a lattice Λ = (V,E), where the local variables σ now take value in any finite set, which
i
size we are going to denote d. For the sake of simplicity, and without loss of generality, we
will again only consider squares lattices, and first limit ourselves to two-dimensional systems
fornow. Atfixedinversetemperatureβ,thepartitionfunctioncanbeexpressedas
(cid:88) (cid:89)
Z(β)= W (σ ,σ ), (A.1)
ij i j
ω∈Ω〈i,j〉∈E
where W is a d ×d matrix, whose entries represent all possible contributions of the bond
ij
〈i,j〉 to the Boltzmann weight of the model, i.e. W (σ,σ′)=e −βϕ ij (σ,σ′) . As an example, for
ij
theIsingmodelwithoutexternalmagneticfield,theenergyassociatedwithagivenbond〈i,j〉
readsϕ (σ,σ′)=−J σσ′ ,andthe2×2matrixW is
ij ij ij
(cid:18)
e
βJ
ij e
−βJ
ij
(cid:19)
W = . (A.2)
ij e −βJ ij e βJ ij
We will use the diagrammatical notation in which a tensor is represented by a vertex or a
small shape, with as many legs sticking out as there are indices; and where joining two lines
28

|     |     |     |     | SciPostPhys. | 14,123(2023) |     |
| --- | --- | --- | --- | ------------ | ------------ | --- |
representsacontractionofthecorrespondingindices. Forexample,amatrixW isrepresented
ij
asfollows,
,
Z(β)canbeexpressedasatensornetworkifweshiftfromadescriptionintermsofmatrices
associated with the bonds of the lattice (A.1) to a description in terms of tensors associated
with its vertices. Let us consider some vertex i with four neighbours and let e(i) denote the
| vertextoitsright. | WedecomposeW | i,e(i) as: |     |     |     |     |
| ----------------- | ------------ | ---------- | --- | --- | --- | --- |
d
(cid:88)
|     |          | (σ,σ′)= | (σ,µ)R | (µ,σ′). |     |     |
| --- | -------- | ------- | ------ | ------- | --- | --- |
|     | W i,e(i) |         | L      | e(i)    |     |     |
i
µ=1
Graphically,
=
.
This can be achieved e.g. through a sing(cid:112)ular value de(cid:112)composition
|     |     | †   |     |     |     | †   |
| --- | --- | --- | --- | --- | --- | --- |
(SVD) W = U Σ ×V , and by setting L = U Σ i,e(i), R = Σ i,e(i)V .
| i,e(i) | i,e(i) i,e(i) i | ,e(i) | i   | i,e(i) | e(i) | i ,e(i) |
| ------ | --------------- | ----- | --- | ------ | ---- | ------- |
Similarly, if n(i),w(i),s(i) denote vertices located above, to the left, and below i respectively,
threeadditionalSVDprovidethedecompositions
d
(cid:88)
|     |          | (σ,σ′)= | (σ,ν)R | (ν,σ′), |     |     |
| --- | -------- | ------- | ------ | ------- | --- | --- |
|     | W w(i),i |         | L w(i) | i       |     |     |
ν=1
d
|     |        | (σ,σ′)= | (cid:88) | (ρ,σ′), |     |     |
| --- | ------ | ------- | -------- | ------- | --- | --- |
|     | W      |         | B (σ,ρ)T |         |     |     |
|     | i,n(i) |         | i        | n(i)    |     |     |
ρ=1
(cid:88) d
|     |          | (σ,σ′)= | (σ,τ)T | (τ,σ′). |     |     |
| --- | -------- | ------- | ------ | ------- | --- | --- |
|     | W s(i),i |         | B s(i) |         |     |     |
i
τ=1
(i)(σ)
We associate a 4-index tensor A with each site i having four neighbours and each spin
valueσ,whosecomponentsare
d
|     | (i)  | (cid:88) |        |               |     |       |
| --- | ---- | -------- | ------ | ------------- | --- | ----- |
|     | A =  | L (σ,µ)R | (ν,σ)B | (σ,ρ)T (τ,σ). |     | (A.3) |
|     | µνρτ | i        | i      | i i           |     |       |
σ=1
| Indiagrammaticnotation,Eq. | A.3reads |     |     |     |     |     |
| -------------------------- | -------- | --- | --- | --- | --- | --- |
=
.
For a system with open boundary conditions, vertices with only three or two neighbours are
dealtwithlikewise. Withthesetensors,thepartitionfunctioncanbeexpressedas
|     |     | Z(β)= | ({A (i)}), |     |     |     |
| --- | --- | ----- | ---------- | --- | --- | --- |
(A.4)
C
where ({A (i)}) denotes the contraction of all the tensors associated with all sites. The entire
C
processfrom(A.1)to(A.4)isillustratedonFigure19fora4×4lattice.
29

|     |     |     |     |     |     |     | SciPostPhys. | 14,123(2023) |     |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------------ | --- |
|     |     |     | (a) |     |     |     | (b)          |              |     |
(d)
(c)
Figure19: GraphicaldepictionoftheconstructionoftheTNassociatedwiththepar-
titionfunctionofanearestneighbourclassicalHamiltonian(4×4latticeinthisillus-
tration). (a) We start with a labelling of the vertices of the lattice in consideration.
(b) Diagrammatic representation of the Boltzmann weights (red circles) associated
with each edge; their contraction yields the partition function. (c) and (d) Singular
valuedecompositionofeachW matrix,andregroupingintotensorsassociatedwith
eachvertexofthelattice.
Similarly,onecanconstructaTNrepresentationofthepartitionfunctionwithsomefixed
|     |     |     |     | Z(β|σ | =s). |     |     |     |     |
| --- | --- | --- | --- | ----- | ---- | --- | --- | --- | --- |
valuesforthedegreeoffreedomatsitei, Itisforinstancesufficientthatforeach
i
|           |                     |     | (σ,σ′) |      | ( s)(σ,σ′)=δ |     | (σ,σ′). |           |        |
| --------- | ------------------- | --- | ------ | ---- | ------------ | --- | ------- | --------- | ------ |
| neighbour | of i, j, we replace |     | W      | with | W            |     | s,σW    | The ratio | of the |
|           |                     |     | i,j    |      | i j          |     | i,j     |           |        |
π(β)(s)
two quantities, Z(β|σ = s)/Z(β), would exactly be the marginal probability of that
i
i
spin being in state black s, that is, a ratio of two TN contractions. Similarly, one can express
| anyconditionalprobabilityπ( |     |         | β)(σ |          |                               |         |     |     |       |
| --------------------------- | --- | ------- | ---- | -------- | ----------------------------- | ------- | --- | --- | ----- |
|                             |     |         | |s   | ...s     | )asaratiooftwoTNcontractions: |         |     |     |       |
|                             |     |         | k k  | 1 k−1    |                               |         |     |     |       |
|                             |     |         |      |          | Z (β |s                       | . . . s | σ ) |     |       |
|                             |     | π( β)(σ | |s   | )=       | 1                             | k − 1   | k   |     |       |
|                             |     |         |      | ...s k−1 |                               |         | .   |     | (A.5) |
|                             |     | k       | k 1  |          | Z (β | s                      | . . . s | )   |     |       |
|                             |     |         |      |          |                               | 1 k −   | 1   |     |       |
As explained in Section 2, if one were able to evaluate TN contractions exactly, one would
have a means to sample according to the Boltzmann distribution exactly. In general, it is
onlypossibletocomputeapproximationstothecontractionsappearingintheratio(A.5)and
|     |     |     | π(β) | π(β) |     |     |     |     |     |
| --- | --- | --- | ---- | ---- | --- | --- | --- | --- | --- |
as a result, get an approximation to . Instead of using these approximations for
(cid:101)k
k
directsamplingwithsystematicerrors,onecanusethemasapriorforareversibleMetropolis-
HastingsMarkovchain. TheimpossiblitytocarryoutexactTNcontractionthentranslatesinto
morecontrollablestatisticalerrors.
For the approximate contraction of an L × L lattice, we have used one of the simplest
schemes available [15, 10]. We define |top〉 to be the tensor resulting from contracting all
30

|     |     |     |     |     |     | SciPostPhys. | 14,123(2023) |
| --- | --- | --- | --- | --- | --- | ------------ | ------------ |
Figure 20: Graphical depiction of the process of contracting a two-dimensional lat-
(k−1)〉.
tice. (a)Approximationtothecontractionofthe k−1topmostrows,|partial
(b)Theapproximationto|partial (k)〉isconstructedbyapplyingtheMPOassociated
| k,TM    |                                                |     |     |     |     |     | k−1first |
| ------- | ---------------------------------------------- | --- | --- | --- | --- | --- | -------- |
| withrow | ,ontheMPSobtainedbyapproximatecontractionofthe |     |     |     |     |     |          |
k
rows,|partial (k−1)〉. TheresultofthatMPO-MPSmultiplicationisaMPSwithalarger
bond dimension. (c) Using standard MPS techniques, the product can be approxi-
matedbyaMPSwithlowerbonddimension.
the top row tensors along horizontal edges; the remaining free indices after this contraction
are legs pointing downward. Similarly, we will call transfer matrix the tensor resulting from
a contraction of the tensors along a horizontal bulk row; the transfer matrix resulting from
|                            |     |     |                 |     | ,2<k< | Finally,inanalogyto|top〉, |     |
| -------------------------- | --- | --- | --------------- | --- | ----- | ------------------------- | --- |
| contractingthetensorsofrow |     | k   | willbedenotedTM |     |       | L.                        |     |
k
wewilldenote|bot〉thecontractionofbottommosttensors.
Withthesenotations,thepartition
functioncanbeexpressedas
|     |     | Z(β)=〈bot|TM |     | ...TM | |top〉. |     | (A.6) |
| --- | --- | ------------ | --- | ----- | ------ | --- | ----- |
|     |     |              |     | L−1   | 2      |     |       |
Both|top〉and〈bot|arematrixproductstates(MPS),whereasthetransfermatricesTM
k are
matrixproductoperators(MPO),allwithabonddimensionanda‘physical’dimensionequal
Z(β)
to d; their length is equal to L. Our approximation of is obtained by estimating the rhs
|                        |     |            | |partial | (1)〉 ≡ |top〉, |     | ∈ {2...L −1}, |           |
| ---------------------- | --- | ---------- | -------- | ------------- | --- | ------------- | --------- |
| of (A.6) sequentially. | We  | initialise |          |               | and | for k         | we define |
|partial (k)〉tobeanMPSapproximationtoTM |partial (k−1)〉obtainedbymatrixproductstate
k
(L−1)〉.
renormalisation,seeFig. 20. Z(β)isfinallyapproximatedwith〈bot|partial Thecutoff
parameterDsetstheaccuracyoftheapproximation. Therearemanymethodsavailableforthe
renormalisation. Throughoutthiswork,wehavemostlyusedtheschemebasedonsuccessive
SVD[11]. Two-sitevariationalcompressionhasbeenusedtoexploreequilibrationofthetwo
dimensionalIsingmodelwithGaussiandisorder[11].
The same method allows to approximate the partition function of a system where some
spins have been set to definite values, Z(cid:101) (β|σ ...σ ). The only difference is that for such a
1 k
| site i withspinvalueσ | ,thetensor(A.3)isreplacedwith |     |     |     |     |     |     |
| --------------------- | ----------------------------- | --- | --- | --- | --- | --- | --- |
i
|     |     | L (σ | ,µ)R (ν,σ | )B (σ ,ρ)T | (τ,σ | ).  | (A.7) |
| --- | --- | ---- | --------- | ---------- | ---- | --- | ----- |
|     |     | i    | i i       | i i i      | i    | i   |       |
Asclaimedinsection2,anapproximateBoltzmannweightπ(ω)canbeevaluatedsince,using
(cid:101)
Bayestheorem,thisprobabilitycanbeexpressedas
|     |     | Z(cid:101) (β|σ | )       | Z(cid:101) (β|σ | ...σ | )   |     |
| --- | --- | --------------- | ------- | --------------- | ---- | --- | --- |
|     |     |                 | 1 ×...× |                 | 1    | n   |     |
.
|     |     |            | (β) | (β|σ       | ...σ  | )   |     |
| --- | --- | ---------- | --- | ---------- | ----- | --- | --- |
|     |     | Z(cid:101) |     | Z(cid:101) | 1 n−1 |     |     |
31

SciPostPhys. 14,123(2023)
Figure 21: Renormalisation of a black column of a PEPS. For the truncation of the
bond dimension of a given a column, its tensors are singled out. The physical bond
dimension and the bond dimension that connects these tensors to other columns,
drawn in red in this diagram, are treated as the physical dimension of an auxiliary
MPS. The bond dimension of that MPS is reduced using a standard truncation algo-
rithm. The resulting tensors of the obtained MPS with lower bond dimension are
inserted back in the PEPS. While this procedure has no guarantee of optimality, it is
computationallycheapandworkswellinpractice.
Figure 22: Renormalisation of a PEPS used to apply the TNMH algorithm to three-
dimensional systems. The bond dimension is first reduced along horizontal bonds,
nextalongverticalbonds.
Two remarks are in order. First, if the tensors A
(i)
are well conditioned and if D is high
enough, the approximations to partition functions we construct will be strictly positive. So
will then be the approximated probabilities (7), and the TNMH is irreducible. Second, if the
tensors{A (i)},theMPS|top〉,|bot〉andthetransfermatricesTM arestored,aTNMHupdate
k
of the whole lattice can be performed at a computational cost that scales linearly with the
latticesize.
Plaquetteinteractionscanbedealtwithsimilarly. Usingsingularvaluedecompositionfor
the Boltzmann weights and regrouping all the matrices relating to a given site, one obtains a
(π/4 rotated) square lattice for the partition function. Bayes formula can thus again be used
forsampling.
Wehavedealtwiththree-dimensionalmodelsinasimilarfashion. Assumingan L×L×L
lattice, the identity (A.4) can again be obtained after sequence of SVD; A
(i)
is now a six-leg
tensor (for a bulk spin). (A.6) is also still valid, but |top〉,|bot〉 are now projected entangled
pair states (PEPS) instead of MPS, and the transfer matrices TM projected entangled pair
k
operators (PEPOs) instead of MPOs. Just as in two dimensions, without any cutoff, the bond
dimension of TM L−1 ...TM 2 |top〉 would generically grow exponentially with L, and renor-
malisationisinorder. Thereexistsaplethoraofmethodstocontractthreedimensionaltensor
32

SciPostPhys. 14,123(2023)
L R
Figure 23: Cylindrical boundary conditions obtained by identification from a rect-
angle. L and R denote lines of spins alternating frozen in order to be able to use a
samplingschemedesignedforopenboundaryconditions.
networks [76, 21]. We have not aimed at optimality and have opted for simplicity. Again,
denoting|partial 〉theapproximatecontractionofthefirst k layersoftheTN,thecoreofthe
k
renormalisationconsistsinconstructingaPEPSapproximation|partial 〉forthecontraction
k+1
TM k+1 |partial k 〉. WhenaPEPOissuperimposedonaPEPS,theresultingstateisaPEPSwith
alargerbonddimension.
ThebonddimensionofTM
k+1
|partial
k
〉hasbeenreducedbyrecyclingthetechniqueused
to compress the bond dimension of MPS. First, an index reshuffling allows us to regard each
column of the PEPS TM k+1 |partial k 〉 as an MPS, with an effective physical index at each site
given by lumping the original physical index of the PEPS with the horizontal virtual indices
atthatsite. ThevirtualbondsofthatMPSaretheverticalvirtualbondsofthecorresponding
column of the PEPO. These ‘thick’ bonds are compressed (or renormalised) as before. This
compression alongcolumns is illustratedon Fig. 21. Theresulting tensors from thecompres-
sionaretheninsertedbackintothePEPS,andthesamecompressionisnextperformedonthe
horizontal bonds of the PEPS. See Fig. 22 for a depiction of this PEPS renormalisation.Two
parameters now govern the accuracy of the approximation: the bond dimension of the PEPS
|partial 〉, D,andthecutofffortheapproximatecontractionoftworowsofaPEPS,χ [76].
k
B Arbitrary boundary conditions
Although we have focused on systems with open boundary conditions, the Markov chain (4)
allows us to deal with any topology that can be obtained from a rectangle by appropriate
identifications. Letusshowhowwiththesimpleexampleofacylinder. Ifwemakeanupdate
wherewedecidetoleaveacolumnofspinsunchanged,e.g. thedashedcolumn’L’ofFig.23,
we will effectively be considering a model with open boundary conditions, where the spins
in the neighbourhood of the frozen line are subjected to a local extra magnetic field. Such a
modelcanbesampledasbefore. Inordertomakesureallspinsarerefreshed,thecutoffrozen
spins alternates between the opposite lines depicted as ’L’ and ’R’ respectively. Alternatively,
thelinesofspinswherewechoosetocutoursystemcanbechosenrandomly.
We have implemented this adaptation of an OBC TNMH code in order to study the equi-
libration of the two dimensional gaussian spin glass studied in the case of periodic boundary
conditions. Thishasallowedustocomparedirectlyourresultswiththestate-of-the-artresults
of[56]. OurfindingsaresummarizedinTable2.
As can be appreciated, this adaptation of Algorithm 1 yields equilibration in a number of
stepssignificantlylowerthanforPTorPT+ICM,asforthecaseofopenboundaryconditions
discussedinthemaintext. NoticethatauxiliaryMetropolisspinflipsarenownolongerneeded
tohelpintheequilibrationofsomeoftheconfigurationsofsomeofthedisorderrealizations.
33

SciPostPhys. 14,123(2023)
Table2: Firstrow: targetvalueof∆. Secondandthirdrow: eachentryrepresentsa
lower bound on the number of Monte Carlo sweeps necessary to decrease ∆ below
thevalueindicatedinthesamecolumnforparalleltempering(PT)andparalleltem-
peringplusisoenergeticclustermoves(PT+ICM)(datareadoffFig.2ofRef.[56]).
Fourthrow: UpperboundsonthenumberofTNMHiterationsnecessaryforthesame
purpose. The setting considered is identical to Ref. [56] (periodic boundary condi-
tions).
∆ 0.25 0.15 0.05 0.025
PT 221 222 223 224
PT+ICM - - 213 214
TNMH 14 21 40 56
ThereasonforthisisthatbyfreezingadifferentportionofthespinsateachTNMHiteration,
one is now dealing with a different effective current configuration for a different effective
OBC hamiltonian, at each TNMH iteration. Even though an effective current configuration
may occasionally suffer from the ill-conditioning issue described in Section 3, it will not as
easilystallourMarkovchainthankstotheselectionofadifferentcutateachiteration.
References
[1] N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller and E. Teller, Equa-
tion of state calculations by fast computing machines, J. Chem. Phys. 21, 1087 (1953),
doi:10.1063/1.1699114.
[2] B.Edegger,V.N.MuthukumarandC.Gros,Gutzwiller-RVBtheoryofhigh-temperaturesu-
perconductivity: Resultsfromrenormalizedmean-fieldtheoryandvariationalMonteCarlo
calculations,Adv.Phys.56,927(2007),doi:10.1080/00018730701627707.
[3] B. L. Hammond, Monte Carlo methods in ab initio quantum chemistry, World Scientific,
Singapore,ISBN9789810203214(1994),doi:10.1142/1170.
[4] I. Montvay and G. Münster, Quantum fields on a lattice, Cambridge University Press,
Cambridge,UK,ISBN9780511470783(1994),doi:10.1017/CBO9780511470783.
[5] W. Krauth, Statistical mechanics: Algorithms and computations, Oxford University Press,
Oxford,UK,ISBN9780198515364(2006).
[6] D. Landau and K. Binder, A guide to Monte Carlo simulations in statistical
physics, Cambridge University Press, Cambridge, UK, ISBN 9781139696463 (2005),
doi:10.1017/CBO9781139696463.
[7] R. H. Swendsen and J.-S. Wang, Nonuniversal critical dynamics in Monte Carlo simula-
tions,Phys.Rev.Lett.58,86(1987),doi:10.1103/PhysRevLett.58.86.
[8] U.Wolff,CollectiveMonteCarloupdatingforspinsystems,Phys.Rev.Lett.62,361(1989),
doi:10.1103/PhysRevLett.62.361.
[9] N. Schuch, D. Pérez-García and I. Cirac, Classifying quantum phases using matrix
product states and projected entangled pair states, Phys. Rev. B 84, 165139 (2011),
doi:10.1103/PhysRevB.84.165139.
34

SciPostPhys. 14,123(2023)
[10] U. Schollwöck, The density-matrix renormalization group in the age of matrix product
states,Ann.Phys.326,1(2011),doi:10.1016/j.aop.2010.09.012.
[11] R. Orús, A practical introduction to tensor networks: Matrix product states and projected
entangledpairstates,Ann.Phys.349,117(2011),doi:10.1016/j.aop.2014.06.013.
[12] T.Nishino,Densitymatrixrenormalizationgroupmethodfor2Dclassicalmodels,J.Phys.
Soc.Jpn.64,3598(1995),doi:10.1143/JPSJ.64.3598.
[13] T.NishinoandK.Okunishi,Cornertransfermatrixrenormalizationgroupmethod,J.Phys.
Soc.Jpn.65,891(1996),doi:10.1143/JPSJ.65.891.
[14] T.NishinoandK.Okunishi,Cornertransfermatrixalgorithmforclassicalrenormalization
group,J.Phys.Soc.Jpn.66,3040(1997),doi:10.1143/JPSJ.66.3040.
[15] V. Murg, F. Verstraete and J. I. Cirac, Efficient evaluation of partition functions
of inhomogeneous many-body spin systems, Phys. Rev. Lett. 95, 057206 (2005),
doi:10.1103/physrevlett.95.057206.
[16] N. Schuch, M. M. Wolf, F. Verstraete and J. I. Cirac, Computational com-
plexity of projected entangled pair states, Phys. Rev. Lett. 98, 140506 (2007),
doi:10.1103/PhysRevLett.98.140506.
[17] I.AradandZ.Landau,Quantumcomputationandtheevaluationoftensornetworks,SIAM
J.Comput.39,3089(2010),doi:10.1137/080739379.
[18] F. Verstraete, M. M. Wolf, D. Perez-Garcia and J. I. Cirac, Criticality, the area law, and
the computational power of projected entangled pair states, Phys. Rev. Lett. 96, 220601
(2006),doi:10.1103/physrevlett.96.220601.
[19] M. Levin and C. P. Nave, Tensor renormalization group approach to two-
dimensional classical lattice models, Phys. Rev. Lett. 99, 120601 (2007),
doi:10.1103/PhysRevLett.99.120601.
[20] Z. Y. Xie, H. C. Jiang, Q. N. Chen, Z. Y. Weng and T. Xiang, Second
renormalization of tensor-network states, Phys. Rev. Lett. 103, 160601 (2009),
doi:10.1103/PhysRevLett.103.160601.
[21] Z. Y. Xie, J. Chen, M. P. Qin, J. W. Zhu, L. P. Yang and T. Xiang, Coarse-graining renor-
malizationbyhigher-ordersingularvaluedecomposition,Phys.Rev.B86,045139(2012),
doi:10.1103/PhysRevB.86.045139.
[22] G. Evenbly and G. Vidal, Tensor network renormalization, Phys. Rev. Lett. 115, 180405
(2015),doi:10.1103/PhysRevLett.115.180405.
[23] S. Yang, Z.-C. Gu and X.-G. Wen, Loop optimization for tensor network renormalization,
Phys.Rev.Lett.118,110504(2017),doi:10.1103/PhysRevLett.118.110504.
[24] A. J. Ferris, Unbiased Monte Carlo for the age of tensor networks, (arXiv preprint)
doi:10.48550/arXiv.1507.00767.
[25] F.Pan,P.Zhou,S.LiandP.Zhang,Contractingarbitrarytensornetworks: Generalapprox-
imate algorithm and applications in graphical models and quantum circuit simulations,
Phys.Rev.Lett.125,060503(2020),doi:10.1103/PhysRevLett.125.060503.
35

SciPostPhys. 14,123(2023)
[26] Z.-C. Gu and X.-G. Wen, Tensor-entanglement-filtering renormalization approach
and symmetry-protected topological order, Phys. Rev. B 80, 155131 (2009),
doi:10.1103/PhysRevB.80.155131.
[27] Q. N. Chen, M. P. Qin, J. Chen, Z. C. Wei, H. H. Zhao, B. Normand and T. Xiang, Partial
order and finite-temperature phase transitions in Potts models on irregular lattices, Phys.
Rev.Lett.107,165701(2011),doi:10.1103/PhysRevLett.107.165701.
[28] Y.ShimizuandY.Kuramashi,CriticalbehaviorofthelatticeSchwingermodelwithatopo-
logicaltermatθ =πusingtheGrassmanntensorrenormalizationgroup,Phys.Rev.D90,
074503(2014),doi:10.1103/PhysRevD.90.074503.
[29] J. F. Yu et al., Tensor renormalization group study of classical XY model on the square
lattice,Phys.Rev.E89,013308(2014),doi:10.1103/PhysRevE.89.013308.
[30] Y. Shimzu, Tensor renormalization group approach to a lattice boson model, Mod. Phys.
Lett.A27,1250035(2012),doi:10.1142/S0217732312500356.
[31] J. F. Unmuth-Yockey, Y. Meurice, J. Osborn and H. Zou, Tensor renormalization group
studyofthe2dO(3)model,Proc.Sci.214,325(2014),doi:10.22323/1.214.0325.
[32] M.Campos,G.SierraandE.López,Tensorrenormalizationgroupinbosonicfieldtheory,
Phys.Rev.B100,195106(2019),doi:10.1103/PhysRevB.100.195106.
[33] S. Wang, Z.-Y. Xie, J. Chen, B. Normand and T. Xiang, Phase transitions of ferromag-
netic Potts models on the simple cubic lattice, Chin. Phys. Lett. 31, 070503 (2014),
doi:10.1088/0256-307x/31/7/070503.
[34] L.Vanderstraeten,B.VanheckeandF.Verstraete,Residualentropiesforthree-dimensional
frustrated spin systems with tensor networks, Phys. Rev. E 98, 042145 (2018),
doi:10.1103/physreve.98.042145.
[35] L. A. Goldberg and M. Jerrum, The complexity of ferromagnetic Ising with local fields,
Combinator.Probab.Comp.16,43(2006),doi:10.1017/S096354830600767X.
[36] A. Galanis, D. Štefankoviˇc and E. Vigoda, Inapproximability of the partition function for
the antiferromagnetic Ising and hard-core models, Combinator. Probab. Comp. 25, 500
(2016),doi:10.1017/s0963548315000401.
[37] A. W. Sandvik and G. Vidal, Variational quantum Monte Carlo simulations with tensor-
networkstates,Phys.Rev.Lett.99,220602(2007),doi:10.1103/PhysRevLett.99.220602.
[38] N. Schuch, M. M. Wolf, F. Verstraete and J. I. Cirac, Simulation of quantum many-body
systemswithstringsofoperatorsandMonteCarlotensorcontractions,Phys.Rev.Lett.100,
040501(2008),doi:10.1103/PhysRevLett.100.040501.
[39] A. J. Ferris and G. Vidal, Perfect sampling with unitary tensor networks, Phys. Rev. B 85,
165146(2012),doi:10.1103/PhysRevB.85.165146.
[40] Y. Meurice, Y. Liu, J. Unmuth-Yockey, L.-P. Yang and H. Zou, Sampling versus blocking,
Proc.Sci.214,319(2015),doi:10.22323/1.214.0319.
[41] L.-P. Yang, Y. Liu, H. Zou, Z. Y. Xie and Y. Meurice, Fine structure of the
entanglement entropy in the O(2) model, Phys. Rev. E 93, 012138 (2016),
doi:10.1103/PhysRevE.93.012138.
36

SciPostPhys. 14,123(2023)
[42] S. R. White, Minimally entangled typical quantum states at finite temperature, Phys. Rev.
Lett.102,190601(2009),doi:10.1103/PhysRevLett.102.190601.
[43] E.M.StoudenmireandS.R.White,Minimallyentangledtypicalthermalstatealgorithms,
NewJ.Phys.12,055026(2010),doi:10.1088/1367-2630/12/5/055026.
[44] M. Berta, F. G. S. L. Brandão, J. Haegeman, V. B. Scholz and F. Verstraete, Thermal
states as convex combinations of matrix product states, Phys. Rev. B 98, 235154 (2018),
doi:10.1103/PhysRevB.98.235154.
[45] K. Ueda, R. Otani, Y. Nishio, A. Gendiar and T. Nishino, Snapshot observation for 2D
classicallatticemodelsbycornertransfermatrixrenormalizationgroup,J.Phys.Soc.Jpn.
74,111(2005),doi:10.1143/JPSJS.74S.111.
[46] M. M. Rams, M. Mohseni, D. Eppens, K. Jałowiecki and B. Gardas, Approximate opti-
mization, sampling, and spin-glass droplet discovery with tensor networks, Phys. Rev. E
104,025308(2021),doi:10.1103/PhysRevE.104.025308.
[47] W.K.Hastings,MonteCarlosamplingmethodsusingMarkovchainsandtheirapplications,
Biometrika57,97(1970),doi:10.1093/biomet/57.1.97.
[48] Z.ZhuandH.G.Katzgraber,Dotensorrenormalizationgroupmethodsworkforfrustrated
spinsystems?,(arXivpreprint)doi:10.48550/arXiv.1903.07721.
[49] G. Bhanot, The Metropolis algorithm, Rep. Prog. Phys. 51, 429 (1988),
doi:10.1088/0034-4885/51/3/003.
[50] K.BinderandA.P.Young,Spinglasses: Experimentalfacts,theoreticalconcepts,andopen
questions,Rev.Mod.Phys.58,801(1986),doi:10.1103/RevModPhys.58.801.
[51] M. E. O’Neill, PCG: A family of simple fast space-efficient statistically good algorithms for
randomnumbergeneration(2014),https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf.
[52] B. A. Cipra, An introduction to the Ising model, Am. Math. Mon. 94, 10 (1987),
doi:10.2307/2322600.
[53] M.Friasetal.,Inpreparation.
[54] T.Horiguchi,FullyfrustratedIsingmodelonasquarelattice,Prog.Theor.Phys.Suppl.87,
33(1986),doi:10.1143/PTPS.87.33.
[55] H. G. Katzgraber, Introduction to Monte Carlo methods, (arXiv preprint)
doi:10.48550/arXiv.0905.1629.
[56] Z. Zhu, A. J. Ochoa and H. G. Katzgraber, Efficient cluster algorithm for
spin glasses in any space dimension, Phys. Rev. Lett. 115, 077201 (2015),
doi:10.1103/PhysRevLett.115.077201.
[57] H.G.Katzgraber,M.PalassiniandA.P.Young,MonteCarlosimulationsofspinglassesat
lowtemperatures,Phys.Rev.B63,184422(2001),doi:10.1103/PhysRevB.63.184422.
[58] D. H. Bailey, Algorithm 719: Multiprecision translation and execution of FORTRAN pro-
grams,ACMTrans.Math.Softw.19,288(1993),doi:10.1145/155743.155767.
[59] X.-Z. Wang and J. S, Kim, The critical line of an Ising antiferromagnet on square and
honeycomblattices,Phys.Rev.Lett.78,413(1997),doi:10.1103/physrevlett.78.413.
37

SciPostPhys. 14,123(2023)
[60] H. W. J. Blöte and M. P. M. den Nijs, Corrections to scaling at two-dimensional Ising tran-
sitions,Phys.Rev.B37,1766(1988),doi:10.1103/PhysRevB.37.1766.
[61] A. M. Ferrenberg, J. Xu and D. P. Landau, Pushing the limits of Monte Carlo sim-
ulations for the three-dimensional Ising model, Phys. Rev. E 97, 043301 (2018),
doi:10.1103/PhysRevE.97.043301.
[62] K. Binder, Statistical mechanics of finite three-dimensional Ising models, Physica 62, 508
(1972),doi:10.1016/0031-8914(72)90237-6.
[63] C. Domb, M. S. Green, Phase transitions and critical phenomena: Series expansion for
latticemodels,AcademicPress,Cambridge,USA,ISBN9780122203039(1974).
[64] J. V. José, L. P. Kadanoff, S. Kirkpatrick and D. R. Nelson, Renormalization, vortices, and
symmetry-breaking perturbations in the two-dimensional planar model, Phys. Rev. B 16,
1217(1977),doi:10.1103/PhysRevB.16.1217.
[65] J.F.Yu,Z.Y.Xie,Y.Meurice,Y.Liu,A.Denbleyker,H.Zou,M.P.Qin,J.ChenandT.Xiang,
Tensorrenormalizationgroupstudyofclassical XY modelonthesquarelattice,Phys.Rev.
E89,013308(2014),doi:10.1103/PhysRevE.89.013308.
[66] L.Vanderstraeten,B.Vanhecke,A.M.LäuchliandF.Verstraete,ApproachingtheKosterlitz-
Thouless transition for the classical XY model with tensor networks, Phys. Rev. E 100,
062136(2019),doi:10.1103/PhysRevE.100.062136.
[67] R.G.Jha,Criticalanalysisoftwo-dimensionalclassical XY model,J.Stat.Mech.: Theory
Exp.083203(2020),doi:10.1088/1742-5468/aba686.
[68] Y. Meurice, R. Sakai and J. Unmuth-Yockey, Tensor lattice field theory for
renormalization and quantum computing, Rev. Mod. Phys. 94, 025005 (2022),
doi:10.1103/RevModPhys.94.025005.
[69] B. Vanhecke, J. Haegeman, K. Van Acoleyen, L. Vanderstraeten, F. Verstraete, Scal-
ing hypothesis for matrix product states, Phys. Rev. Lett. 123, 250604 (2019),
doi:10.1103/physrevlett.123.250604.
[70] J.B.Kogut,Anintroductiontolatticegaugetheoryandspinsystems,Rev.Mod.Phys.51,
659(1979),doi:10.1103/RevModPhys.51.659.
[71] U.Frisch,B.HasslacherandY.Pomeau,Lattice-gasautomatafortheNavier-Stokesequa-
tion,Phys.Rev.Lett.56,1505(1986),doi:10.1103/PhysRevLett.56.1505.
[72] K.HukushimaandK.Nemoto,ExchangeMonteCarlomethodandapplicationtospinglass
simulations,J.Phys.Soc.Jpn.65,1604(1996),doi:10.1143/JPSJ.65.1604.
[73] A.N.ChowdhuryandR.D.Somma,QuantumalgorithmsforGibbssamplingandhitting-
timeestimation,QuantumInf.Comput.17,0041(2017),doi:10.26421/QIC17.1-2-3.
[74] M.Motta,C.Sun,A.T.K.Tan,M.J.O’Rourke,E.Ye,A.J.Minnich,F.G.S.L.Brandãoand
G. K.-L. Chan, Determining eigenstates and thermal states on a quantum computer using
quantumimaginarytimeevolution,Nat.Phys.16,205(2019),doi:10.1038/s41567-019-
0704-4.
[75] E. Giné, G. R. Grimmett and L. Saloff-Coste, Lectures on probability theory and
statistics, Springer, Berlin, Heidelberg, Germany, ISBN 9783540631903 (1997),
doi:10.1007/BFb0092617.
38

SciPostPhys. 14,123(2023)
[76] M. Lubasch, J. I. Cirac and M.-C. Bañuls, Algorithms for finite projected entangled pair
states,Phys.Rev.B90,064425(2014),doi:10.1103/PhysRevB.90.064425.
39
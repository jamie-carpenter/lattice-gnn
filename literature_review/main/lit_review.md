# Literature review: ML surrogates for the effective stiffness of lattice unit cells (GNNs, datasets, homogenization, defects)

## 1. Overview

The closest prior work to this project is a small cluster of papers from the Cambridge group (Deshpande, Csanyi) built on the same crystal-network catalog. Indurkar 2022 applied a message-passing GNN to classify stretching vs bending. Grega 2024 (ICLR) built an SE(3)-equivariant MACE model with a positive semi-definite output layer for the full stiffness tensor. A JMPS companion paper exists (abstract only here). The wider field (Bastek 2022, Zheng 2023, Meyer 2022) uses its own datasets, mostly at other relative densities and with other beam models, so reported accuracies from those papers are **not** baselines for our dataset.

Reading limits you should know about:
- The PNAS PDF of the dataset paper could not be loaded by read_pdf (several URLs tried). I read its Methods through the Europe PMC full-text XML (first ~100k of 122k characters). That text carries no PDF page numbers, so those citations use section names. The Supplementary Information was **not** read.
- Bastek 2022 and Zheng 2023 were also read through Europe PMC text (first ~100k characters). No page numbers.
- Full PDF pages were read for Grega 2024 (pp. 1-13 of 25), Indurkar 2022 (pp. 1-4 of 7), Jekel 2022 (pp. 1-4 of 17) and Choukir 2026 (pp. 1-2 of 50). Everything else is abstract-only.
- I did not find any paper that reports an error on our dataset in log10 modulus units. The reported baselines are in other metrics (Section 5).

## 2. Our dataset

### 2.1 Source: confirmed as Lumpe & Stankovic (PNAS 2021)
- The catalog is decoded from the RCSR and EPINET crystal-net databases. The paper gives the count as "2,730 + 14,532 - 135 - 40 = 17,087" entries, excluding double entries and structures with very small members that cause numerical issues [Lumpe 2021, Methods (Europe PMC text; no PDF page)].
- Independent confirmation that the catalog with properties is the one cited by later GNN work: Indurkar 2022 uses "17,201 unique 3D lattice topologies adopted from an open-source catalogue" [Indurkar 2022, p. 1]. Grega 2024 says it uses the Lumpe & Stankovic data, with nodal positions, edge connectivity, crystal constants and some elastic properties [Grega 2024, p. 4].
- Data were deposited in the ETH Research Collection (doi 10.3929/ethz-b-000457598 and -000457595) [Lumpe 2021, Data Availability (Europe PMC text)].

### 2.2 Count discrepancies (unresolved)
Four different counts are in play: 17,087 (paper Methods), 17,201 (Indurkar), 17,221 with properties (our solve of the catalog), and 17,262 total (ours, with 41 "very small members"). I could not reconcile them from the text I read. The paper does not say how "135" and "40" map onto double entries vs very small members. Possible reasons (unverified): different catalog versions on ETH, or the supplementary catalog containing entries the main text excludes. **Action: check the two ETH deposits and the SI.**

### 2.3 How the 41 "very small members" were handled
The Methods say structures with very small members "cause numerical issues" and are excluded from the entries analyzed [Lumpe 2021, Methods (Europe PMC text)]. That matches our 41 records having no properties. I found nothing on whether they were retried with other parameters. Treat them as unlabeled; keep them out of training and use them, at most, for a robustness test of the featurization.

### 2.4 How labels were computed
- Method: numerical homogenization with an Euler-Bernoulli FE beam model and periodic boundary conditions. Six load cases give the 3D effective stiffness C^H, and compliance S^H = (C^H)^-1 [Lumpe 2021, Methods and Section 2 (Europe PMC text)].
- Struts: circular cross-section, constant bar radius [Lumpe 2021, Methods]. Whether node volumes or overlaps are corrected for relative density: **not stated in the text I read**.
- Base material: Es = 1 MPa, nu_s = 0.3; all structures scaled to relative density 0.01 [Lumpe 2021, Section 2 and Methods; Indurkar 2022, p. 1].
- Scaling law: E_i = Es * C_i * rho^n_i, with n_i ~ 1 for stretching-dominated and ~ 2 for bending-dominated behaviour, and 1 < n_i < 2 for mixed behaviour [Indurkar 2022, p. 2, Eq. 1; extracted formula was garbled, paraphrased].
- **Your 1% relative-density inference is confirmed**: all catalog structures are at rho = 0.01 by construction. The scatter you see when inverting C and n is the rounding of C and n to 2 decimals.
- How C and n were fitted: the main text refers to the SI [Lumpe 2021, Section 2 (Europe PMC text)]. **Not verified.** Since the catalog is evaluated at one density, n and C presumably come from a second density or from repeated runs. Please read the SI before using n or C as targets or features.
- Moduli span a wide range: Indurkar reports E_i between 1e-8 and 1e-3 (normalized) [Indurkar 2022, p. 2]. This supports predicting log10 moduli.

### 2.5 Known limitations
- Structures are crystal nets used as-is, with no optimization [Lumpe 2021, Section 4 (Europe PMC text)].
- The paper validates against analytical literature values and selected compression tests (SI Fig. S1) [Lumpe 2021, Section 2]. Validation covers selected structures only, not the catalog.
- Euler-Bernoulli beams at rho = 0.01 ignore shear deformation and node effects, which matters little for slender struts. If a defect study raises the radius, the model will drift. Zheng 2023 uses Timoshenko beams [Zheng 2023, Creating the design space].
- Data come from a single density, so "log10(total strut length / volume)" is a proxy for the one fixed radius: at constant rho, strut length determines radius. Check that it is not redundant with Z_avg for your split.
- Mechanisms and near-rigid structures: Grega 2024 notes that in their processed data all eigenvalues of C are positive [Grega 2024, p. 8, footnote 2]. They also fixed or dropped problematic lattices, going from the catalog to 8954 base lattices [Grega 2024, p. 5]. Expect a tail of near-singular lattices in ours.

## 3. Other datasets

| Dataset | Size | Labels | Notes |
|---|---|---|---|
| Grega 2024 (derived from Lumpe) | 8,954 base lattices; 635,454 geometries after nodal perturbation; FE at 3 relative densities | full stiffness tensor | Perturbations 0.01-0.1; public DOI 10.17863/CAM.106854 [Grega 2024, pp. 5, 10] |
| Zheng 2023 | 965,736 cubic-symmetry trusses (27 candidate nodes) | 9 orthotropic components /Es at rho = 0.15, Timoshenko beams | own generation [Zheng 2023, Creating the design space (Europe PMC text)] |
| Bastek 2022 | 3,000,000 training samples, 30,000 test; 262 topologies | 9 orthotropic + 21 sheared constants /E_b, rho in [0.002, 0.1] | [Bastek 2022, Section 2 (Europe PMC text)] |
| Choukir 2026 | ~1.95 million voxel unit cells, 36 cubic space groups | strain energy density | voxel CNN, not struts [Choukir 2026, p. 1] |
| Meyer 2022 | shell-lattices (size not read) | stiffness and conductivity tensors | [Meyer 2022, abstract] |


## 4. Model architectures

| Paper | Representation | Model | Targets (units / normalization) | Data size | Reported accuracy | Page |
|---|---|---|---|---|---|---|
| Indurkar 2022 | periodic lattice graph; node: coords, connectivity, angle stats; edge: length, radius, direction cosines, edge coordination; periodic edges to neighbouring cells | conv message passing + GRU update, global pooling, MLP + softmax | 3-class stretching / bending / mixed from n_x (n_x < 1.05, n_x > 1.95) | 17,201 lattices; random 90:10 | val accuracy 81.8%, train 95.2% (classification, not stiffness) | pp. 3-4 |
| Grega 2024 (ICLR) | strut graph, spherical-harmonic edge embedding | MACE equivariant MPNN + PSD stack on 6x6 Mandel matrix | full 4th-order C (21 comps) in their own units; L_comp metric defined in appendix, not read | 7,000 train / 1,296 val-test base lattices; test = 0.1 nodal perturbation (OOD) | L_comp: MACE 3.88, MACE+ve 3.61, CGC 8.37, CGC+tr 4.47 | pp. 5-7, Table 1 |
| Bastek 2022 | one-hot topology + relative density + stretch eigenvalues | two MLPs with differentiable Voigt rotations | 9 orthotropic constants, then 21; /E_b | 3,000,000 | R2 >= 0.98 on 30,000 test lattices | Europe PMC text (no page) |
| Zheng 2023 | 27-node adjacency + node offsets | VAE + MLP property predictor | 9 orthotropic C components /Es at rho = 0.15 | 965,736 | R2 >= 98.2% | Europe PMC text (no page) |
| Meyer 2022 | graph of shell-lattice (+ dual graph in mCGC) | crystal graph convolution | 21 stiffness components, smooth-L1 | not read | not read | Grega 2024 p. 4 |
| Choukir 2026 | voxels | 3D CNN | strain-energy density | ~1.95M | not read | p. 1 |

Do not compare R2 values (Bastek, Zheng) with L_comp (Grega) or classification accuracy (Indurkar): they use different data, density, units and metrics.

Takeaways for our model:
- Periodic edges: Indurkar 2022 connects boundary nodes to nodes in neighbouring cells [Indurkar 2022, p. 3]. This matches our design.
- Dual-graph message passing (Meyer 2022) gave Grega 2024 no gain on their data [Grega 2024, p. 6, footnote 1]. Skip it initially.
- Higher-order messages and Lmax = 4 helped Grega 2024 for the 4th-order tensor, and models truncated to Lmax = 2 or 3 could not reproduce anisotropy of a simple cubic lattice [Grega 2024, p. 8]. This concerns predicting rotating tensors. Our targets are orthotropic constants in the lattice frame, so equivariance is not required, but see Section 7.
- Mean + max pooling and Z_avg as a graph input: no paper I read tests these. Indurkar identifies average connectivity as the main driver of stretching vs bending (citing literature) [Indurkar 2022, p. 2]. Z_avg as a graph-level input is therefore well motivated.

## 5. Baselines on our dataset

No paper I found reports log10 modulus errors on this catalog.
- **Indurkar 2022**: classification only. 81.8% validation accuracy for 3 classes of n_x on a random split [Indurkar 2022, p. 4]. A random split puts near-identical members of one crystal family in both train and validation.
- **Grega 2024**: derived dataset (8,954 base lattices after cleaning, augmented, three densities), metric L_comp, tensor output. Not comparable to ours in units or in density. Their split keeps perturbations of one lattice together in train or test [Grega 2024, p. 5], which is a good practice to copy.
- Papers that cite the catalog but I did not find evidence of trained stiffness models: Meyer 2024 (plate-lattices), Frey 2025 and Zheng 2023 appeared in the citation list but I could not confirm they train on this catalog. Zheng 2023 states its own data generation procedure.

**Recommendation**: report your own baselines (MLP on Z_avg and log10 length density; gradient-boosted trees on hand-made features) in log10 units with RMSE and R2 per target, using a split by crystal system or by RCSR net family as well as a random split. Say explicitly that it is not comparable to published numbers.

## 6. Enforcing tensor symmetry and physical constraints

- Symmetry of the stiffness tensor: the fourth-order tensor has major and minor symmetries, leaving 21 independent components [Grega 2024, p. 4 and Appendix A.1, p. 13]. An orthotropic material has 9 independent components in Voigt notation [Jekel 2022, p. 4, Eq. 2]. Predicting 9 engineering constants is a parametrization of that orthotropic form.
- Positive definiteness: stored energy must be non-negative, so all eigenvalues of C must be non-negative [Grega 2024, p. 4]. A fitted model does not guarantee this: unconstrained MACE predicted negative eigenvalues for 30% of lattices [Grega 2024, p. 7, Table 2], and Jekel 2022 says plain fits to SPD data do not stay SPD [Jekel 2022, p. 1].
- Remedies: Cholesky or eigen-decomposition layers [Jekel 2022, p. 1; Grega 2024, pp. 5, 13]. Grega found the matrix-square map A^2 best, at slightly higher error than unconstrained (L_comp 3.61 vs 3.88) [Grega 2024, pp. 7-8, Table 3]. A loss penalty on directional stiffness was not effective [Grega 2024, p. 8].
- For our parametrization (general textbook facts, not cited to a paper I read): predicting log10 of E and G guarantees positive moduli. The orthotropic compliance matrix is positive definite iff E, G > 0 and the Poisson terms satisfy the standard stability inequalities (e.g. 1 - nu_xy*nu_yx > 0 and the combined determinant condition). Poisson coupling terms predicted independently can break this. **Recommendation**: after predicting, assemble S (compliance) from E, G, nu, invert to C, and check eigenvalues. Report the fraction of violations. Consider predicting a bounded transform of nu (e.g. via Cholesky of S) if violations exceed ~1%.
- Cubic and other crystal symmetries: Grega and Bastek exploit orthotropy or equivariance, but none of the papers I read exploits the crystal system as a constraint (e.g. forcing Ex = Ey = Ez for cubic). With the crystal-system code in the name, tie-breaking or averaging predictions across symmetric axes is a cheap test.
- Axis ordering: if x, y, z carry no special meaning in the catalog (cell angles and lengths are normalized per cell), augment by permuting axes and permute targets accordingly, as Meyer 2022 and Grega 2024 do for rotations [Grega 2024, pp. 4, 6]. Check first how the catalog fixes the axis of non-cubic cells.


## 7. Label generation / homogenization

- Our labels: periodic Euler-Bernoulli beam FE, six load cases, circular struts, Es = 1 MPa, nu_s = 0.3, rho = 0.01 [Lumpe 2021, Methods and Section 2 (Europe PMC text)].
- Zheng 2023 uses Timoshenko beams with periodic FE at rho = 0.15 [Zheng 2023, Creating the design space]. Bastek 2022 normalizes by the base modulus with nu = 0.3 and covers rho in [0.002, 0.1] [Bastek 2022, Section 2]. A model trained on our rho = 0.01 data should not be applied to other densities without retraining.
- Grega 2024 reruns FE after perturbing nodes and for three relative densities, so strut radius is a variable in their data [Grega 2024, p. 5]. This is the template if you later need FE labels for defected lattices. Their code and dataset are public [Grega 2024, p. 10].
- Homogenization reviews exist (Somnic 2022, abstract only). Higher-order continuum extensions are beyond the scope of a unit-cell effective stiffness.
- Label check you can do cheaply: recompute Ex for a few well-known nets (pcu, bcu, fcu type) with your own beam solver and compare with the catalog. The original paper validates only against analytical literature values and selected compression tests [Lumpe 2021, Section 2].

## 8. Manufacturing defects

What I found is thin and mostly experiment-oriented. None of it is an ML defect-stiffness model, and I found no GNN trained on defective strut lattices other than the node-perturbation dataset of Grega 2024.

- **Strut waviness, thickness variation, oversizing**: Liu 2017 extracted distributions from X-ray CT of SLM lattices (octet and rhombicuboctahedron units) and used them to build FE models of statistically representative imperfect lattices [Liu 2017, abstract (secondary summary)]. Raghavendra 2019 introduces oversizing, varying thickness and waviness in FE by changing cross-sections along a strut [Raghavendra 2019, abstract]. I did not read the knockdown values, so no numbers from these papers should be quoted until a later agent reads them.
- **Node misplacement**: Grega 2024 perturbs every node by a fixed distance in a random direction at levels 0.01-0.1 and recomputes FE [Grega 2024, p. 5]. The JMPS companion reports four classes of mechanical behaviour under perturbation [Grega JMPS 2024, abstract]. This is the closest published setup to our node-misplacement defect. Caveat: perturbations apply only to lattices with at least 2 fundamental nodes [Grega 2024, p. 5].
- **Disorder in general**: Karapiperis 2023 uses GNNs on irregular beam lattices, but for fracture paths [Karapiperis 2023, abstract]. Luan 2023 links strut orientation to stiffness in ordered and disordered metamaterials [Luan 2023, abstract].
- **Missing struts and porosity**: no paper I found treats these in a way I can cite. Removing struts lowers connectivity and can turn a stretching-dominated lattice into a bending-dominated one or a mechanism. This follows from the scaling-law interpretation in Indurkar 2022, p. 2, not from a defect study.
- **Taxonomy**: Zok 2019 frames lattice structure as topology, morphology and defects [Zok 2019, abstract].

Implementation notes (my reasoning, not literature claims): keep the same Euler-Bernoulli FE solver for consistency. Vary radius per strut for diameter variation, add midspan offsets for waviness, delete struts at random while checking rigidity, and perturb nodes as in Grega 2024. Because the catalog fixes rho = 0.01, state whether defected lattices are compared at equal mass (rho held) or equal design (radius held). The two give different knockdowns.


## 9. Gaps and recommendations for this project

1. **Read the SI of Lumpe and Stankovic** to confirm (a) how C and n were fitted, (b) what the 135 and 40 exclusions are, (c) whether node volumes are corrected in rho, (d) axis conventions for non-cubic cells. These are the items I could not verify.
2. **Reconcile counts** 17,087 / 17,201 / 17,221 / 17,262 (Section 2.2).
3. **Splits**: random splits leak crystal families (Indurkar used a random 90:10 split). Grega keeps perturbations of one base lattice together [Grega 2024, p. 5]. Use grouped splits by net or crystal system and report both.
4. **Metric**: report RMSE of log10 for E and G and of the Poisson terms, plus the share of lattices whose assembled C is not positive definite. Standardized targets hide absolute error, so also report unstandardized values.
5. **Poisson terms**: define the three coupling terms precisely (which ratio, which direction) and check bounds. None of the papers I read predicts them directly.
6. **Equivariance**: not required for in-frame orthotropic constants, but it is the main architectural idea of Grega 2024. Test cheap augmentation (axis permutation with target permutation) before adopting MACE-type models.
7. **Defects**: read Liu 2017, Raghavendra 2019 and Dallago 2019 in full to get knockdown ranges for validating the defect model. Dallago 2019 was found in search but not recorded or read.
8. **Normalization of defected data**: decide whether E/Es uses the ideal or the actual relative density.

### Correction to Section 5
Section 5 says Meyer 2024, Frey 2025 and Zheng 2023 appeared in a citation list for the catalog. That is not verified: they surfaced in keyword searches, and the full citation list of the dataset paper was not reviewed. The only confirmed users of the catalog are Indurkar 2022 and Grega 2024. Zheng 2023 cites it but builds its own data.

## 10. References (PDF version read)

- Lumpe T.S., Stankovic T. (2021) Exploring the property space of periodic cellular structures based on crystal networks. PNAS 118(7), doi:10.1073/pnas.2003504118. Read via Europe PMC XML (PMC7896306), no PDF pages; SI not read.
- Indurkar P.P., Karlapati S., Shaikeea A.J.D., Deshpande V.S. (2022) Predicting deformation mechanisms in architected metamaterials using GNN. arXiv:2202.09427v2, pp. 1-4 of 7.
- Grega I., Batatia I., Csanyi G., Karlapati S., Deshpande V.S. (2024) Energy-conserving equivariant GNN for elasticity of lattice architected metamaterials. ICLR 2024, arXiv:2401.16914v2, pp. 1-13 of 25.
- Grega I., Batatia I., Indurkar P.P., Csanyi G. et al. (2024) Graph neural networks for strut-based architected solids. JMPS, doi:10.1016/j.jmps.2024.105966. Abstract only.
- Jekel C.F. et al. (2022) Neural network layers for prediction of positive definite elastic stiffness tensors. arXiv:2203.13938v1, pp. 1-4 of 17.
- Bastek J.-H. et al. (2022) Inverting the structure-property map of truss metamaterials by deep learning. PNAS 119(1), doi:10.1073/pnas.2111505119. Europe PMC XML (PMC8740766), no pages.
- Zheng L. et al. (2023) Unifying the design space and optimizing linear and nonlinear truss metamaterials by generative modeling. Nat. Commun., doi:10.1038/s41467-023-42068-x. Europe PMC XML (PMC10663604), no pages.
- Meyer P.P. et al. (2022) Graph-based metamaterials: Deep learning of structure-property relations. Mater. Des., doi:10.1016/j.matdes.2022.111175. Abstract only.
- Frey R. et al. (2025) Multi-objective design of multi-material truss lattices utilizing graph neural networks. Sci. Rep., doi:10.1038/s41598-025-86812-3. Abstract only.
- Choukir S. et al. (2026) An Atlas of Extreme Properties in Cubic Symmetric Metamaterials. arXiv:2603.10934v1, pp. 1-2 of 50.
- Xie T., Grossman J.C. (2017) Crystal graph convolutional neural networks. arXiv:1710.10324. Abstract only.
- Batatia I. et al. (2022) MACE. arXiv:2206.07697. Abstract only.
- Karapiperis K., Kochmann D.M. (2023) Prediction and control of fracture paths in disordered architected materials using graph neural networks. doi:10.1038/s44172-023-00085-0. Abstract only.
- Hendriks F. et al. (2025) Similarity equivariant graph neural networks for homogenization of metamaterials. doi:10.1016/j.cma.2025.117867. Abstract only.
- Luan S. et al. (2023) A data-driven framework for structure-property correlation in ordered and disordered cellular metamaterials. doi:10.1126/sciadv.adi1453. Abstract only.
- Maurizi M., Gao C., Berto F. (2022) Inverse design of truss lattice materials with superior buckling resistance. doi:10.1038/s41524-022-00938-w. Abstract only.
- Kumar S. et al. (2020) Inverse-designed spinodoid metamaterials. doi:10.1038/s41524-020-0341-6. Abstract only.
- Somnic J., Jo B.W. (2022) Status and challenges in homogenization methods for lattice materials. doi:10.3390/ma15020605. Abstract only.
- Liu L. et al. (2017) Elastic and failure response of imperfect three-dimensional metallic lattices. JMPS, doi:10.1016/j.jmps.2017.07.003. Abstract only via a secondary summary; the PDF failed to load.
- Raghavendra S. et al. (2019) Effect of strut cross section and strut defect on tensile properties of cubic cellular structure. doi:10.1002/mdp2.118. Abstract only.
- Zok F.W. (2019) Integrating lattice materials science into the traditional processing-structure-properties paradigm. doi:10.1557/mrc.2019.152. Abstract only.
- Utzeri M. et al. (2024) Multiscale experiments and predictive modelling for inverse design and failure mitigation in additively manufactured lattices. arXiv:2402.16452. Abstract only.

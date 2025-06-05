# Conventions

## Coordinate frames

### The diffractometer equation

We use the vector $\vec{h}$ to describe a position in *fractional reciprocal space* in terms of the reciprocal lattice basis vectors $\vec{a^*}$, $\vec{b^*}$ and $\vec{c^*}$.

$$\vec{h} = \begin{pmatrix}
h \\
k \\
l \\
\end{pmatrix} = h \vec{a^*} + k \vec{b^*} + l \vec{c^*}$$

The special positions at which h, k and l are integer define the *reciprocal lattice points* for which (hkl) are the *Miller indices*.

The basic diffractometer equation relates a position $\vec{h}$ to a position $\vec{r_\phi}$ in *Cartesian reciprocal space*. This space is defined so that its axes coincide with the axes of the *laboratory frame*. The distinction is necessary because distances in reciprocal space are measured in units of $\text{\AA}^{-1}$. However, for convenience it is often acceptable to refer to either Cartesian reciprocal space or the real space laboratory frame as the "lab frame", when the correct choice is clear by context. The diffractometer equation is:

$$\vec{r_\phi} = \mathbf{R} \mathbf{A} \vec{h}$$

where $\mathbf{R}$ is the *goniostat rotation matrix* and $\mathbf{A}$ is the *crystal setting matrix*, while its inverse $\mathbf{A}^{-1}$ is referred to as the *indexing matrix*. The product $\mathbf{A} \vec{h}$ may be written as $\vec{r_0}$, which is a position in the $\phi$-axis frame, a Cartesian frame that coincides with the laboratory frame at a rotation angle of $\phi=0$. This makes clear that the setting matrix does not change during the course of a rotation experiment (notwithstanding small "misset" rotations — see *Orientation matrix*).

For an experiment performed using the rotation method we use here $\phi$ to refer to the angle about the actual axis of rotation, even when this is effected by a differently labelled axis on the sample positioning equipment (such as an $\omega$ axis of a multi-axis goniometer). Only in code specifically dealing with sample positioning equipment might we need to redefine the labels of axes. Outside of such modules, the rotation angle is $\phi$ and the axis of rotation is $\vec{e}$, which together with the definition of the laboratory frame determine the rotation matrix $\mathbf{R}$.

### Orthogonalisation convention

Following [2] we may decompose the setting matrix $\mathbf{A}$ into the product of two matrices, conventionally labelled $\mathbf{U}$ and $\mathbf{B}$. We name $\mathbf{U}$ the *orientation matrix* and $\mathbf{B}$ the *reciprocal space orthogonalisation matrix*. These names are in common, but not universal use. In particular, some texts (for example [6]) refer to the product (i.e. our setting matrix) as the "orientation matrix".

Of these two matrices, $\mathbf{U}$ is a pure rotation matrix and is dependent on the definition of the lab frame, whilst $\mathbf{B}$ is not dependent on this definition. $\mathbf{B}$ does depend however on a choice of orthogonalisation convention, which relates $\vec{h}$ to a position in the *crystal-fixed Cartesian system*. The basis vectors of this orthogonal Cartesian frame are fixed to the reciprocal lattice *via* this convention.

There are infinitely many ways that $\mathbf{A}$ may be decomposed into a pair $\mathbf{U} \mathbf{B}$. The symbolic expression of $\mathbf{B}$ is simplified when the crystal-fixed Cartesian system is chosen to be aligned with crystal real or reciprocal space axes. For example, [2] use a frame in which the basis vector $\vec{i}$ is parallel to reciprocal lattice vector $\vec{a^*}$, while $\vec{j}$ is chosen to lie in the plane of $\vec{a^*}$ and $\vec{b^*}$. Unfortunately, this convention is then disconnected from the standard *real space* orthogonalisation convention, usually called the *PDB convention* [7]. This standard is essentially universal in crystallographic software for the transformation of fractional crystallographic coordinates to positions in orthogonal space, with units of $\text{\AA}$. In particular, it is the convention used in the cctbx [4]. The convention states that the orthogonal coordinate $x$ is determined from a fractional coordinate $u$ by:

$$\vec{x} = \mathbf{O} \vec{u}$$

where the matrix $\mathbf{O}$ is the *real space orthogonalisation matrix*. This matrix transforms to a crystal-fixed Cartesian frame that is defined such that its basis vector $\vec{i}$ is parallel to the real space lattice vector $\vec{a}$, while $\vec{j}$ lies in the $(\vec{a}, \vec{b})$ plane. The elements of this matrix made explicit in a compact form are:

$$\mathbf{O} =
\begin{pmatrix}
a & b\cos{\gamma} &  c\cos{\beta} \\
0 & b\sin{\gamma} & -c\sin{\beta}\cos{\alpha^*} \\
0 & 0             &  c\sin{\beta}\sin{\alpha^*} \\
\end{pmatrix}$$

It is desirable to specify our *reciprocal space* orthogonalisation convention in terms of this real space orthogonalisation convention. [3] derives relationships between real and reciprocal space. Of particular interest from that text we have:

$$
\begin{align}
\vec{x} &= \mathbf{M}^\mathsf{T} \vec{x}^\prime \\
\vec{x^*} &= \mathbf{M}^{-1} \vec{x^*}^\prime
\end{align}
$$

By analogy, equate $\vec{x^*}^\prime$ with $\vec{h}$ and $\mathbf{B}$ with $\mathbf{M}^{-1}$. Also equate $\mathbf{M}^\mathsf{T}$ with $\mathbf{O}$ and $\vec{x}^\prime$ with $\vec{u}$. We then see that:

$$\mathbf{B} = \left( \mathbf{O}^{-1} \right)^\mathsf{T} = \mathbf{F}^\mathsf{T}$$

where $\mathbf{F}$ is designated the *real space fractionalisation matrix*. This is easily obtained in cctbx by a method of a `cctbx.uctbx.unit_cell` object.

A symbolic expression for $\mathbf{F}$ in terms of the real space unit cell parameters is given by [8] from which we derive $\mathbf{B}$ simply:

$$\mathbf{B} =
\begin{pmatrix}
\frac{1}{a} &
0 &
0 \\
-\frac{\cos{\gamma}}{a\sin{\gamma}} &
\frac{1}{b\sin{\gamma}} &
0 \\
\frac{bc}{V}\left( \frac{\cos{\gamma} \left( \cos{\alpha} - \cos{\beta}\cos{\gamma} \right)}{\sin{\gamma}} - \cos{\beta}\sin{\gamma} \right) &
-\frac{ac \left( \cos{\alpha} - \cos{\beta}\cos{\gamma} \right)}{V\sin{\gamma}} &
\frac{ab\sin{\gamma}}{V} \\
\end{pmatrix}$$

with $V = abc \sqrt{ 1 - \cos^2{\alpha} - \cos^2{\beta} - \cos^2{\gamma} + 2 \cos{\alpha}\cos{\beta}\cos{\gamma}}$

## Orientation matrix

The matrix $\mathbf{U}$ "corrects" for the orthogonalisation convention implicit in the choice of $\mathbf{B}$. As the crystal-fixed Cartesian system and the $\phi$-axis frame are both orthonormal, Cartesian frames with the same scale, it is clear that $\mathbf{U}$ must be a pure rotation matrix. Its elements are clearly dependent on the mutual orientation of these frames.

It is usual to think of the orientation as a fixed property of the "sequence". In practice the orientation is parameterised such that it becomes a function of time, to account for crystal slippage (the true degree of this is unknown but expected to be small; Mosflm uses crystal orientation parameters to account for inadequacies in other aspects of the experimental description). To reconcile these points, the current orientation may be expanded into a fixed, datum part and a variable time-dependent part that is parameterised. That gives:

$$\vec{r_\phi} = \mathbf{\Psi}\mathbf{R}\mathbf{U_0}\mathbf{B}\vec{h}$$

where $\mathbf{\Psi}$ is the combined rotation matrix for the misset expressed as three angles, $\psi_x, \psi_y$ and $\psi_z$ in the laboratory frame.

In Mosflm these angles are converted to their equivalents in the $\phi-$ axis frame, where:

$$\vec{r_\phi} = \mathbf{R}\mathbf{\Phi}\mathbf{U_0}\mathbf{B}\vec{h}$$

At this stage it is unclear which set of angles are the best choice for parameterisation of the crystal orientation.

### The laboratory frame

An important design goal of the DIALS project is that all algorithms should be fully vectorial. By this we mean that it should be possible to change the reference frame arbitrarily and all calculations should work appropriately in the new frame.

Nevertheless, it is useful to adopt a particular standard frame of reference for meaningful comparison of results, communication between components of the software and for an agreed definition of what the laboratory consists of (incompatible definitions can be reasonably argued for, such as that it should be either fixed to the detector, or to the rotation axis and beam).

In the interests of both standardisation and practicality, we choose to adopt the Image CIF (imgCIF) reference frame [1], [5], for cases with a single axis horizontal goniometer. For beamlines with a vertical goniometer, we align the rotation axis with the $Y$ rather than the $X$ axis. This ensures the axis appears vertical in the image viewer, reducing confusion for the users. Such decisions can be made on a case-by-case basis within the specific format class used to read the images, giving the freedom to choose the most convenient coordinate system for the geometry of each experiment.

### Summary of coordinate frames

* $\vec{h}$ gives a position in *fractional reciprocal space*, fixed to the crystal.
* $\mathbf{B}\vec{h}$ gives that position in the *crystal-fixed Cartesian system* (basis aligned to crystal axes by the orthogonalization convention)
* $\mathbf{UB}\vec{h}$ gives the $\phi$-axis frame (rotates with the crystal, axes aligned to lab frame at $\phi=0$)
* $\mathbf{RUB}\vec{h}$ gives *Cartesian reciprocal space* (fixed wrt the laboratory)
* The diffraction geometry relates this to the direction of the scattering vector $\vec{s}$ in the *laboratory frame*
* Projection along $\vec{s}$ impacts an *abstract sensor frame* giving a 2D position of the reflection position on a sensor.
* This position is converted to the *pixel position* for the 2D position on an image in number of pixels (starting 0,0 at the origin).

### The DXTBX goniometer model

The following information is likely only to be of interest to developers of DIALS, since it is concerned with internal conventions for the representation of the goniostat rotation operator.

When one performs a rotation diffraction experiment, the goniostat rotation operator $\mathbf{R}$, which was introduced in the diffractometer equation, represents the rotation of the sample in the real-space laboratory frame. Equivalently, it relates the laboratory frame to the rotated real-space coordinate system of the sample. By numbering the physical motors of a goniometer such that motor 1 is mounted to the laboratory floor, motor 2 is mounted on motor 1, motor 3 is mounted on motor 2, etc., $\mathbf{R}$ can be expressed as a composition of rotation operators, one for each motor:

$$\mathbf{R} = \mathbf{R}_1 \circ \mathbf{R}_2 \circ \mathbf{R}_3 \circ \cdots$$

It is useful to represent $\mathbf{R}$ this way because, in practice, the position of a gonimometer is usually recorded as an angular displacement for each motor $i$, which gives the magnitude of the rotation $\mathbf{R}_i$, combined with prior knowledge of the orientation of each motor's axis in the laboratory frame when the goniometer is at its zero datum, which gives the axis vector of $\mathbf{R}_i$.

DIALS uses the DXTBX package to handle the geometry of diffraction experiments. In DXTBX, it is assumed that only one goniometer motor will turn during a measurement scan. Numbering that motor $n$, the various operators $\mathbf{R}_i$ are grouped into three:

$$\mathbf{R} = \mathbf{S} \circ \mathbf{R}' \circ \mathbf{F}$$

where $\mathbf{R}' = \mathbf{R}_n$ is the scanning axis rotation and $\mathbf{S}$ and $\mathbf{F}$ are combined rotation operators:

$$
\begin{align}
   \mathbf{S} &= \cdots \circ \mathbf{R}_{n - 1} \\
   \mathbf{F} &= \mathbf{R}_{n + 1} \circ \cdots
\end{align}
$$

$\mathbf{S}$ is referred to as the 'setting rotation' (not to be confused with the crystal setting operator $\mathbf{A}$) and represents the combined rotation of all motors between the laboratory floor and the scanning motor $n$. The setting rotation is so-named because it sets the orientation of the axis of $n$. $\mathbf{F}$ is referred to as the 'fixed rotation' and represents the combined rotation of all axes between $n$ and the sample. Both $\mathbf{S}$ and $\mathbf{F}$ are constant throughout a scan, since they represent motors whose positions are set before starting the scan and remain fixed throughout the scan.

For example, a common diffractometer apparatus is the three-circle $\kappa$ geometry. In this arrangement, the $\omega$ motor is fixed to the laboratory floor, the $\kappa$ motor is mounted on $\omega$, and the $\phi$ motor is mounted on $\kappa$ and holds the sample mount. During a typical rotation scan, either $\phi$ or $\omega$ will rotate, while the other two axes are held in a fixed orientation, chosen so as to explore a particular region of reciprocal space. In the case of a $\phi$ scan, $\mathbf{S} = \mathbf{R}_\omega \circ \mathbf{R}_\kappa$, $\mathbf{R}' = \mathbf{R}_\phi$ and $\mathbf{F} = \mathbf{1}$. In the case of an $\omega$ scan, $\mathbf{S} = \mathbf{1}$, $\mathbf{R}' = \mathbf{R}_\omega$ and $\mathbf{F} = \mathbf{R}_\kappa \circ \mathbf{R}_\phi$.

## References

[1] [Bernstein, H. J. in Int. Tables Crystallogr. 199–205 (IUCr, 2006).](http://it.iucr.org/Ga/ch3o7v0001/)

[2] Busing, W. R. & Levy, H. A. Angle calculations for 3- and 4-circle X-ray and neutron diffractometers. Acta Crystallogr. 22, 457–464 (1967).

[3] Giacovazzo, C. Fundamentals of Crystallography. (Oxford University Press, USA, 2002).

[4] Grosse-Kunstleve, R. W., Sauter, N. K., Moriarty, N. W. & Adams, P. D. The Computational Crystallography Toolbox: crystallographic algorithms in a reusable software framework. J. Appl. Crystallogr. 35, 126–136 (2002).

[5] [Hammersley, A. P., Bernstein, H. J. & Westbrook, D. in Int. Tables Crystallogr. 444–458 (IUCr, 2006).](http://it.iucr.org/Ga/ch4o6v0001/)

[6] Paciorek, W. A., Meyer, M. & Chapuis, G. On the geometry of a modern imaging diffractometer. Acta Crystallogr. Sect. A Found. Crystallogr. 55, 543–557 (1999).

[7] [PDB. Atomic Coordinate and Bibliographic Entry Format Description. Brookhaven Natl. Lab. (1992).](http://www.wwpdb.org/docs/documentation/file-format/PDB_format_1992.pdf)

[8] [Rupp, B. Coordinate system transformation.](http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm)

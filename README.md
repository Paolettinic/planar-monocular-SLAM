# Planar Monocular Slam
Probabilistic Robotics Project:
This repository implements a bundle adjustment system for optimizing robot
poses and landmark positions. The optimization is performed using total
least squares, with odometry and landmarks projection constraints.
The resulting graphs svg and an html page containing a representation of the
landmarks (see Results) can be found in the output directory,

## Project description

### Triangulation
The initial estimate of the landmark position is obtained via triangulation.
Both the methods described in the request were implemented, to check which gave
better results:
- The first way consists in triangulating the point using pairs of poses, in
    which the same point is visible, for all points. The resulting 3d
    coordinates for the same point are merged by averaging them.
- The second takes into account all the poses in which the same point is
    visible, and finds the solution that minimizes the reprojection error.

Both methods use SVD decomposition to find the Null space of the system
explained in: "Computer Vision: Algorithms and Applications, Szeliski"
Chap. 7.1 eq. 7.5, 7.6.
$$
\begin{bmatrix}u\\v\end{bmatrix} = \pi(PX_l)
$$
$$
\begin{bmatrix}u\\v\end{bmatrix} =
\begin{bmatrix}\frac{P_0X_l}{P_2X_l}\\\frac{P_1X_l}{P_2X_l}\end{bmatrix}
$$
$$
\begin{bmatrix}u*P_2X_l-P_0X_l\\v*P_2X_l-P_1X_l\end{bmatrix} =
\begin{bmatrix}0\\0\end{bmatrix}
$$
$$
\begin{bmatrix}u*P_2-P_0\\v*P_2-P_1\end{bmatrix}\begin{bmatrix}X_l\end{bmatrix}
= \begin{bmatrix}0\\0\end{bmatrix}
$$

### Total Least squares
#### State definition
The motion of the robot is planar, so it is represented as a set of $SE(2)$
transformation matrices:\
$X_r = \{X_r^{[0]},X_r^{[1]}, ... , X_r^{[n]}\}$, $X_r^{[i]} \in SE(2)$.\
Landmarks are points in the space, so they are represented as a set of 3d
points:\
$X_l =  \{X_l^{[0]},X_l^{[1]}, ... , X_l^{[m]}\}$, $X_l^{[j]} \in \real^3$.\
The perturbation of the robot position is a vector in $\real^3$:
$
\Delta x_r^{[i]} =
\begin{bmatrix}\delta x \\ \delta y \\ \delta \theta
\end{bmatrix}
$\
The perturbation of the landmark position is a vector in $\real^3$:
$
\Delta x_l^{[j]} =
\begin{bmatrix}\delta x \\ \delta y \\ \delta z
\end{bmatrix}
$\
During the computation of the jacobian for the pose-projection system, the
state of the robot is converted to an $SE(3)$ representation, so it can be
combined with the camera displacement matrix.
#### Boxplus
The boxplus operator for the robot pose is:
$X_r^{[i]} \boxplus \Delta x_r^{[i]} = v2t(\Delta x_r^{[i]})X_r^{[i]}$\
The boxplus operator for the landmark position is obtained by simply adding
the perturbation to the current estimate:
$X_l^{[j]} \boxplus \Delta x_l^{[j]} = X_l^{[j]}+\Delta x_l^{[j]}$
#### Measurements
Pose measurements values are the relative motions between subsequent poses:\
$z_r^{[i,j]} \in SE(2) = (X_r^{[i]})^{-1}X_r^{[j]}$\
Projection measurements were directly available in the dataset.
#### Predictions
- Pose-Pose
$h(X_r^{i}\boxplus\Delta x_r^{[i]},X_r^{j}\boxplus\Delta x_r^{[j]})$ computed
as the pose measurements above
- Pose-Projection:
$h(X_r\boxplus \Delta x_r, X_l + \Delta x_l) =
\pi(K * {}^cX_r(X_r\boxplus \Delta x_r)^{-1}(X_l + \Delta x_l))$, with $\pi$
being the projection function, $K$ the intrinsics matrix of the camera,
${}^cX_r$ the transformation of the camera in the robot frame.
#### Error
- Pose-Pose error is computed using the Chordal distance between poses:
$$
flatten(h(X_r^{[i]}\boxplus\Delta x_r^{[i]},X_r^{[j]}\boxplus\Delta x_r^{[j]}))
-flatten(z_r^{[i,j]})
$$

- Pose-Projection error:
$h(X_r\boxplus \Delta x_r, X_l + \Delta x_l) - z_{proj}^{[i,j]}$

#### Jacobian
The resulting jacobian matrix are:
- Pose-Pose:
$$
J_j^{[i,j]} = \frac{d}{d \Delta x_r^{[j]}}
h(X_r^{[i]}, X_r^{j}\boxplus\Delta x_r^{[j]})=
\begin{bmatrix}
0_{4\times2} & flatten(R_i^T\begin{pmatrix} 0&-1\\1&0 \end{pmatrix}R_j)\\
R_i^T & flatten(R_i^T\begin{pmatrix} 0&-1\\1&0 \end{pmatrix}t_j)\\
\end{bmatrix}
$$
$$
J_i^{[i,j]}= -J_j^{[i,j]}
$$
- Pose-Projection:
$$
J_{i}^{[i,j]} = \frac{d}{d \Delta x_r^{[i]}}
h(X_r^{[i]}\boxplus \Delta x_r^{[i]}, X_l^{[j]} + \Delta x_l^{[j]})=J_{proj}K
\begin{bmatrix}
-R_c^TR^T\begin{pmatrix}0&1\\1&0\\0&0\end{pmatrix} &
R_c^TR^T\begin{pmatrix}0&1&0\\-1&0&0\\0&0&0\end{pmatrix}X_l
\end{bmatrix}
$$
$$
J_{j}^{[i,j]} = \frac{d}{d \Delta x_l^{[j]}}
h(X_r^{[i]}\boxplus \Delta x_r^{[i]}, X_l^{[j]} + \Delta x_l^{[j]})=
J_{proj}K[R_c^TR^T]
$$
$$
J_{proj}=\frac{d}{dp}\pi\Big|_{p=pcam} =
\begin{bmatrix}
\frac{1}{w}&0&-\frac{u}{w^2}\\
0&\frac{1}{w}&-\frac{v}{w^2}
\end{bmatrix}
$$


## Executing the program
- Create a conda environment
    ```
    conda create -n probrob python=3.9
    conda activate probrob
    ```
- Install the dependencies
    ```
    pip install -r requirements.txt
    ```
- Start the program by running
    ```
    python main.py
    ```

## Results
### Robot position
![TLSresults](output/odom.svg "Results")
### Residuals and inliers
![TLSresults](output/plot.svg "Results")
### Landmark visualization
![SCATTERresults](output/scatter.svg "Results")



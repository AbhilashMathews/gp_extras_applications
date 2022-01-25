# Multidimensional adaptive Gaussian process regression

To get started, here is a link to the original [gp_extras package](https://github.com/jmetzen/gp_extras) which should be installed along with [scikit-learn (sklearn)](https://github.com/scikit-learn/scikit-learn). These packages and scripts in this repository are developed with Python 3 compatibility, and installation is as simple as

    git clone git@github.com:scikit-learn/scikit-learn.git
    cd sklearn
    python setup.py install

and

    git clone git@github.com:jmetzen/gp_extras.git
    cd gp_extras
    python setup.py install

The multidimensional adaptive Gaussian process code is applied on experimental data with a few examples of potential outputs displayed below. Corresponding codes necessary for the creation of similar plots can be found [here](https://github.com/AbhilashMathews/gp_extras_applications/tree/master/codes). One example capability of this tool is the fitting of noisy multidimensional data as visualized below for both electron density and temperature measurements. This can yield time-varying plasma profiles and associated gradients with uncertainties.

![Output sample](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/2D_GPR-1160718013.gif)

Note for users: GP regression cannot fix the observed data, e.g. outliers at ψ < 0.95 towards the core breaking monotonicity. Good measurements with preprocessing will correspondingly help provide good fits, and the GP will help quantify what is already captured. Additionally, it is worth noting that running the same fitting routine on identical data sets can yield different results due to the stochastic optimization process employed. The script `GP-train.py` is currently configured to simply keep repeating until training is successful in case the optimization has not converged. Primary inputs to modify to help with stability during training include `N_clusters`, `l_samples_input`, `lower_l` (i.e. lower limit), and `upper_l` (i.e. upper limit).

![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/2D-GPR_n%2Bdndx.png)

Adaptive length scales can also be further analyzed across the spatiotemporal grid using `GP-obtain-2D-LLS.py`, if sought. It is worth noting that significantly different length scales can be learned across the input domain even after initializing identical adaptive kernels due to the stochastic nature of the optimization process. 

![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/lls_2d_n.png)

![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/lls_2D_n%2BT.png)

As described in the original [gp_extras package](https://github.com/jmetzen/gp_extras), the adaptive length scales and heteroscedastic noise model enable improved fitting in the above cases, and this is visually exemplified in the following 1-dimensional scenarios:

![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/lls_1d_data.png) ![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/lls_1d_scales.png)

![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/heteroscedastic.png) ![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/homoscedastic.png)

If you have any research questions or comments, please feel free to contact mathewsa@mit.edu

Note: This program is distributed in the hope that it will be useful, but without any warranty; not even the implied warranty of merchantibility or fitness for a particular service. Therefore, testing the code and ensuring it performs as you expect is important.

If you find this code useful, please remember to cite it: https://arxiv.org/abs/2103.01305

    @article{GPR_mathewsa2021,
    title={Quantifying Experimental Edge Plasma Evolution Via Multidimensional Adaptive Gaussian Process Regression},
    volume={49},
    ISSN={1939-9375},
    url={http://dx.doi.org/10.1109/TPS.2021.3123046},
    DOI={10.1109/tps.2021.3123046},
    number={12},
    journal={IEEE Transactions on Plasma Science},
    publisher={Institute of Electrical and Electronics Engineers (IEEE)},
    author={Mathews, Abhilash and Hughes, Jerry W.},
    year={2021},
    month={Dec},
    pages={3841–3847}}

This work is supported by the U.S. Department of Energy (DOE) Office of Science under the Fusion Energy Sciences program under contract DESC0014264 and the Natural Sciences and Engineering Research Council of Canada (NSERC) by the doctoral postgraduate scholarship (PGS D) and the Alcator C-Mod team. 

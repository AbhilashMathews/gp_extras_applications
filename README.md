# Multidimensional adaptive Gaussian process regression

To get started, here is a link to the original [gp_extras package](https://github.com/jmetzen/gp_extras) which should be installed along with [scikit-learn (sklearn)](https://github.com/scikit-learn/scikit-learn). 

Examples of potential outputs from the GP code are reproduced below: 

Analysis of the pedestal on Alcator C-Mod:

![Output sample](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/2D-GPR-1160718013.gif)

(Note for users: GP regression cannot fix the observed data, e.g. outliers at ψ < 0.95 towards the core breaking monotonicity. Good measurements with preprocessing will correspondingly help provide good fits, and the GP will help quantify what is already captured. Additionally, it is worth noting that running the same fitting routine on identical data sets can yield different results due to the stochastic optimization process employed.)

![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/2D-GPR_n%2Bdndx.png)

![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/heteroscedastic.png) ![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/homoscedastic.png)

![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/lls_1d_data.png)
![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/lls_1d_scales.png)

![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/lls_2d_n.png)

![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/lls_n.png | width=10) ![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/lls_t.png | width=10)


![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/n_profiles.png) ![alt tag](https://github.com/AbhilashMathews/gp_extras_applications/blob/master/outputs/T_profiles.png)

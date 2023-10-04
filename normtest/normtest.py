"""

##### List of functions (alphabetical order) #####

## Functions WITH good TESTS ###
- rj_critical_value(n, alpha=0.05)

## Functions WITH some TESTS ###
- ordered_statistics(n, method)
- ryan_joiner(x_data, alpha=0.05, method="ryan-joiner")

## Functions WITHOUT tests ###

##### List of CLASS (alphabetical order) #####

##### Dictionary of abbreviations #####



Author: Anderson Marcos Dias Canteli <andersonmdcanteli@gmail.com>

Created: September 22, 2023.

Last update: September 27, 2023



"""

##### IMPORTS #####

### Standard ###
from collections import namedtuple

### Third part ###
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy import interpolate

### home made ###
from .utils import checkers
from .utils import constants

##### CONSTANTS #####


##### CLASS #####
# rj_critical_value(n, alpha=0.05)
# ryan_joiner(x_data, alpha=0.05, method="blom", weighted=False)
# rj_correlation_plot(axes, x_data, method="blom", weighted=False)
# rj_dist_plot(axes, x_data, method="blom", min=4, max=50, deleted=False, weighted=False)
# rj_dist_plot(axes, x_data, method="blom", min=4, max=50, deleted=False, weighted=False)
# rj_p_value(statistic, n)
# ordered_statistics(n, method)
##### FUNCTIONS #####





## RYAN - JOINER TEST ##

# com testes ok
def rj_critical_value(n, alpha=0.05):
    """This function calculates the critical value of the Ryan-Joiner test [1]_
    
    Parameters
    ----------
    n : ``int``
        The sample size. Must be greater than ``3``;
    alpha : ``float``, optional
        The level of significance (``ɑ``). Must be ``0.01``, ``0.05`` (default) or ``0.10``;
        
    Returns
    -------        
    critical : ``float``
        The critical value of the test
        
    See Also
    --------        
    pass
    

    Notes
    -----
    The critical values are calculated using [1]_ the following equations:
    
    .. math::

            R_{p;\\alpha=0.10}^{'} = 1.0071 - \\frac{0.1371}{\\sqrt{n}} - \\frac{0.3682}{n} + \\frac{0.7780}{n^{2}}
            
            R_{p;\\alpha=0.05}^{'} = 1.0063 - \\frac{0.1288}{\\sqrt{n}} - \\frac{0.6118}{n} + \\frac{1.3505}{n^{2}}
            
            R_{p;\\alpha=0.01}^{'} = 0.9963 - \\frac{0.0211}{\\sqrt{n}} - \\frac{1.4106}{n} + \\frac{3.1791}{n^{2}}

    where :math:`n` is the sample size.


    References
    ----------
    .. [1]  RYAN, T. A., JOINER, B. L. Normal Probability Plots and Tests for Normality, Technical Report, Statistics Department, The Pennsylvania State University, 1976. Available at `www.additive-net.de <https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner>`_. Access on: 22 Jul. 2023.


    Examples
    --------
    >>> rj_critical_value(10, alpha=0.05)  
    0.9178948637370312

    """
    checkers._check_param_values(alpha, "alpha", param_values=[0.01, 0.05, 0.10])
    checkers._check_is_integer(n, "n")
    checkers._check_value_is_equal_or_higher_than(n, "n", 4)
    
    if alpha == 0.1:
        return 1.0071 - (0.1371 / np.sqrt(n)) - (0.3682 / n) + (0.7780 / n**2)
    elif alpha == 0.05:
        return 1.0063 - (0.1288 / np.sqrt(n)) - (0.6118 / n) + (1.3505 / n**2)
    else: # alpha == 0.01: 
        return 0.9963 - (0.0211 / np.sqrt(n)) - (1.4106 / n) + (3.1791 / n**2)

# com testes ok
def ryan_joiner(x_data, alpha=0.05, method="blom", weighted=False):
    """This function applies the Ryan-Joiner Normality test [1]_.

    Parameters
    ----------
    x_data : ``numpy array``
        One dimension :doc:`numpy array <numpy:reference/generated/numpy.array>` with at least ``4`` observations.
    alpha : ``float``, optional
        The level of significance (``ɑ``). Must be ``0.01``, ``0.05`` (default) or ``0.10``;
    method : ``str``, optional
        A string with the approximation method that should be adopted. The options are ``"blom"`` (default), ``"blom2"``, ``"blom3"`` or ``"filliben"``. See `ordered_statistics` for details.
    weighted : ``bool``, optional
        Whether to estimate the Normal order considering the repeats as its average (``True``) or not (``False``, default). Only has an effect if the dataset contains repeated values
        
        
    Returns
    -------        
    result : ``tuple`` with
        statistic : ``float``
            The test statistic.
        critical : ``float``
            The critical value.
        p_value : ``float`` or ``str``
            The probability of the test
        conclusion : ``str``
            The test conclusion (e.g, Normal/Not Normal).


    See Also
    --------        
    ordered_statistics
    

    Notes
    -----
    The test statistic (:math:`R_{p}`) is estimated through the correlation between the ordered data and the Normal statistical order:
    
    .. math::

            R_{p}=\\dfrac{\\sum_{i=1}^{n}x_{(i)}z_{(i)}}{\\sqrt{s^{2}(n-1)\\sum_{i=1}^{n}z_{(i)}^2}}

    where :math:`z_{(i)}` values are the z-score values of the corresponding experimental data (:math:`x_({i)}`) value and :math:`s^{2}` is the sample variance.
    
    The correlation is estimated using ``stats.pearsonr()``.

    The Normality test has the following assumptions:
    
    .. admonition:: \u2615

       :math:`H_0:` Data was sampled from a Normal distribution.

       :math:`H_1:` The data was sampled from a distribution other than the Normal distribution.


    The conclusion of the test is based on the comparison between the ``critical`` value (at ``ɑ`` significance level) and ``statistic`` of the test:

    .. code:: python

       if critical <= statistic:
           Fail to reject :math:`H_0:` (e.g., data is Normal)
       else:
           Reject :math:`H_0:` (e.g., data is not Normal)



    References
    ----------
    .. [1]  RYAN, T. A., JOINER, B. L. Normal Probability Plots and Tests for Normality, Technical Report, Statistics Department, The Pennsylvania State University, 1976. Available at `www.additive-net.de <https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner>`_. Access on: 22 Jul. 2023.


    Examples
    --------
    >>> x_data = np.array([1.90642, 2.22488, 2.10288, 1.69742, 1.52229, 3.15435, 2.61826, 1.98492, 1.42738, 1.99568])
    >>> result = ryan_joiner(x_data)
    (0.9599407779411523, 0.9178948637370312, 'Fail to Reject H0')
    """
    checkers._check_is_numpy_1_D(x_data, 'x_data')
    checkers._check_is_bool(weighted, "weighted")

    # ordering
    x_data = np.sort(x_data)    
    if weighted:
        df = pd.DataFrame({
            "x_data": x_data
        })
        # getting mi values    
        df["Rank"] = np.arange(1, df.shape[0]+1)
        df["Ui"] = ordered_statistics(x_data.size, method=method)
        df["Mi"] = df.groupby(["x_data"])["Ui"].transform('mean')
        normal_ordered = stats.norm.ppf(df["Mi"])
    else:
        ordered = ordered_statistics(x_data.size, method=method)
        normal_ordered = stats.norm.ppf(ordered)

    # calculatiing the stats
    statistic = stats.pearsonr(normal_ordered, x_data)[0]
    # getting the critical values
    critical_value = rj_critical_value(x_data.size, alpha=alpha)
    # conclusion
    if statistic < critical_value:
        conclusion = constants.REJECTION
    else:
        conclusion = constants.ACCEPTATION
    # pvalue
    p_value = rj_p_value(statistic, x_data.size)
    result = namedtuple("RyanJoiner", ("statistic", "critical", "p_value", "conclusion"))
    return result(statistic, critical_value, p_value, conclusion)


# com alguns testes
def rj_correlation_plot(axes, x_data, method="blom", weighted=False):
    """This function creates an axis with the Ryan-Joiner test correlation graph

    Parameters
    ----------
    axes : ``matplotlib.axes.SubplotBase``
        The axes to plot    
    x_data : ``numpy array``
        One dimension :doc:`numpy array <numpy:reference/generated/numpy.array>` with at least ``4`` observations.
    method : ``str``
        A string with the approximation method that should be adopted. The options are ``"blom"`` (default), ``"blom2"``, ``"blom3"`` or ``"filliben"``. See `ordered_statistics` for details.
    weighted : ``bool``, optional
        Whether to estimate the Normal order considering the repeats as its average (``True``) or not (``False``, default). Only has an effect if the dataset contains repeated values
        
    Returns
    -------        
    axes : ``matplotlib.axes._subplots.AxesSubplot``
        The axis of the graph.

    See Also
    --------        
    ordered_statistics
    
    Examples
    --------
    
    """
    constants.warning_plot()
    checkers._check_is_subplots(axes, "axes")
    checkers._check_is_numpy_1_D(x_data, 'x_data')

    # ordering the sample
    x_data = np.sort(x_data)    
    
    # ordering
    x_data = np.sort(x_data)    
    if weighted:
        df = pd.DataFrame({
            "x_data": x_data
        })
        # getting mi values    
        df["Rank"] = np.arange(1, df.shape[0]+1)
        df["Ui"] = ordered_statistics(x_data.size, method=method)
        df["Mi"] = df.groupby(["x_data"])["Ui"].transform('mean')
        normal_ordered = stats.norm.ppf(df["Mi"])
    else:
        ordered = ordered_statistics(x_data.size, method=method)
        normal_ordered = stats.norm.ppf(ordered)

    # performing regression
    reg = stats.linregress(normal_ordered, x_data)
    # pred data
    y_pred = normal_ordered*reg.slope + reg.intercept
    
    ## making the plot

    # adding the data
    axes.scatter(normal_ordered, x_data, fc="none", ec="k")
   
    # adding the trend line
    axes.plot(normal_ordered, y_pred, c="r")
    
    # adding the statistic
    text = "$R_{p}=" + str(round(reg.rvalue,4)) + "$"
    axes.text(.1, .9, text, ha='left', va='center', transform=axes.transAxes)
    
    # perfuming
    axes.set_xlabel("Normal statistical order")
    axes.set_ylabel("Ordered data")    
    
    return axes


# com alguns testes
def rj_dist_plot(axes, x_data, method="blom", min=4, max=50, deleted=False, weighted=False):
    """This function generates axis with critical data from the Ryan-Joiner Normality test
    
    Parameters
    ----------
    axes : ``matplotlib.axes.SubplotBase``
        The axes to plot    
    x_data : ``numpy array``
        One dimension :doc:`numpy array <numpy:reference/generated/numpy.array>` with at least ``4`` observations.
    method : ``str``
        A string with the approximation method that should be adopted. The options are ``"blom"`` (default), ``"blom2"``, ``"blom3"`` or ``"filliben"``. See `ordered_statistics` for details.
    min : ``int``
        The lower range of the number of observations for the critical values (default is ``4``);
    max : ``int``
        The upper range of the number of observations for the critical values (default is ``50``);      
    deleted : ``bool``
        Whether it is (``True``) to insert the deleted data method or not (``False``, default). This function is only for exploring possibilities
    weighted : ``bool``, optional
        Whether to estimate the Normal order considering the repeats as its average (``True``) or not (``False``, default). Only has an effect if the dataset contains repeated values
        
    Returns
    -------        
    axes : ``matplotlib.axes._subplots.AxesSubplot``
        The axis of the graph.


    See Also
    --------        
    ryan_joiner
    

    Notes
    -----
    O método deleted consiste em aplicar o testes nos n-1 subconjuntos de dados obtidos com a remoção de 1 ponto do conjunto de dados.


    References
    ----------
    .. [1]  RYAN, T. A., JOINER, B. L. Normal Probability Plots and Tests for Normality, Technical Report, Statistics Department, The Pennsylvania State University, 1976. Available at `www.additive-net.de <https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner>`_. Access on: 22 Jul. 2023.


    Examples
    --------    
    """
    constants.warning_plot()
    checkers._check_is_subplots(axes, "axes")
    checkers._check_is_numpy_1_D(x_data, 'x_data')
    checkers._check_is_float_or_int(min, "min")
    checkers._check_is_float_or_int(max, "max")
    checkers._check_a_lower_than_b(min, max, "min", "max")
    checkers._check_is_bool(deleted, "deleted")
    checkers._check_is_bool(weighted, "weighted")

    if x_data.size > max:
        print(f"The graphical visualization is best suited if the sample size is smaller ({x_data.size}) than the max value ({max}).")
    if x_data.size < min:
        print(f"The graphical visualization is best suited if the sample size is greater ({x_data.size}) than the min value ({min}).")        
        
    n_samples = np.arange(min,max+1)    
    alphas = [0.10, 0.05, 0.01]
    alphas_label = ["$R_{p;10\%}^{'}$", "$R_{p;5\%}^{'}$", "$R_{p;1\%}^{'}$"]
    colors = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
            (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
            (0.3333333333333333, 0.6588235294117647, 0.40784313725490196)]

    # main test
    result = ryan_joiner(x_data, method=method, weighted=weighted)
    axes.scatter(x_data.size, result[0], color="r", label="$R_{p}$")
    
    # adding critical values
    for alp, color, alp_label in zip(alphas, colors, alphas_label):
        criticals = []
        for sample in n_samples:
            criticals.append(rj_critical_value(sample, alpha=alp))
        axes.scatter(n_samples, criticals, label=alp_label, color=color, s=10)
    
    # if robust
    if deleted:
        for i in range(x_data.size):
            x_reduced = np.delete(x_data, i)
            result = ryan_joiner(x_reduced, method=method, weighted=weighted)
            if i == 0:            
                axes.scatter(x_reduced.size, result[0], ec="k", fc="none", label="Deleted")
            else:
                axes.scatter(x_reduced.size, result[0], ec="k", fc="none")
        axes.set_title("Ryan-Joiner Normality test - robust approach")
    else:
        axes.set_title("Ryan-Joiner Normality test")
    
    # adding details
    axes.legend(loc=4)   
    axes.set_xlabel("Sample size")
    axes.set_ylabel("Critical value")
    
    return axes

# com testes ok
def rj_p_value(statistic, n):
    """This function estimates the probability associated with the Ryan-Joiner Normality test

    Parameters
    ----------
    statistic : ``float`` (positive)
        The Ryan-Joiner test statistics
    n : ``int``
        The sample size. Must be greater than ``3``;        

    Returns
    -------        
    p_value : ``float`` or ``str``
        The probability of the test
        
    See Also
    --------        
    ryan_joiner
    

    Notes
    -----
    The test probability is estimated through linear interpolation of the test statistic with critical values from the Ryan-Joiner test [1]_. The Interpolation is performed using the ``stats.interpolate.interp1d`` function.
    
    * If the test statistic is greater than the critical value for :math:`\alpha=0.10`, the result is always "p > 0.100".
    * If the test statistic is lower than the critical value for :math:`\alpha=0.01`, the result is always "p < 0.010""



    References
    ----------
    .. [1]  RYAN, T. A., JOINER, B. L. Normal Probability Plots and Tests for Normality, Technical Report, Statistics Department, The Pennsylvania State University, 1976. Available at `www.additive-net.de <https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner>`_. Access on: 22 Jul. 2023.


    Examples
    --------
    >>> p_value = nm.rj_p_value(.90, 10)
    >>> print(p_value)  
    0.030930589077996555

    """
    checkers._check_data_in_range(statistic, "statistic", 0, 1)
    alphas = np.array([0.10, 0.05, 0.01])
    criticals = np.array([rj_critical_value(n, alpha=alphas[0]), rj_critical_value(n, alpha=alphas[1]), rj_critical_value(n, alpha=alphas[2])])
    f = interpolate.interp1d(criticals, alphas)
    if statistic > max(criticals):
        return "p > 0.100"
    elif statistic < min(criticals):
        return "p < 0.010"
    else:
        p_value = f(statistic) 
        return p_value


## GENERIC FUNCTIONS ##

# falta teste para o filliben
def ordered_statistics(n, method):
    """This function estimates the statistical order (:math:`m_{i}`) using some approximations
    
    Parameters
    ----------
    n : ``int``
        The sample size. Must be greater than ``3``;
    method : ``str``
        A string with the approximation method that should be adopted. The options are ``"blom"``, ``"blom2"``, ``"blom3"`` or ``"filliben"``. See more details in the Notes section.
        
        
    Returns
    -------        
    mi : :doc:`numpy array <numpy:reference/generated/numpy.array>`
        The estimated statistical order (:math:`m_{i}`)
        
    See Also
    --------        
    ryan_joiner
    

    Notes
    -----
    * If ``method=="blom"``, :math:`m_{i}` is estimated using [1]_:
    
    .. math::

            m_{i} = \\frac{i - 3/8}{n + 1/4}
            
    * If ``method=="blom2"``, :math:`m_{i}` is estimated using [1]_:
    
    .. math::

            m_{i} = \\frac{i}{n + 1}            
            
    * If ``method=="blom3"``, :math:`m_{i}` is estimated using [1]_:
    
    .. math::

            m_{i} = \\frac{i - \\frac{1}{2}}{n}            
            
    * If ``method=="filliben"``, :math:`m_{i}` is estimated using [3]_:
    
    .. math::

            m_{i} = \\begin{cases}1-0.5^{1/n} & i = 1\\\ \\frac{i-0.3175}{n+0.365} & i = 2, 3,  \\ldots , n-1 \\\ 0.5^{1/n}& i=n \\end{cases}
            
    where :math:`n` is the sample size and :math:`i` is the ith observation.

    In the implementations of the Ryan-Joiner test in Minitab and Statext software, the bloom method is used, which is cited by [2]_ as an alternative

    References
    ----------
    .. [1] BLOM, G. Statistical Estimates and Transformed Beta-Variables. New York: John Wiley and Sons, Inc, p. 71-72, 1958.

    .. [2]  RYAN, T. A., JOINER, B. L. Normal Probability Plots and Tests for Normality, Technical Report, Statistics Department, The Pennsylvania State University, 1976. Available at `www.additive-net.de <https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner>`_. Access on: 22 Jul. 2023.

    .. [3] FILLIBEN, J. J. The Probability Plot Correlation Coefficient Test for Normality. Technometrics, 17(1), 111–117, (1975). Available at `doi.org/10.2307/1268008 <https://doi.org/10.2307/1268008>`_.
    



    Examples
    --------
    >>> from normtest import normtest as nm
    >>> result = nm.ordered_statistics(7, method="blom") 
    >>> print(result)
    [0.0862069  0.22413793 0.36206897 0.5        0.63793103 0.77586207
    0.9137931 ]

    """
    
    checkers._check_param_values(method, "method", param_values=["blom", "blom2", "blom3", "filliben"])
    checkers._check_is_integer(n, "n")
    checkers._check_value_is_equal_or_higher_than(n, "n", 4)

    i = np.arange(1,n+1)
    if method == "blom":
        mi = (i - 3/8)/(n + 1/4)  
    elif method == "blom2":
        mi = i/(n+1)
    elif method == "blom3":
        mi = (i - 1/2)/(n)  
    else: # method == "filliben":
        mi = (i-0.3175)/(n+0.365)
        mi[0] = 1 - 0.5**(1/n)
        mi[-1] = 0.5**(1/n)
  
    return mi
    

def normal_distribution_plot(axes, n_rep, seed=None, xinfo=[0.00001, 0.99999, 1000], loc=0.0, scale=1.0, safe=False):
    """This function draws a normal distribution chart with the experimental points and the distribution histogram

    Parameters
    ----------
    axes : ``matplotlib.axes.SubplotBase``
        The axes to plot    
    n_rep : ``int`` (positive)
        The number of samples in the dataset (must be greater than 3)
    seed : ``int`` (optional)
        The seed used to obtain random data from the Normal distribution. The default value is ``None`` which results in a random seed
    xinfo : ``list`` (optional)
        A list with three elements:
        * ``xinfo[0]`` the smallest value used as Percent point (positive, default is ``10-6``);
        * ``xinfo[1]`` the highest value used as Percent point (positive, default is ``1-10-6``);;
        * ``xinfo[2]`` the number of equally spaced points that are generated to estimate the Normal distribution (positive, default is ``1000``);;
    loc : ``float`` or ``int`` (optional)
        The loc parameter of the Normal distribution (default is ``0.0``)
    scale : ``float`` or ``int`` (optional)
        The scale parameter of the Normal distribution (positive, default is ``1.0``)
    safe : ``bool`` (optional)
        Whether to check the inputs before performing the calculations (``True``) or not (``False``, default). Useful for beginners to identify problems in data entry (may reduce algorithm execution time).

    Returns
    -------        
    axes : ``matplotlib.axes._subplots.AxesSubplot``
        The axis of the graph.

    See Also
    --------        
    
    
    Examples
    --------
        
    """
    constants.warning_plot()
    if safe:
        checkers._check_is_subplots(axes, "axes")
        if seed is not None:
            checkers._check_is_integer(seed, "seed")
            checkers._check_is_positive(seed, "seed")
        checkers._check_is_integer(n_rep, "n_rep")
        checkers._check_value_is_equal_or_higher_than(n_rep, "n_rep", 4)
        checkers._check_is_list(xinfo, "xinfo")
        checkers._check_list_size(xinfo, "xinfo", 3)
        checkers._check_is_float_or_int(xinfo[0], "xinfo[0]")
        checkers._check_data_in_range(xinfo[0], "xinfo[0]", 1e-6, 1-1e-6)
        checkers._check_data_in_range(xinfo[1], "xinfo[1]", 1e-6, 1-1e-6)
        checkers._check_is_integer(xinfo[2], "xinfo[2]")
        checkers._check_is_positive(xinfo[2], "xinfo[2]")
        checkers._check_is_float_or_int(loc, "loc")
        checkers._check_is_float_or_int(scale, "scale")
        checkers._check_is_positive(scale, "scale")
        
    x = np.linspace(stats.norm.ppf(xinfo[0]), stats.norm.ppf(xinfo[1]), xinfo[2])
    axes.plot(x, stats.norm.pdf(x), ls='--', c="gray", label='Theoretical')
    rng = np.random.default_rng(seed)
    normal_data = rng.normal(loc=loc, scale=scale, size=n_rep)
    axes.scatter(normal_data, stats.norm.pdf(normal_data), color="r", alpha=.6, label=f'Seed = {seed}')
    axes.hist(normal_data, density=True, alpha=0.5, color="skyblue")
    axes.text(0.025, 0.95, f"kurtosis={round(stats.kurtosis(normal_data) + 3, 3)}", 
            horizontalalignment='left', verticalalignment='center', transform=axes.transAxes)
    axes.text(0.025, 0.85, f"skew={round(stats.skew(normal_data), 3)}", 
            horizontalalignment='left', verticalalignment='center', transform=axes.transAxes)
    axes.set_title(f"Seed = {seed}")


    return axes

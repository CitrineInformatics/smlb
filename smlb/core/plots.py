"""Figures and plots.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2019-2020, Matthias Rupp, Citrine Informatics.

Graphical summaries of results.
Uses matplotlib, a Python plotting library available at https://matplotlib.org/

A note on interactive use in Jupyter (lab) notebooks:
With the default `%matplotlib inline` setting, matplotlib will output plots
as soon as drawing commands have been issued, even without `plt.show()`.
A backend that suppresses output is `%matplotlib agg`. Alternatively,
(i) `plt.ioff()` can be used to disable inline mode, or,
(ii) `%%capture` cab be used, which prevents any output of the cell.

An implementation enabling different plotting backends was put on hold 
due to increased complexity (pull request #22).
"""

from typing import Optional, Union, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from smlb import BenchmarkError, InvalidParameterError
from smlb import is_sequence
from smlb import params
from smlb import Evaluation, EvaluationConfiguration

# todo: A possible optimization is to not re-render Plots upon successive calls to render().
#       Example: rendering to several different file formats.

# todo: add auxiliary information to Evaluations, in particular LearningCurvePlot

# todo: set top and right axes labels


class PlotConfiguration(EvaluationConfiguration):
    """Configuration settings for Plots.

    Provides backend-independent basic settings:
    * font sizes
    * color sets
    """

    # Color sets

    PREDEFINED_COLORSETS = {
        # color set from
        #     Masataka Okabe, Kei Ito: Color Universal Design (CUD). How to make figures
        #     and presentations that are friendly to Colorblind people, 2008.
        # via https://betterfigures.org. Ordering of colors was changed
        0: np.asfarray(
            [
                (0, 114, 178),
                (230, 159, 0),
                (0, 158, 115),
                (240, 228, 66),
                (0, 0, 0),
                (213, 94, 0),
                (204, 121, 167),
                (86, 180, 233),
            ]
        )
        / 255,
        # color set inspired by Wolfram Research Mathematica 'Scientific' color scheme
        1: np.asfarray(
            [
                (0.945, 0.594, 0.000),
                (0.900, 0.360, 0.054),
                (0.365, 0.428, 0.758),
                (0.646, 0.253, 0.685),
                (0.286, 0.560, 0.451),
                (0.700, 0.336, 0.000),
                (0.491, 0.345, 0.800),
                (0.718, 0.569, 0.000),
                (0.707, 0.224, 0.542),
                (0.287, 0.490, 0.665),
                (0.982, 0.577, 0.012),
                (0.588, 0.288, 0.750),
                (0.426, 0.558, 0.278),
                (0.943, 0.415, 0.071),
                (0.415, 0.394, 0.784),
            ]
        ),
    }

    def __init__(self, font_size: int = 11, color_set: int = 1, **kwargs):
        """Initialize plot configuration.

        Parameters:
            font_size: base font size in absolute points
            color_set: color scheme
        """

        super().__init__(**kwargs)

        self._font_size = params.integer(font_size, above=0)
        self._color_set = self.PREDEFINED_COLORSETS[
            params.integer(color_set, from_=0, below=len(self.PREDEFINED_COLORSETS))
        ]

    @property
    def font_size(self):
        """Query base font size in absolute points"""

        return self._font_size

    @property
    def color_set(self):
        """Query color set in RGB color space

        Returns:
            n x 3 array, where n is number of colors and each entry consists of
                red, green and blue values in the range [0,1]
        """

        return self._color_set

    def color(self, i: int):
        """Query color from current colorset.

        Color in RGB color space. Colors do not cycle.

        Parameters:
            i: index of queried color

        The rationale for not cycling colors is to alert the user that there are not enough
        unique colors, as opposed to not being able to tell apart elements of the plot.
        If cycling is desired, pass i mod length of color scheme.
        """

        i = params.integer(i, from_=0, below=len(self._color_set))

        return self._color_set[i]


class Plot(Evaluation):
    """Base class for static graphical outcomes.

    Uses matplotlib (https://matplotlib.org) as rendering engine.

    Provides basic plotting abstractions so that specific plots
    can focus on the plot logic, as opposed to rendering details.
    """

    def __init__(
        self,
        target=None,
        configuration: Optional[PlotConfiguration] = None,
        axes_labels=(None, None, None, None),
        axes_scales=("linear", "linear"),
        **kwargs,
    ):
        """Initialize Evaluation.

        Parameters:
            target: rendering target that evaluation outcome is rendered to;
                can be a single filename, or a matplotlib Axes or (Figure, Axes) pair,
                or a sequence thereof; if a matplotlib Axes or (Figure, Axes) pair,
                evaluation will add to it; if None, a new rendering target is created
            configuration: optional plot configuration controlling rendering details
            axes_labels: labels for all axes (bottom, left, top, right), None to not label an axis;
                         for shorter tuples remaining entries are assumed None, so ('x', 'y') is valid
            axes_scales: scales ("linear" or "log") for horizontal and vertical axes

        Examples:
            __init__(axes_labels=("bottom", "left", "top"))  # right is None
            __init__(axes_scales=("log", "log"))
        """

        configuration = params.any_(
            configuration, lambda arg: params.instance(arg, PlotConfiguration), params.none
        )

        super().__init__(configuration=configuration, **kwargs)

        # Axes, (Figure, Axes), filename, None, or sequence (without None)
        target_f = lambda arg: params.any_(
            arg,
            lambda arg: params.instance(arg, mpl.axes.Axes),
            lambda arg: params.tuple_(
                arg,
                lambda arg: params.instance(arg, mpl.figure.Figure),
                lambda arg: params.instance(arg, mpl.axes.Axes),
                arity=2,
            ),
            params.string,
        )
        self._target = params.any_(
            target, target_f, params.none, lambda arg: params.tuple_(arg, target_f)
        )

        self._axes_labels = params.tuple_(
            axes_labels,
            lambda arg: params.any_(arg, params.string, params.none),
            arity=4,
            default=None,
        )

        self._axes_scales = params.tuple_(
            axes_scales, lambda arg: params.enumeration(arg, {"linear", "log"}), arity=2
        )

        self._figaxis = None

    def _default_configuration(self):
        """Query default Plot configuration.

        Factory method that returns default plot configuration."""

        return PlotConfiguration()

    def _fontdict(self):
        """Construct matplotlib fontdict argument from configuration."""

        return {
            "fontsize": self.configuration.font_size,
            "fontfamily": "sans-serif",
            "fontstretch": "normal",
            "fontstyle": "normal",
            "fontvariant": "normal",
            "fontweight": "normal",
        }

    def render(self):
        """Renders evaluation.

        Specific derived classes should override `_render`, not this method.
        """

        target = self._target  # shortcut

        # if sequence of targets, render each of them
        if is_sequence(target) and not isinstance(target[0], mpl.figure.Figure):
            for tgt in target:
                self.render(tgt)
            return

        # process single target
        if isinstance(target, mpl.axes.Axes):
            target = (plt.gcf(), target)

        # remember filename for export
        if isinstance(target, str):
            filename = target
            target = None
        else:
            filename = None

        # create new plot if necessary
        owner = False
        if target is None:
            owner = True
            target = plt.subplots()

        self._figax = target  # tuple(Figure, Axes)

        # set matplotlib plot settings
        # at this time, settings such as axes labels or scales contain the
        # correct values, but have not been set yet as the figure and axes
        # were just created. re-assignment sets (or 'activates') these values
        # for the new figure and axes.
        self.axes_labels = self.axes_labels
        self.axes_scales = self.axes_scales

        self._render(target)

        # export to filename if requested
        if filename is not None:
            self.fig.savefig(filename, bbox_inches="tight", pad_inches=0)

        # clean up if owner of Axes
        if owner:
            plt.close(self.fig)  # fig.clear() might not release all memory
            self._figax = None

    @property
    def fig(self):
        """Access underlying matplotlib Figure."""

        return self._figax[0]

    @property
    def ax(self):
        """Access underlying matplotlib Axes."""

        return self._figax[1]

    @property
    def fig_ax(self):
        """Access underlying matplotlib Figure and Axes."""

        return self._figax

    @property
    def axes_labels(self):
        """Query axes labels."""

        return self._axes_labels

    @axes_labels.setter
    def axes_labels(self, labels=(None, None, None, None), **kwargs):
        """Set axes labels.

        Parameters:
            axes_labels: labels for bottom, left, top, right axes
                None indicates to use the current value

        Examples:
            axes_labels = (None, "y")  # set only left axis label
        """

        string_or_none_f = lambda arg: params.any_(arg, params.string, params.none)
        labels = params.tuple_(labels, string_or_none_f, arity=4, default=None)

        # re-assign tuple as a whole
        self._labels = tuple(
            self.axes_labels[i] if labels[i] is None else labels[i] for i in range(4)
        )

        # set labels if specified (not None)
        # this allows to pass kwargs specific to one axis
        if labels[0] is not None:
            self.ax.set_xlabel(labels[0], fontdict=self._fontdict(), **kwargs)
        if labels[1] is not None:
            self.ax.set_ylabel(labels[1], fontdict=self._fontdict(), **kwargs)
        if labels[2] is not None or labels[3] is not None:
            # todo; possible implementation via xtwin/ytwin, storing these axes in outcome
            raise NotImplementedError

    @property
    def axes_scales(self):
        """Query axes scales."""

        return self._axes_scales

    @axes_scales.setter
    def axes_scales(self, scales=(None, None), **kwargs):
        """Set axes scales.

        Parameters:
            axes_scales: scales (None, "linear" or "log") for horizontal and vertical axes;
                None indicates to use the current value

        Examples:
            axes_scales = (None, "log")  # change only vertical axis
        """

        scale_or_none_f = lambda arg: params.any_(
            arg, lambda arg: params.enumeration(arg, {"linear", "log"}), params.none
        )
        scales = params.tuple_(scales, scale_or_none_f, arity=2, default=None)

        # re-assign tuple as a whole
        self._scales = (
            self.axes_scales[0] if scales[0] is None else scales[0],
            self.axes_scales[1] if scales[1] is None else scales[1],
        )

        # set axes if specified (not None)
        # this allows to pass kwargs specific to one axis
        if scales[0] is not None:
            self.ax.set_xscale(scales[0], **kwargs)
        if scales[1] is not None:
            self.ax.set_yscale(scales[1], **kwargs)

    def points(self, points, color=0, **kwargs):
        """Draw set of points.

        Parameters:
            points: n x 2 matrix of n points in two dimensions
            color: color index
        """

        points = params.real_matrix(points, ncols=2)
        color = params.integer(color, from_=0, below=len(self.configuration.color_set))

        self.ax.plot(
            points[:, 0],
            points[:, 1],
            linestyle="",
            marker="o",
            color=self.configuration.color(color),
            **kwargs,
        )

    def line(self, line, color=0, **kwargs):
        """Draw a line.

        Parameters:
            line: n x 2 matrix of n points in two dimensions
            color: color index
        """

        line = params.real_matrix(line, ncols=2)
        color = params.integer(color, from_=0, below=len(self.configuration.color_set))

        self.ax.plot(
            line[:, 0], line[:, 1], linestyle="-", color=self.configuration.color(color), **kwargs
        )

    def box_whisker(self, positions, values, color=0, widths=0.5, **kwargs):
        """Draw box-whisker plots.

        Parameter:
            positions: where to place plots on horizontal axis
            values: samples for each location
            color: color index
            widths: widths of boxes
        """

        positions = params.real_vector(positions)
        point_set_f = lambda arg: params.real_vector(arg)
        values = params.tuple_(values, params.real_vector, arity=len(positions))
        color = params.integer(color, from_=0, below=len(self.configuration.color_set))
        widths = params.real_vector(widths, dimensions=len(positions), domain=(0, 999))

        color = self.configuration.color(color)

        self.ax.boxplot(
            values,
            positions=positions,
            whis="range",
            bootstrap=None,
            widths=widths,
            notch=False,
            showmeans=True,
            boxprops={"color": color},
            whiskerprops={"color": color},
            capprops={"color": color},
            meanprops={"marker": "*", "markerfacecolor": color, "markeredgecolor": color},
            medianprops={"color": color},
            manage_ticks=False,
            **kwargs,
        )

    def shaded_line(
        self,
        positions: np.ndarray,
        values: List[np.ndarray],
        color_idx: int = 0,
        label: Optional[str] = None,
        quantile_width: float = 0.5,
        alpha: float = 0.2,
        show_extrema: bool = True,
        **kwargs,
    ):
        """Draw a line plot with shaded quantiles.

        Parameters:
            positions: 1-d array of point locations on the horizontal axis
            values: list of arrays, each one containing all of the values at a given location.
                len(values) must equal len(positions)
            color_idx: color index
            label: line label
            quantile_width: fraction of the range to shade. For the default value, 0.5,
                shade from the 25th percentile to the 75th percentile.
            alpha: shading alpha level
            show_extrema: whether or not to draw dashed lines at the best/worst point
        """
        positions = params.real_vector(positions)
        values = params.tuple_(values, params.real_vector, arity=len(positions))
        color_idx = params.integer(color_idx, from_=0, below=len(self.configuration.color_set))
        quantile_width = params.real(quantile_width, from_=0, to=1)
        alpha = params.real(alpha, from_=0, to=1)

        color = self.configuration.color(color_idx)
        lower_bound = 0.5 - quantile_width / 2.0
        upper_bound = 0.5 + quantile_width / 2.0

        median = [np.median(samples) for samples in values]
        lower_shading = [np.quantile(samples, lower_bound) for samples in values]
        upper_shading = [np.quantile(samples, upper_bound) for samples in values]

        self.ax.plot(positions, median, linestyle="-", color=color, label=label, **kwargs)
        self.ax.fill_between(
            positions,
            lower_shading,
            upper_shading,
            color=color,
            alpha=alpha,
            **kwargs,
        )

        if show_extrema:
            min_val = [np.min(samples) for samples in values]
            max_val = [np.max(samples) for samples in values]
            self.ax.plot(positions, min_val, linestyle="--", color=color, **kwargs)
            self.ax.plot(positions, max_val, linestyle="--", color=color, **kwargs)


class GeneralizedFunctionPlot(Plot):
    """Plot generalized functions.

    The term "generalized function" here means set-valued functions on the real axis.

    Plots multiple outputs (vertical axis) associated to inputs (horizontal axis).
    Multiple outputs per input can be visualized as points or box-whisker plots.

    If several functions provide values at the same horizontal location,
    the corresponding visualizations can be rectified (horizontally displaced)
    so as to not overlap and thus improve readability of the plot.
    """

    # Current limitations:
    # - horizontal rectification automatic determination chooses a maximal value
    #   possible improvement: choose factor based on range, with current factor as upper bound
    # - horizontal rectification is global
    #   possible improvement: allow rectification factor per horizontal site and
    #   remove 'gaps' from curves not having values at that site

    def __init__(
        self, visualization_type: str = "points", rectify: Union[float, bool] = False, **kwargs
    ):
        """Initialize generalized function plot.

        Parameters:
            visualization_type: how to visualize generalized functions.
                Either single value or list of appropriate length.
                Possible values: "points" (default), "box-whisker", "shaded-line"
            rectify: whether and by how much each curves' values will be horizontally displaced
                to visually disentangle markers from different curves at the same location.
                True indicates automatic displacement, False indicates no displacement.
                If not specified, horizontal axis positions are not modified (default).
                If the horizontal axis scaling is logarithmic, the rectification factor
                is applied in log-space.

        Examples:
            # show three curves with automatic horizontal rectification
            __init__(visualization_type=("points", "points", "box-whisker"), rectify=True)
        """

        super().__init__(**kwargs)

        # parameter validation

        enum_f = lambda arg: params.enumeration(arg, {"points", "box-whisker", "shaded-line"})
        self._visualization_type = params.any_(
            visualization_type, enum_f, lambda arg: params.tuple_(arg, enum_f)
        )
        # arity can only be tested in evaluate()

        self._rectify = params.any_(rectify, lambda arg: params.real(arg, from_=0), params.boolean)

    # multiple points / box plots / ... are drawn next to each other if rectified
    # these are the relative offsets
    RECTIFY_DELTAS = {
        1: [0],
        2: [-1, 1],
        3: [-2, 0, 2],
        4: [-3, -1, 1, 3],
        5: [-4, -2, 0, 2, 4],
        6: [-5, -3, -1, 1, 3, 5],
        7: [-6, -4, -2, 0, 2, 4, 6],
        8: [-7, -5, -3, -1, 1, 3, 5, 7],
        9: [-8, -6, -4, -2, 0, 2, 4, 6, 8],
    }

    def evaluate(self, results, **kwargs):
        """Compute plot data for multiple generalized (set-valued) functions.

        Multiple curves C_1, ..., C_k can be drawn.
        Each curve C_i is specified by a non-empty sequence of 2-tuples,
        where the first value is location on horizontal axis, and the
        other value is a sequence of locations on the vertical axis.

        Each curve can be drawn in a different way (points, box-whisker).

        Parameters:
            results: sequence of generalized functions data (curve data).
                     Each datum is a sequence of tuples (x,fx), where
                     x is a real number and fx is a sequence of real numbers.

        Examples:
            # two curves sharing one horizontal location
            evaluate([
                [(1,(1,0.9,1.1)), (3,(2,))],  # curve 1
                [(1,(0.7,)), (2,(3.1,2.8)), (4,(5.5,7.3,6))], # curve 2
            ])
        """

        super().evaluate(results=results, **kwargs)

        # parameter validation

        tuple_testf = lambda arg: params.tuple_(arg, params.real, params.real_vector, arity=2)
        curve_testf = lambda arg: params.tuple_(arg, tuple_testf)
        results = params.tuple_(results, curve_testf)

        # _rectify evaluates to True if True or if > 0
        if len(results) > len(self.RECTIFY_DELTAS) and self._rectify:
            raise InvalidParameterError(
                f"at most {len(self.RECTIFY_DELTAS)} curves", f"{len(self.RECTIFY_DELTAS)} curves"
            )

        # finalize parameter validation for visualization_type
        if not is_sequence(self._visualization_type):
            self._visualization_type = (self._visualization_type,) * len(results)
        self._visualization_type = params.tuple_(
            self._visualization_type,
            lambda arg: params.enumeration(arg, {"points", "box-whisker", "shaded-line"}),
            arity=len(results),
            default="points",
        )

        # prepare plot

        # determine all distinct horizontal positons in the results data
        all_positions = np.unique([entry[0] for curve in results for entry in curve])

        # there is nothing to do without data to plot
        if len(all_positions) == 0:
            self._plotdata = []
            return

        # do not rectify if there is only a single horizontal position
        if len(all_positions) == 1 or self._rectify is False:
            self._rectify = 0.0

        # automatic determination of horizontal rectification factor
        #
        # the correct way to draw box-plots on a logarithmic horizontal axis is to have
        # different left-width and right-width of the boxes. However, matplotlib does not
        # support this. Because box widths are small compared to horizontal plot range,
        # it suffices to use the sum of left- and right-half widths.
        between_groups_spacing = 0.4
        in_group_spacing = 0.9  # box-whisker plots
        if self.axes_scales[0] == "linear":
            logf = lambda arg: arg
            powf = lambda arg: arg
        elif self.axes_scales[0] == "log":
            base = 10
            logf = lambda arg: np.log(arg) / np.log(base)
            powf = lambda arg: np.power(base, arg)

        if self._rectify is True:
            # diff(...) requires at least two horizontal locations; this is ensured above
            self._rectify = (
                between_groups_spacing * min(np.diff(logf(all_positions))) / len(results)
            )

        # determine positions
        self._plotdata = [None] * len(results)
        deltas = self.RECTIFY_DELTAS[len(results)] if self._rectify else np.zeros(len(results))
        for (i, curve) in enumerate(results):
            # point markers, every single point is drawn
            if self._visualization_type[i] == "points":
                positions = powf(
                    np.hstack(
                        [
                            logf(entry[0] * np.ones(len(entry[1]))) + deltas[i] * self._rectify / 2
                            for entry in curve
                        ]
                    )
                )
                values = np.hstack([entry[1] for entry in curve])
                self._plotdata[i] = np.transpose([positions, values])
            # box-whisker plots
            elif self._visualization_type[i] == "box-whisker":
                positions = np.asfarray(
                    [logf(entry[0]) + deltas[i] * self._rectify / 2 for entry in curve]
                )
                values = [entry[1] for entry in curve]
                # can't use rectify for width if 0; 1 is a wild guess
                # todo: if plot ranges have been set, a better default value could
                #       be 10% of horizontal plot range
                w = 1 if not self._rectify else self._rectify
                widths = powf((positions + w / 2) * in_group_spacing) - powf(
                    (positions - w / 2) * in_group_spacing
                )
                positions = powf(positions)
                self._plotdata[i] = (positions, values, widths)
            elif self._visualization_type[i] == "shaded-line":
                positions = np.asfarray([entry[0] for entry in curve])
                values = [entry[1] for entry in curve]
                self._plotdata[i] = (positions, values)
            else:
                raise BenchmarkError("internal error, unknown visualization type")

    def _render(self, target, **kwargs):
        """Render generalized function plot.

        Parameters:
            target: rendering target which evaluation outcome is rendered to; see Evaluation._render method
        """

        # draw curves
        for (i, pd) in enumerate(self._plotdata):
            # point markers, every single point is drawn
            if self._visualization_type[i] == "points":
                self.points(pd, color=i)
            elif self._visualization_type[i] == "box-whisker":
                self.box_whisker(pd[0], pd[1], color=i, widths=pd[2])
            else:
                raise BenchmarkError("internal error, unknown visualization type")


# todo: weighting; currently not done


class LearningCurvePlot(GeneralizedFunctionPlot):
    r"""Plot learning performance as a function of training set size.

    "Learning curves" are plots of empirical prediction error as a function of training set size.
    Asymptotically, the prediction error decays as a negative power, $\epsilon = a' n^{-b}$. [1]
    On a log-log plot, $\epsilon$ is therefore linear, $\log \epsilon = a - b \log(n)$,
    and the offset $a$ and slope $b$ can be used to characterize predictive performance of models. [2]

    [1] Shun-ichi Amari, Naotake Fujita, Shigeru Shinomoto: Four Types of Learning Curves,
        Neural Computation 4(4): 605-618, 1992. DOI 10.1162/neco.1992.4.4.605
    [2] Bing Huang, O. Anatole von Lilienfeld: Communication: Understanding molecular
        representations in machine learning: the role of uniqueness and target similarity,
        Journal of Chemical Physics 145(16): 161102, 2016. DOI 10.1063/1.4964627
    """

    def __init__(
        self,
        fits: bool = True,
        fit_lambda: float = 1e-7,
        fit_weights: Optional[str] = None,
        base=10,
        **kwargs,
    ):
        """Initialize learning curve plot.

        Parameters:
            fits: if True, show estimated asymptotic fits
            fit_lambda: regularization strength for asymptotic fits; defaults to 1e-7
            fit_weights: if and how to weight fits; one of
                None: no weighting, "variance": weigh by variance for each training set size
            base: base for logarithmic plotting
            All parameters from base classes, in particular GeneralizedFunctionPlot and Plot.
        """

        # set learning curve-specific arguments if not explicitly set
        kwargs["axes_scales"] = kwargs.get("axes_scales", ("log", "log"))
        kwargs["axes_labels"] = kwargs.get(
            "axes_labels", ("training set size", "evaluation metric", None, None)
        )

        super().__init__(**kwargs)

        # parameters
        self._fits = params.boolean(fits)
        self._fit_lambda = params.real(fit_lambda, from_=0)
        self._fit_weights = params.any_(
            fit_weights, lambda arg: params.enumeration(arg, {"variance"}), params.none
        )
        self._base = params.real(base, from_=2)

        self._logf = lambda x: np.log(x) / np.log(self._base)
        self._powf = lambda x: np.power(self._base, x)

    def asymptotic_fit(self, fdata):
        r"""Compute asymptotic fit in log-space for a single curve.

        The asymptotic fit is computed using a simple form of linear ridge regression,
        estimating two parameters, offset b and slope a: $f(x) = b + a x$.
        In short, we augment x with a second dimension of constant value 1 to remove the bias,
        $f( (x,1) ) = <(a,b),(x,1)>$. Then, solving
        $\argmin_{a,b} \sum_{i=1}^n (y_i - f((x_i,1)))^2 + \lambda ||(a,b)||^2$
        by rewriting in matrix notation, setting the derivative to zero and solving for (a,b) yields
        $(a,b) = (X^T X + \lambda I)^{-1} X^T y$, where the $n \times 2$-dimensional matrix X
        contains the data the fit is based on. The variance, or mean squared error (MSE),
        indicates how well empirical errors follow the asymptotic fit.

        Parameters:
            fdata: data for a single curve
        """

        # compute mean in log-space as the fit is linear in log space
        # todo: verify that this is the correct procedure
        sizes = self._logf(np.asfarray(tuple(entry[0] for entry in fdata)))
        means = np.asfarray(tuple(np.mean(self._logf(entry[1])) for entry in fdata))
        n = len(sizes)  # number of training set sizes

        if self._fit_weights is None:
            weights = np.ones(n)
        elif self._fit_weights == "variance":
            raise NotImplementedError  # todo: do weighting properly
            if min(len(entry[1]) for entry in fdata) < 2:
                raise InvalidParameterError(
                    "multiple values per horizontal location",
                    "fewer than two samples for at least one location",
                    explanation="weighting by variance not defined for fewer than two samples",
                )
            # todo: check for zero variance cases and replace by one
            weights = tuple(1 / np.var(entry[1]) for entry in fdata)
        else:
            raise BenchmarkError("internal error, invalid weighting scheme")
        weights /= np.sum(weights)

        X = np.ones((n, 2))  # second column is 1
        X[:, 0], y = sizes, means  # fit is in log-space
        assert y.shape == (n,), f"loss vector has wrong dimensions {y.shape}"

        # standard linear ridge regression in log-space
        slope, offset = np.linalg.pinv(X.T @ X + self._fit_lambda * np.identity(2)) @ X.T @ y

        # variance of the fit
        residuals = y - (offset + slope * self._logf(n))
        variance = np.mean(np.asfarray(residuals ** 2))

        return offset, slope, residuals, variance

    def evaluate(self, results, **kwargs):
        """Evaluate learning curve plot.

        Parameters:
            results: sequence of curve data, where each curve datum is a sequence of tuples (n,fx)
                of training set size n (positive integer) and performance values fx (sequence of real numbers).
        """

        # parameter validation

        tuple_testf = lambda arg: params.tuple_(
            arg, lambda arg: params.real(arg, above=0), params.real_vector, arity=2
        )
        curve_testf = lambda arg: params.tuple_(arg, tuple_testf)
        results = params.tuple_(results, curve_testf)

        super().evaluate(results=results, **kwargs)

        ypowf = self._powf if self.axes_scales[1] == "log" else lambda arg: arg

        # asymptotic estimates
        if self._fits:
            asymptotic_fits = tuple(self.asymptotic_fit(fdata) for fdata in results)

            all_sizes = np.unique([entry[0] for fdata in results for entry in fdata])
            sizes = np.linspace(start=np.min(all_sizes), stop=np.max(all_sizes), num=25)
            self._fit_data = np.empty(shape=(len(results), 2, len(sizes)))
            for i, (offset, slope, _, _) in enumerate(asymptotic_fits):
                yvalues = [ypowf(offset + slope * self._logf(n)) for n in sizes]
                self._fit_data[i, 0, :] = sizes
                self._fit_data[i, 1, :] = yvalues

            self.add_auxiliary(
                "asymptotic_fits",
                tuple(
                    {
                        "offset": offset,
                        "slope": slope,
                        "residuals": residuals,
                        "variance": variance,
                    }
                    for (offset, slope, residuals, variance) in asymptotic_fits
                ),
            )

    def _render(self, target, **kwargs):
        """Render learning curve plot.

        Parameters:
            target: rendering target which evaluation outcome is rendered to; see Evaluation._render method
        """

        super()._render(target=target, **kwargs)

        for (i, linedata) in enumerate(self._fit_data):
            self.line(np.transpose(linedata), color=i)


# todo: add export of plots to ASCII art
#       this is an important feature :-)


class OptimizationTrajectoryPlot(GeneralizedFunctionPlot):
    """Plot a series of optimization trajectories, each one tracking the best score found at
    that point in the optimization run.
    Each trial for a given optimizer is currently plotted as a dot.

    Parameters:
        optimizer_names: list of names with which to label each optimizer trajectory
        log_scale: whether or not to use a log scale on the _horizontal_ axis
        quantile_width: fraction of the range to shade. Shading is centered around the median,
            going from median - quantile_width / 2 to median + quantile_width / 2
    """

    def __init__(
        self,
        optimizer_names: Optional[List[str]] = None,
        log_scale: bool = False,
        quantile_width: float = 0.5,
        show_extrama: bool = True,
        **kwargs,
    ):
        self._optimizer_names = params.optional_(
            optimizer_names, lambda arg: params.sequence(arg, type_=str)
        )
        self._show_extrema = params.boolean(show_extrama)
        log_scale = params.boolean(log_scale)
        scale = "log" if log_scale else "linear"

        self._quantile_width = params.real(quantile_width, from_=0, to=1)

        kwargs["axes_scales"] = kwargs.get("axes_scales", (scale, "linear"))
        kwargs["axes_labels"] = kwargs.get(
            "axes_labels", ("function evaluations", "best score", None, None)
        )
        kwargs["rectify"] = False
        kwargs["visualization_type"] = "shaded-line"

        super().__init__(**kwargs)

    def evaluate(self, results, **kwargs):
        """Evaluate optimization trajectory plot.

        Parameters:
            results: sequence of curve data, where each curve datum is a sequence of
                tuples (index, scores) of function evaluation number (positive integer)
                and best scores found after that many evaluations (sequence of real numbers).
        """
        tuple_testf = lambda arg: params.tuple_(
            arg, lambda arg: params.real(arg, above=0), params.real_vector, arity=2
        )
        curve_testf = lambda arg: params.tuple_(arg, tuple_testf)
        results = params.tuple_(results, curve_testf)

        super().evaluate(results=results, **kwargs)

    def _render(self, target, **kwargs):
        """Render optimization trajectory plot.

        Parameters:
            target: rendering target which evaluation outcome is rendered to; see Evaluation._render method
        """
        if self._optimizer_names is None:
            self._line_labels = [None] * len(self._plotdata)
        else:
            self._line_labels = params.sequence(self._optimizer_names, length=len(self._plotdata))

        for i, (pd, label) in enumerate(zip(self._plotdata, self._line_labels)):
            self.shaded_line(
                pd[0],
                pd[1],
                color_idx=i,
                label=label,
                quantile_width=self._quantile_width,
                show_extrema=self._show_extrema,
            )

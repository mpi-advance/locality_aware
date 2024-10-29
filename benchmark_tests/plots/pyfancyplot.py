import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid", palette="muted")
font_style = 'serif'
linewidth = 3
xscale = 'linear'
yscale = 'linear'
palette_name = "tab20" 
palette_n_colors = 7
palette_desat = None
color_ctr = 0
lgd = None
fig = None

################################
###  Set Default Options
###  Serif Font, Size 24
###  Initial standard color palette
################################
def default_options():
    fontsize = 24
    fig_width_pt = 700.0
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size = [fig_width, fig_height]
    params = {'backend': 'ps',
        'font.family': 'serif',
        'font.serif':  'cm',
        'font.sans-serif': 'arial',
        'axes.labelsize': fontsize,
        'font.size': fontsize,
        'axes.titlesize': fontsize,
        'legend.fontsize': fontsize-2,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'text.usetex': True,
        'figure.figsize': fig_size,
        'lines.linewidth': 4,
        'hatch.linewidth': 3.0}
    plt.rcParams.update(params)

    global font_style
    global linewidth
    global xscale
    global yscale
    global palette_name
    global palette_n_colors
    global palette_desat
    global color_ctr
    global ldg
    global fig

    font_style = 'serif'
    linewidth = 3
    xscale = 'linear'
    yscale = 'linear'
    palette_name = "tab20"
    palette_n_colors = 7
    palette_desat = None
    color_ctr = 0
    lgd = None

    fig = plt.figure(figsize=(fig_width, fig_height));
    plt.gcf()

################################
###  The reason you're 
###  using this script
################################
def add_luke_options():
    from matplotlib import rc
    font = {'family' : 'serif',
            'size' : 22}
    rc("font", **font)
    rc("lines", linewidth=3)

    ax = plt.gca()
    ax.xaxis.grid(False)
    #plt.grid(True)
    sns.despine(left=True, right=True)

################################
###  Clear previously plotted data 
###  Reset all global variables
###  Initialize default options
################################
def clear():
    plt.clf()
    plt.close('all')
    default_options()

# Automatically set default options
default_options()
add_luke_options() ## add these by default, too


################################
###  Set color palette
###  Palette options:
###    - Seaborn color palette name
###    - list of colors
###    - list of RGB values
################################
def set_palette(palette = "tab20", n_colors = None, desat = None):
    global palette_name
    global palette_n_colors 
    global palette_desat
    global color_ctr
    
    palette_name = palette
    palette_n_colors = n_colors
    palette_desat = desat
    color_ctr = 0

################################
###  Returns Seaborn color palette
###  If num colors is passed, will 
###    override palette_n_colors
###  For other options, first
###    call set_palette
################################
def get_palette(num_colors = None):
    global palette_name
    global palette_n_colors 
    global palette_desat
    if not num_colors is None:
        palette_n_colors = num_colors
    return sns.color_palette(palette_name, palette_n_colors, palette_desat)

################################
###  Returns next color in 
###   Seaborn palette
################################
def next_color():
    global palette_name
    global palette_n_colors 
    global palette_desat
    global color_ctr
   
    color_palette = sns.color_palette(palette_name, palette_n_colors, palette_desat)
    color = color_palette[color_ctr];
    color_ctr = (color_ctr + 1) % len(color_palette)
    return color

################################
###  Set scale of x and y dimensions
###  Either 'linear' or 'log'
################################
def set_scale(xscale, yscale):
    ax = plt.gca()
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_autoscaley_on(False)    

def get_ax():
    return plt.gca()

################################
###  Set figure size (by inches)
################################
def set_figure_size(dim_inches):
    global fig
    fig.set_size_inches(dim_inches)

################################
###  Set figure size (by dpi)
################################
def set_figure_dpi(dpi):
    fig.set_dpi(dpi)

################################
###  Add standard matplotlib legend
################################
def add_legend(ncol = 1,
        loc = 'best',
        frameon = False,
        fontsize = 20,
        **kargs):

    plt.legend(loc = loc, ncol = ncol, frameon = frameon, fontsize =
            fontsize, **kargs)

################################
###  Add multi column legend
###  Anchored about plot by default
################################
def add_anchored_legend(ncol = 2, 
        loc = "upper center", 
        anchor = (0., 1.10, 1.,.102),
        frameon = False,
        fontsize = 22,
        **kargs):

    global lgd
    lgd = plt.legend(loc = loc, ncol = (int)(ncol), bbox_to_anchor = anchor, 
            frameon = frameon, fontsize = fontsize, **kargs)

################################
###  Adds legend for barplots
###  By selecting rectangles
###  Adds multi column legend
###  Anchored about plot by default
################################
def barplot_legend(labels, positions, ax, n_cols = 0, **kargs):
    import matplotlib.patches as patches
    objs = ax.findobj(match=patches.Rectangle)
    legend_lines = list()
    
    for i in range(len(labels)):
        idx = positions[i]
        legend_lines.append(objs[idx])
    if n_cols <= 0:
        n_cols = ((len(labels) - 1) / 2) + 1
        if (n_cols < 2):
            n_cols = 2
    add_anchored_legend(handles=legend_lines, labels=labels, ncol=n_cols,
            **kargs)    

################################
###  Add a title
################################
def add_title(title):
    plt.title(title)

################################
###  Add labels for x and y dims
################################
def add_labels(xlabel, 
        ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)



################################
###  Set limits for x dimension
################################
def set_xlim(xmin = None, 
        xmax = None):
    ax = plt.gca()
    if xmin is None:
        xmin, _ = ax.get_xlim()
    if xmax is None:
        _, x1 = ax.get_xlim()
        xmax = x1 + (x1  / 20.0)
    ax.set_xlim((xmin, xmax))

################################
###  Set limits for y dimension
################################
def set_ylim(ymin = None, 
        ymax = None):
    ax = plt.gca()
    ax.set_autoscaley_on(False)
    if ymin is None:
        y0, _ = ax.get_ylim()
        ymin = y0 - (y0 / 50.0)
    if ymax is None:
        _, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    ax.axis('tight')



################################
###  Sets x-tick labels
################################
def set_xticklabels(xticklabels,
        rotation = 'vertical', 
        fontsize = 16,
        **kargs):
    ax = plt.gca()
    ax.set_xticklabels(xticklabels, rotation = rotation,
            fontsize = fontsize, **kargs)

################################
###  Sets x-ticks and corresponding labels
################################
def set_xticks(xdata, xticklabels, rotation = 'horizontal', **kargs):
    ax = plt.gca()
    ax.set_xticks(xdata)
    ax.set_xticklabels(xticklabels, rotation = rotation, **kargs)


################################
###  Sets y-tick labels
################################
def set_yticklabels(yticklabels,
        rotation = 'horizontal', 
        fontsize = 16, 
        **kargs):
    ax = plt.gca()
    ax.set_yticklabels(yticklabels, rotation = rotation,
            fontsize = fontsize, **kargs)

################################
###  Set y-ticks and corresponding labels
################################
def set_yticks(ydata, 
        yticklabels, 
        rotation = 'horizontal', 
        **kargs):
    ax = plt.gca()
    ax.set_yticks(ydata)
    ax.set_yticklabels(yticklabels, rotation = rotation, **kargs)



################################
###  Standard line plot
################################
def line_plot(y_data, 
        x_data = None, 
        tickmark = '-', 
        alpha = 1.0, 
        linewidth=3, 
        ax = plt,
        color = None,
        **kargs):
    if x_data is None:
        x_data = np.arange(0, len(y_data))
    if color is None:
        color = next_color()
    return ax.plot(x_data, y_data, tickmark,
            color = color, clip_on = False,
            alpha = alpha, linewidth = linewidth, **kargs)


def violin_plot(x_data, y_data, labels = None, add_legend = True, ax = None,**kargs): 
    if ax is None:
        ax = plt.gca()

    if labels is None:
        vplot = sns.barplot(x=x_data, y=y_data)
    else:
        pd_dict = dict()
        pd_dict['x'] = x_data
        for i in range(len(labels)):
            pd_dict[labels[i]] = y_data[i]
        df = pd.DataFrame(pd_dict)
        df = df.melt(id_vars=['x'], var_name='measure', value_vars=labels,
                value_name='time')
        vplot = sns.violinplot(data=df, x='x', y='time', hue='measure', ax =
                ax, palette = get_palette(), edgecolor='black', **kargs)

    return vplot

################################
###  Standard scatter plot
################################
def scatter_plot(x_data, 
        y_data, 
        marker = 'o', 
        color = None,
        **kargs):
    if color is None:
        color = next_color()
    return plt.scatter(x_data, y_data, c = color, edgecolors='none', 
            clip_on = False, marker = marker, **kargs)

################################
###  Spy of Matrix 
################################
def spy(A, color = 'black', markersize = None):
    plt.spy(A, rasterized=True, markersize=markersize)

################################
###  Creates a standard barplot
################################
def barplot(x_data,
        y_data, 
        labels = None, 
        ax = None, 
        add_legend = True,
        color = None,
        **kargs):
    if ax is None:
        ax = plt.gca()

    bplot = ""
    if labels is None:
        if color is None:
            color = next_color()
        bplot = sns.barplot(x=x_data, y=y_data, color=color, ax = ax,
                edgecolor='black', **kargs)
    else:
        pd_dict = dict()
        pd_dict['x'] = x_data
        for i in range(len(labels)):
            pd_dict[labels[i]] = y_data[i]
        df = pd.DataFrame(pd_dict)
        df = df.melt(id_vars=['x'], var_name='measure', value_vars=labels,
                value_name='time')
        bplot = sns.barplot(data=df, x='x', y='time', hue='measure', ax =
                ax, palette = get_palette(), edgecolor='black', **kargs)
        if add_legend:
            positions = [i * len(x_data) for i in range(len(labels))]
            barplot_legend(labels, positions, ax)


    return bplot
    
################################
###  Creates a stacked barplot
################################
def stacked_barplot(x_data, # simple list
        y_data, #list of lists (each of len(x_data))
        labels, #list of labels corresponding to y_data
        ax = None, 
        **kargs):
    if ax is None:
        ax = plt.gca()
    bplots = list()

    new_y_data = list()
    for i in range(len(y_data)):
        new_y_data.append(list())
        for j in range(len(y_data[i])):
            new_y_data[i].append(y_data[i][j])

    for i in range(len(labels)):
        for j in range(len(labels)-1, i, -1):
            for k in range(0, len(new_y_data[i])):
                new_y_data[i][k] += new_y_data[j][k]

    colors = get_palette(len(labels))
    plots = list()
    for i in range(len(labels)):
        pd_dict = dict()
        pd_dict['x'] = x_data
        pd_dict[labels[i]] = new_y_data[i]
        df = pd.DataFrame(pd_dict)
        df = df.melt(id_vars=['x'], var_name='measure',
                value_vars=[labels[i]],
                value_name='time')
        bplots.append(sns.barplot(ax = ax, data=df, x='x', y='time',
            color=colors[i], edgecolor='black', **kargs))
    positions = [i * len(x_data) for i in range(len(labels))]
    barplot_legend(labels, positions, ax)
    return bplots

################################
###  Create a partially stacked barplot
###  Stacking some bars, but not all
###  All bars share the same x_data
################################
def partially_stacked_barplot(x_data, # simple list
        y_data, # list of data and lists e.g.
                # [y0, y1, [y2, y3], y4] would leave
                # y0, y1, and y4 as simple bars
                # but would stack y3 on top of y2
        labels, # labels in same format as y_data
                # e.g. [l0, l1, [l2, l3], l4] corresponds
                # to example y_data 
        ax = None,
        **kargs):
    if ax is None:
        ax = plt.gca()
    bplots = list()
    max_stack_size = 1
    num_bars = len(y_data)
    num_colors = 0
    bar_num_stacked = list()
    stacked_bars = list()
    positions = list()
    for i in range(num_bars):
        if type(labels[i]) == type("string"):
            bar_num_stacked.append(1)
            positions.append(i)
        else:
            stack_size = len(y_data[i])
            bar_num_stacked.append(stack_size)
            stacked_bars.append(i)
            for j in range(stack_size):
                positions.append(i)
            if stack_size > max_stack_size:
                max_stack_size = stack_size
        num_colors += bar_num_stacked[-1] 
    indices = np.zeros((num_colors, ), dtype = 'int')

    colors = get_palette(num_colors)
    iter_colors = list() 
    iter_ctrs = list()    
    iter_labels = list()
    iter_y_data = list()

    ctr = 0
    for i in range(num_bars):
        stack_size = bar_num_stacked[i]
        if stack_size == 1:
            iter_labels.append(labels[i])
            iter_y_data.append(y_data[i])
        else:
            iter_labels.append(labels[i][-1])
            iter_y_data.append(y_data[i][0])
            for j in range(1, stack_size):
                for k in range(0, len(iter_y_data[i])):
                    iter_y_data[i][k] += y_data[i][j][k]
        iter_colors.append(colors[ctr + stack_size - 1])
        iter_ctrs.append(ctr + stack_size - 1)
        ctr += stack_size

    # Bar plot for each stack
    for i in range(max_stack_size):
        set_palette(iter_colors, len(iter_colors))
        barplot(x_data, iter_y_data, iter_labels)
        for idx in stacked_bars:
            stack_size = bar_num_stacked[idx]
            if stack_size <= i+1: 
                continue
            iter_ctrs[idx] -= 1
            indices[iter_ctrs[idx]] = i+1
            iter_labels[idx] = labels[idx][stack_size-i-2]
            for k in range(len(iter_y_data[idx])):
                iter_y_data[idx][k] -= y_data[idx][stack_size-i-1][k]
            iter_colors[idx] = colors[iter_ctrs[idx]]

    label_list = list()
    for i in range(num_bars):
        stack_size = bar_num_stacked[i]
        if stack_size == 1:
            label_list.append(labels[i])
        else:
            for j in range(stack_size):
                label_list.append(labels[i][j])
    for i in range(num_colors):
        positions[i] = (indices[i]*num_bars*len(x_data)) + (positions[i]*len(x_data))
    barplot_legend(label_list, positions, ax)


################################
### Save the plot to a file
### Clears all data after by default
################################
def save_plot(filename, 
        clear_plot = True, 
        **kargs):
    global lgd

    if lgd is None:
        plt.savefig(filename, bbox_inches = "tight", 
                transparent=True, **kargs)
    else:
        plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches = "tight",
                transparent=True, **kargs)

    if clear_plot:
        clear()

################################
### Display your plot 
################################
def display_plot():
    plt.show()


    



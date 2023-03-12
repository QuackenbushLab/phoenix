import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as font_manager
import matplotlib as mpl

def set_font_settings():
    mpl.rcParams.update(mpl.rcParamsDefault)
    font_dirs = ['/usr/share/fonts/' ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    font_list = font_manager.createFontList(font_files)
    font_manager.fontManager.ttflist.extend(font_list)

    font = {'family' : "Times New Roman",
            'style' : "normal",
            'weight': "normal",
            'size'   : 12}
    plt.rc('font', **font)
    plt.rcParams['font.weight'] = 'normal'

def save_figure(fig, path, width=5.9, height=None, square=False):
    """Save the supplied picture to path.

        The width is set in inches, with height according to the golden ratio if not specifically set.
        Saves all figures with Times New Roman 12 pt font to match thesis font. 

        Square makes the height the same as the width, if height is not explicitly set.  
    """
    
    if not height and not square:
        height = width / 1.618
    if not height and square:
        height = width
    fig.set_size_inches((width, height))
    
    #fig.subplots_adjust()
    #fig.tight_layout()
    fig.tight_layout()
    fig.savefig(path)
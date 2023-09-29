import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler

class CustomCycler:

    def __init__(self, color=None):
        if color is None:
            self.color = ['k', 'r', 'b', 'g', 'm', 'c', 
             'orange', 'gray', 'purple', 'pink', 'olive', 'cyan', 
             'lime', 'teal', 'lavender', 'maroon', 'navy', 'gold', 
             'azure', 'ivory', 'indigo', 'silver', 'beige', 
             'darkgreen', 'lightgrey', 'salmon', 'darkblue', 'violet', 
             'turquoise', 'tan', 'orchid', 'darkorange', 
             'darkred', 'darkmagenta', 'darkcyan', 
             'darkkhaki', 'darkviolet', 'darkturquoise', 
             'darkolivegreen', 'darkseagreen', 'darkgrey', 
             'darkslateblue', 'darkslategrey', 'darkslategray', 
             'darkgoldenrod', 'darkorchid', 'darksalmon'
             ]
        else:
            self.color = color

        self.color = np.vstack([self.color, self.color]).reshape(-1, order='F')
        

    # @classmethod
    def get_ColorCycler(self):
        custom_cycler = (cycler(color=self.color))
        # print(self.color)
        # exit()

        return custom_cycler        
    

def set_matplotlib_settings():
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
    plt.rcParams.update(params)


def get_CustomCycler(color=None):
    if color is None:
        color = ['k', 'r', 'b', 'g', 'm', 'c', 
            'orange', 'gray', 'purple', 'pink', 'olive', 'cyan', 
            'lime', 'teal', 'maroon', 'navy', 'indigo', 
            'silver', 'darkgreen', 'salmon', 'darkblue', 'violet', 
            'turquoise', 'tan', 'orchid', 'darkorange', 
            'darkred', 'darkmagenta', 'darkcyan', 
            'darkkhaki', 'darkviolet', 'darkturquoise', 
            'darkolivegreen', 'darkseagreen', 'darkgrey', 
            'darkslateblue', 'darkslategrey', 'darkslategray', 
            'darkgoldenrod', 'darkorchid', 'darksalmon'
            ]

    color = np.vstack([color, color]).reshape(-1, order='F')
        

    custom_cycler = (cycler(color=color))

    return custom_cycler     
    
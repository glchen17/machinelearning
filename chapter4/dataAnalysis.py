# !/usr/bin/env python
# -*- coding: utf-8 -*-











def lineplot(x_data, y_data, x_label="", y_label="", title=""):

   # Plot the best fit line, set the linewidth (lw), color and
   # transparency (alpha) of the line
   plt.plot(x_data, y_data, lw=2, color='#539caf', alpha=1)
   # Label the axes and provide a title
   plt.title(title)
   plt.xlabel(x_label)
   plt.ylabel(y_label)
   plt.show()
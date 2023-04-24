import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math

def floatRgb(mag, cmin, cmax):
    """ Return a tuple of floats between 0 and 1 for R, G, and B. """
    # Normalize to 0-1
    try: x = float(mag-cmin)/(cmax-cmin)
    except ZeroDivisionError: x = 0.5 # cmax == cmin
    blue  = min((max((4*(0.75-x), 0.)), 1.))
    red   = min((max((4*(x-0.25), 0.)), 1.))
    green = min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
    return red, green, blue

def make_plot():
    files = os.listdir("logs")

    y_ends = []
    x_ends = []
    times = []
    for file in files:
        y, x = file.split('_')[-1].split('.')[0].split('-')

        df = pd.read_csv(f"logs/{file}")[2:4]['Chip clock is at 1.2 GHz']
        start = int(df[0].strip(' '))
        end = int(df[1].strip(' '))
        y_ends.append(int(y))
        x_ends.append(int(x))
        times.append(end - start)

    normalized_times = []
    max_time = max(times)
    min_time = min(times)

    for time in times:
        rgb = floatRgb(time, min_time, max_time)
        normalized_times.append(rgb)

        if time == max_time:
            max_normalized = rgb
        elif time == min_time:
            min_normalized = rgb

    plt.gca().invert_yaxis()
    for (x_end, y_end, time) in zip(x_ends, y_ends, normalized_times):
        if time == max_normalized:
            marker = 'X'
        elif time == min_normalized:
            marker = '*'
        else:
            marker = 'o'

        plt.scatter(x_end, y_end, c=time, marker=marker, edgecolors='black')

    star_marker = mlines.Line2D([], [], color=min_normalized, marker='*', linestyle='None',
                            markersize=10, label='Min cycles: ' + str(min_time))
    x_marker = mlines.Line2D([], [], color=max_normalized, marker='X', linestyle='None',
                            markersize=10, label='Max cycles: ' + str(max_time))

    plt.legend(handles=[star_marker, x_marker])
    plt.xlabel('x_end', fontsize=18)
    plt.ylabel('y_end', fontsize=16)
    plt.title('plot of datacopy dispatch (from core (1, 11))\n duration in which y_start=1 and x_start=1')

    plt.savefig("temp.png")

if __name__ == "__main__":
    make_plot()

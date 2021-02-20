from math import pow, log, exp
from tetration import tetr_execute
from scipy import interpolate

epsilon1 = 3.0
epsilonN = 1.10256687189e-11
centerRe = -1.48571730546
centerIm = 0.0823985635523
N = 300

# Zooming should be done by multiplying with a constant. The constant is based on the number of frames
# To not make the ending as "sudden", the last 20% of the way is extended to two parts each of length 20%.
# Of this 80% of the zoom is done in the first part, 20% in the second part.

zoom_factor = (log(epsilonN) - log(epsilon1)) / float(N)
start = log(epsilon1)
end = log(epsilonN)

# Double the final interval
x_points = [ float(N) * .8, N, float(N) * 1.2 ]
y_points = [ start + float(N) * .8 * zoom_factor, start + float(N) * .96 * zoom_factor, end ]

print x_points
print y_points

tck = interpolate.splrep( x_points, y_points, k=2)

def spline(t):
    return max(interpolate.splev(t, tck), end)

frameXLocation = 'movie/output{0:05d}.png'

def create_frame(iteration):
    if (iteration < float(N) * .8):
        log_eps = start + iteration * zoom_factor
    else:
        log_eps = spline(iteration)
    epsilon = exp(log_eps)
    print "epsilon = ", epsilon
    tetr_execute(str(centerRe), str(centerIm), str(epsilon), '20000', '2000', frameXLocation.format(iteration))

for i in xrange(243, int(float(N) * 1.2)):
    create_frame(i)
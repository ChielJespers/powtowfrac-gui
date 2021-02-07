from math import pow, log, exp
from tetration import tetr_execute

epsilon1 = 3.0
epsilonN = 1.10256687189e-11
centerRe = -1.48571730546
centerIm = 0.0823985635523
N = 300

zoom_factor = exp((log(epsilonN) - log(epsilon1)) / float(N))

frameXLocation = 'movie/output{0:05d}.png'

def create_frame(iteration):
    epsilon = epsilon1 * pow(zoom_factor, iteration)
    tetr_execute(str(centerRe), str(centerIm), str(epsilon), '20000', '2000', frameXLocation.format(iteration))



for i in xrange(277, N):
    create_frame(i)


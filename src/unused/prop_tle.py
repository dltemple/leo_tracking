from sgp4.api import Satrec

s = '1  7004U 73107B   20056.41432514  .00021997  00000-0  19211-3 0  9991'
t = '2  7004  73.9247  50.7662 0106634  25.4206 335.2201 15.61627510389549'

satellite = Satrec.twoline2rv(s, t)

jd, fr = 2458902, .54540834

e, r, v = satellite.sgp4(jd, fr)

print(r)
print(v)

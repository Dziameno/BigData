from simpful import *

FS = FuzzySystem()

TLV = AutoTriangle(3, terms=['poor', 'average', 'good'], universe_of_discourse=[0,10])
FS.add_linguistic_variable("service", TLV)
FS.add_linguistic_variable("quality", TLV)

O1 = TriangleFuzzySet(0,0,10,   term="low")
O2 = TriangleFuzzySet(0,10,20,  term="medium")
O3 = TriangleFuzzySet(10,20,30, term="high")
FS.add_linguistic_variable("tip", LinguisticVariable([O1, O2, O3], universe_of_discourse=[0,25]))

FS.add_rules([
	"IF (quality IS poor) OR (service IS poor) THEN (tip IS low)",
	"IF (service IS average) THEN (tip IS medium)",
	"IF (quality IS good) OR (service IS good) THEN (tip IS high)"
	])
FS.set_variable("quality", 6.5, True)
FS.set_variable("service", 9.8, True)
print(FS.Mamdani_inference())

FS.set_variable("quality", 2, True)
FS.set_variable("service", 3, True)
print(FS.Mamdani_inference())

FS.set_variable("quality", 25, True)
FS.set_variable("service", 2, True)
tip = FS.inference()
print(FS.Mamdani_inference())


FS.plot_variable("quality")
FS.plot_variable("service")
FS.plot_variable("tip")
import numpy as np


# Things classes:
# car: 1
# bicycle: 2
# motorcycle: 3
# truck: 4
# other-vehicle: 5
# person: 6
# bicyclist: 7
# motorcyclist: 8
## pole: 18
## traffic-sign: 19
thing_ids = [1, 2, 3, 4, 5, 6, 7, 8]

class_frequencies = {
    "1_1": np.array([5.4226e+09, 1.5640e+07, 1.1710e+05, 1.1879e+05, 6.0278e+05, 8.3570e+05,
        2.6682e+05, 2.6566e+05, 1.6459e+05, 6.1145e+07, 4.2558e+06, 4.4079e+07,
        2.5098e+06, 5.6889e+07, 1.5568e+07, 1.5888e+08, 2.0582e+06, 3.7056e+07,
        1.1631e+06, 3.3958e+05]),
    "1_2": np.array([2.2871e+08, 3.8447e+06, 2.7456e+04, 2.7629e+04, 1.3372e+05, 2.0355e+05,
            6.0825e+04, 6.4621e+04, 4.5209e+04, 1.3355e+07, 9.3588e+05, 1.0121e+07,
            6.4508e+05, 1.4682e+07, 3.4279e+06, 3.7339e+07, 4.9653e+05, 8.3823e+06,
            3.1972e+05, 9.3521e+04]),
    "1_4": np.array([2.1103e+07, 8.5094e+05, 5.5210e+03, 6.3270e+03, 2.8673e+04, 4.5410e+04,
            1.4061e+04, 1.4023e+04, 1.0599e+04, 2.8025e+06, 2.0268e+05, 2.2306e+06,
            1.6008e+05, 3.9068e+06, 7.5475e+05, 8.5634e+06, 1.2573e+05, 1.9244e+06,
            9.7388e+04, 3.0642e+04])
}
class_names = [
    "empty",         # 0 
    "car",           # 1
    "bicycle",       # 2
    "motorcycle",    # 3
    "truck",         # 4
    "other-vehicle", # 5
    "person",        # 6
    "bicyclist",     # 7
    "motorcyclist",  # 8
    "road",          # 9
    "parking",       # 10
    "sidewalk",      # 11
    "other-ground",  # 12
    "building",      # 13
    "fence",         # 14
    "vegetation",    # 15
    "trunk",         # 16
    "terrain",       # 17
    "pole",          # 18
    "traffic-sign",  # 19
]
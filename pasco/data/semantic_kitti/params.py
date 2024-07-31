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
    "1_1": np.array(
        [
            5.4226e09,
            1.5640e07,
            1.1710e05,
            1.1879e05,
            6.0278e05,
            8.3570e05,
            2.6682e05,
            2.6566e05,
            1.6459e05,
            6.1145e07,
            4.2558e06,
            4.4079e07,
            2.5098e06,
            5.6889e07,
            1.5568e07,
            1.5888e08,
            2.0582e06,
            3.7056e07,
            1.1631e06,
            3.3958e05,
        ]
    ),
    "1_2": np.array(
        [
            2.2871e08,
            3.8447e06,
            2.7456e04,
            2.7629e04,
            1.3372e05,
            2.0355e05,
            6.0825e04,
            6.4621e04,
            4.5209e04,
            1.3355e07,
            9.3588e05,
            1.0121e07,
            6.4508e05,
            1.4682e07,
            3.4279e06,
            3.7339e07,
            4.9653e05,
            8.3823e06,
            3.1972e05,
            9.3521e04,
        ]
    ),
    "1_4": np.array(
        [
            2.1103e07,
            8.5094e05,
            5.5210e03,
            6.3270e03,
            2.8673e04,
            4.5410e04,
            1.4061e04,
            1.4023e04,
            1.0599e04,
            2.8025e06,
            2.0268e05,
            2.2306e06,
            1.6008e05,
            3.9068e06,
            7.5475e05,
            8.5634e06,
            1.2573e05,
            1.9244e06,
            9.7388e04,
            3.0642e04,
        ]
    ),
}
class_names = [
    "empty",  # 0
    "car",  # 1
    "bicycle",  # 2
    "motorcycle",  # 3
    "truck",  # 4
    "other-vehicle",  # 5
    "person",  # 6
    "bicyclist",  # 7
    "motorcyclist",  # 8
    "road",  # 9
    "parking",  # 10
    "sidewalk",  # 11
    "other-ground",  # 12
    "building",  # 13
    "fence",  # 14
    "vegetation",  # 15
    "trunk",  # 16
    "terrain",  # 17
    "pole",  # 18
    "traffic-sign",  # 19
]

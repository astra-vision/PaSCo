import numpy as np


kitti_360_class_frequencies = {
    "1_1": np.array([
        2264087502,
        20098728,
        104972,
        96297,
        1149426,
        4051087,
        125103,
        105540713,
        16292249,
        45297267,
        14454132,
        110397082,
        6766219,
        295883213,
        50037503,
        1561069,
        406330,
        30516166,
        1950115,
    ]),
    "1_2": np.array([
        1648700309, 4738149, 25988, 24313, 280462, 984297, 33727, 24807231, 4309489, 10693629, 4025486, 29825455, 1648037, 77637495, 12865639, 443676, 116094, 7184544, 481844
    ]),
    "1_4": np.array([
        180561625, 1095918, 6042, 6084, 66599, 238732, 9490, 5895526, 1105257, 2618018, 1076064, 7925164, 397552, 18942509, 3306364, 135436, 39270, 1804354, 131580 
    ])
}


thing_ids = [1, 2, 3, 4, 5, 6]



kitti_360_class_names = [
    "empty",            # 0
    "car",              # 1    
    "bicycle",          # 2
    "motorcycle",       # 3  
    "truck",            # 4
    "other-vehicle",    # 5
    "person",           # 6
    "road",             # 7
    "parking",          # 8
    "sidewalk",         # 9
    "other-ground",     # 10 
    "building",         # 11
    "fence",            # 12
    "vegetation",       # 13
    "terrain",          # 14
    "pole",             # 15
    "traffic-sign",     # 16
    "other-structure",  # 17
    "other-object",     # 18
]
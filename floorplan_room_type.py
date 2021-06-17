ROOM_TYPE = {
	0:  [224,224,224], # nothing
	1:  [224,255,192], # living
	2:  [ 72,135,224], # dining
	3:  [ 72,135,224], # kitchen
	4:  [255,224,128], # bedroom
	5:  [192,255,255], # bathroom
	6:  [192,255,255], # toilet
	7:  [149,230,140], # office
    8:  [255,160, 96], # hallway
    9:  [ 68, 97, 31], # utility
    10: [104,117, 87], # storage
    11: [192,192,224], # closet
    12: [168,157, 54], # landing
    13: [ 97, 90, 31], # attic
    14: [255,224,224], # balcony
    15: [ 31, 97, 46], # garden
    16: [168, 92, 54], # patio
    17: [ 97, 53, 31], # parking
    18: [117, 97, 87], # porch
    19: [230,135,131], # garage
    20: [168, 58, 54], # shed
    21: [224,224,224], # pillar
    22: [224,224,224], # column
    23: [255,224,128], # dressroom
    24: [192,192,224], # builtincloset
    25: [224,255,192], # living & dining
    26: [224, 72,118], # meeting
    27: [ 97, 31, 51], # terrace
    28: [168, 54, 88], # baby & kids
    29: [117,168, 66], # basement
    30: [  0,  0,  0]  # background
}

MODEL_EDITOR = {
    0: [30],
    1: [11, 24],
    2: [5, 6],
    3: [1, 2, 3, 25],
    4: [4, 23],
    5: [8],
    6: [14, 27],
    7: [0, 7, 8, 9, 10, 12, 13, 15, 16,\
        17, 18, 19, 20, 21, 22, 26, 28, 29]
}

def rgb_bgr_converter(colormap):
    new_cmap = colormap.copy()
    tmp = new_cmap[0]
    new_cmap[0] = new_cmap[2]
    new_cmap[2] = tmp
    return new_cmap

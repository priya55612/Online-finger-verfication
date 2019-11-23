import numpy as np
''' Class to represent Minutiae point'''
class Ridge:
    # x coordinate of Ridge
    x = None
    # y coordinate of Ridge
    y = None
    # Orientation of ridge
    orientation = None

    def __init__(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation

    def __repr__(self):
        return 'Ridge:(%s,%s)@%s' % (self.x, self.y, np.around(self.orientation,2))

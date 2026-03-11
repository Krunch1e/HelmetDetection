from config import HELMET_RATIO_MIN, HELMET_RATIO_MAX


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def inside_region(box, region):
    cx, cy = box_center(box)
    x1, y1, x2, y2 = region
    return x1 < cx < x2 and y1 < cy < y2


def aspect_ratio_valid(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if h == 0:
        return False
    ratio = w / h
    return HELMET_RATIO_MIN < ratio < HELMET_RATIO_MAX

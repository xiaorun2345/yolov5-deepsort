import numpy as np
import cv2

from Point2GPS import *

# h = [[ 1.05993651e+01, 5.16641556e+01, -2.64867965e+03],
#  [ 7.95691926e-01, -1.17141412e+00, -5.68733967e+01],
#  [-1.05498900e-02,  6.78473583e-03,  1.00000000e+00]]
#
# h_inv = [[3.33439633e-02, 1.61596801e+00, 1.12862911e+02],
#  [8.30565013e-03, 4.02488973e-01, 2.81123976e+01],
#  [2.95424072e-04, 1.43175054e-02, 1.00000000e+00]]

#RGB --- BGR
COLORS_10 = [(215, 11, 26),  (4, 30, 215),   (215, 7, 204),  (98, 215, 4),    (45, 99, 2),    (6, 44, 125),
             (62, 204, 204), (195, 201, 82), (194, 93, 190), (100, 191, 186), (114, 186, 83), (130, 97, 230),
             (201, 9, 121),  (103, 139, 143), (63, 85, 87)]
# CLASSES_4 = ['unknown', 'motor', 'non-motor', 'pedestrian']
# CLASSES_4 = ['u', 'm', 'n', 'p']
def draw_bbox(img, box, cls_name, identity=None, offset=(0,0)):
    '''
        draw box of an id
    '''
    x1,y1,x2,y2 = [int(i+offset[idx%2]) for idx,i in enumerate(box)]
    # set color and label text
    color = COLORS_10[identity%len(COLORS_10)] if identity is not None else COLORS_10[0]
    label = '{} {}'.format(cls_name, identity)
    # box text and bar
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
    cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
    cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    return img


def draw_bboxes(img, bbox, identities, cls, trace, h_inv, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        clsIdx = int(cls[i])
        # className = CLASSES_4[clsIdx]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = COLORS_10[id % len(COLORS_10)]
        # label = '{}{:d}'.format(className, id)
        label = '{}{:d}'.format('', id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.1, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.rectangle(img, (x1, y1 + 40), (x1 + t_size[0] + 23, y1 + t_size[1] + 44), color, -1)

        # cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [200,0,0], 1)
        cv2.putText(img, label, (x1 + 4, y1 + t_size[1] + 44), cv2.FONT_HERSHEY_COMPLEX, 1.1, [255, 255, 255], 2)

        if len(trace[i]) > 1:
            for j in range(len(trace[i]) - 1):
                cv2.line(img, tuple(trace[i][j]), tuple(trace[i][j+1]), color, 4)

        if len(trace[i]) > 20:
            object_ceter = np.ones((3, 1))
            object_ceter[0] = trace[i][19][0]
            object_ceter[1] = trace[i][19][1]
            after = np.matmul(h_inv, object_ceter)
            after[0] = after[0] / after[2]
            after[1] = after[1] / after[2]
            after[2] = 1
            after_longitude, after_latitude = after[0], after[1]

            object_ceter[0] = trace[i][0][0]
            object_ceter[1] = trace[i][0][1]
            before = np.matmul(h_inv, object_ceter)
            before[0] = before[0] / before[2]
            before[1] = before[1] / before[2]
            before[2] = 1
            before_longitude, before_latitude = before[0], before[1]

            speed, heading = GPS2Speed(before_longitude, before_latitude, after_longitude, after_latitude, 1.2)
            speed=float('%.3f' % (speed*3.6))
            label=str(speed)+"km/h"
            #cv2.putText(img, label, (x1 , y1-10), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0, 0, 255], 2)

    return img

def draw_bboxesDetect(img, bbox, cls):
    for i,box in enumerate(bbox):
        x1,y1,w,h = [int(i) for i in box]
        x2 = x1 + w
        y2 = y1 + h
        # box text and bar
        # id = int(identities[i]) if identities is not None else 0
        id = int(cls[i]) if cls is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
        cv2.rectangle(img,(x1, y1+20),(x1+t_size[0]+3,y1+t_size[1]+24), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+ 24), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

def softmax(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(x*5)
    return x_exp/x_exp.sum()

def softmin(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(-x)
    return x_exp/x_exp.sum()

if __name__ == '__main__':
    x = np.arange(10)/10.
    x = np.array([0.5,0.5,0.5,0.6,1.])
    y = softmax(x)
    z = softmin(x)
    import ipdb; ipdb.set_trace()

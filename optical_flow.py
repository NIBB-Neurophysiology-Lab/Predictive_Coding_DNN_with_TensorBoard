import argparse
import csv
import cv2
import itertools
import numpy as np
import yaml
import os

# TODO return np array
# draws vector tracks on the image
def draw_tracks(img, data, vector_scale=60, circle_size=2, circle_color="yellow", line_width=2, line_color="red"):
    colormap = {'blue': [255, 0, 0], 'green': [0, 255, 0], 'red': [0, 0, 255],
            'yellow': [0, 255, 255], 'white': [255, 255, 255]}
    # draw the tracks
    for x, y, dx, dy in data:
        cv2.line(img, (x, y), (int(x + vector_scale*dx), int(y + vector_scale*dy)), colormap[line_color], line_width)
        cv2.circle(img, (int(x), int(y)), circle_size, colormap[circle_color], -1)

    return img

def save_data(image, vectors, output_path, original_filename):
    filename = original_filename.split("/")
    filename = filename[len(filename)-1]
    temp = filename.split(".")

    output_file = output_path + "/" + temp[0] + ".png"
    print("saving", output_file)
    cv2.imwrite(output_file, image)    
    output_file = output_path + "/csv/" + temp[0] +".csv"
    with open(output_file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(vectors)

# returns image with vector tracks and vector data
def lucas_kanade(file1, file2, output_path = "./",
    vector_scale=60, circle_size=2, circle_color="yellow", line_width=2, line_color="red",
    save = True):

    conf_path = os.path.dirname(os.path.abspath(__file__)) + "/optical_flow_config.yaml"
    if not os.path.exists(output_path+"/csv/"):
        os.makedirs(output_path+"/csv/")

    config = yaml.load(open(conf_path), Loader=yaml.FullLoader)
    conf = config['LucasKanade']
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100,
                          qualityLevel = conf['quality_level'],
                          minDistance = 7,
                          blockSize = 7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize = (conf['window_size'],
                                conf['window_size']),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(img1_gray, mask = None, **feature_params)

    data = []
    if p0 is None:
        pass
    else:
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p0, None, **lk_params)
        good_new = p1[st==1]
        good_old = p0[st==1]

        # gather results in array
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            dx = a - c
            dy = b - d
            data.append([c, d, dx, dy])

    if save:
        img2 = draw_tracks(img2, data, vector_scale, circle_size, circle_color, line_width, line_color)
        save_data(img2, data, output_path, file1)

    results = {"image":img2, "vectors":data}
    return results

def farneback(file1, file2, output_path = "./", 
    vector_scale=60, circle_size=2, circle_color="yellow", line_width=2, line_color="red",
    save = True):
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    conf = config['Farneback']

    img1 = cv2.imread(file1)
    prv = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img1)
    img2 = cv2.imread(file2)
    nxt = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prv, nxt, None, 0.5, 3,
                                        conf['window_size'],
                                        3, 5, 1.1, 0)
    height, width = prv.shape

    data = []
    for x, y in itertools.product(range(0, width, conf['stride']),
                                  range(0, height, conf['stride'])):
        if np.linalg.norm(flow[y, x]) >= conf['min_vec']:
            dy, dx = flow[y, x].astype(int)
            data.append([x, y, dx, dy])

    if save:
        img2 = draw_tracks(img2, data, vector_scale, circle_size, circle_color, line_width, line_color)
        save_data(img2, data, output_path, file1)

    results = {"image":img2, "vectors":data}
    return results

if __name__ == "__main__":
    colormap = {'blue': [255, 0, 0], 'green': [0, 255, 0], 'red': [0, 0, 255],
            'yellow': [0, 255, 255], 'white': [255, 255, 255]}

    parser = argparse.ArgumentParser(description='Calculate optical flow.')
    parser.add_argument('image1', type=str, help='Input image No1.')
    parser.add_argument('image2', type=str, help='Input image No2.')
    parser.add_argument('--output_path', '-o', type=str, default='flow/', help='path to output directory')
    parser.add_argument('--method', '-m', type=str, default='lk', choices=['lk', 'fb'], help='Select a method.')
    parser.add_argument('--circle_color', '-cc', type=str, default='yellow', choices=colormap.keys(), help='Select a color for circle.')
    parser.add_argument('--line_color', '-lc', type=str, default='red', choices=colormap.keys(), help='Select a color for line.')
    parser.add_argument('--vector_scale', '-vs', type=float, default=60.0, help='Scale saving vector data.')
    parser.add_argument('--circle_size', '-s', type=int, default=2, help='Size of original point marker.')
    parser.add_argument('--line', '-l', type=int, default=2, help='Width of vector line.')
    args = parser.parse_args()

    if args.method == 'lk':
        lucas_kanade(args.image1, args.image2, args.output_path, args.vector_scale,
            args.circle_size, args.circle_color, args.line, args.line_color)
    elif args.method == 'fb':
        farneback(args.image1, args.image2, args.output_path, args.vector_scale,
            args.circle_size, args.circle_color, args.line, args.line_color)

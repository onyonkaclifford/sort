"""
SORT: A Simple, Online and Realtime Tracker
Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from sort import Sort

np.random.seed(0)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    return parser.parse_args()


if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.randint(0, 266, (32, 3))  # used only for display
    colours = [(int(i[0]), int(i[1]), int(i[2])) for i in colours]
    text_colour = (55, 255, 155)
    to_exit = False

    if display:
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n'
                '    Create a symbolic link to the MOT benchmark\n'
                '    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n'
                '    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n'
            )
            exit()

    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')

    for seq_dets_fn in glob.glob(pattern):
        if to_exit:
            break
        mot_tracker = Sort(args.max_age, args.min_hits, args.iou_threshold)
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=",")
        seq = seq_dets_fn[pattern.find("*"):].split(os.path.sep)[0]

        with open(os.path.join("output", "%s.txt" % seq), "w") as out_file:
            print("Processing %s." % seq)
            for frame in range(int(seq_dets[:, 0].max())):
                if to_exit:
                    break
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if display:
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % frame)
                    im = cv2.imread(fn)

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print(
                        "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1" % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                        file=out_file
                    )
                    if display:
                        d = d.astype(np.int32)
                        colour = colours[d[4] % 32]
                        cv2.rectangle(im, (d[0], d[1]), (d[2], d[3]), colour, 2)
                        cv2.putText(
                            im, seq + ' Tracked Targets', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_colour, 2
                        )

                if display:
                    cv2.imshow("Tracking", im)
                    if cv2.waitKey(33) == 27:
                        to_exit = True

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    if display:
        print("Note: to get real runtime results run without the option: --display")

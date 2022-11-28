import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt

import modal

stub = modal.Stub("vehicle-counter")
volume = modal.lookup('videos_volume')

#******* INCOMING OUTGOING FEATURE *******
activateIncomingOutgoing=True
objectsTrackInOut=["car"]#["truck","car"]
incomingOutgoingLineHorizontal=True
incomingLineWrtHeightOrWidth=0.17
incomingLineThicknessWrtHeightOrWidth=0.025
outgoingLineWrtHeightOrWidth=0.23
outgoingLineThicknessWrtHeightOrWidth=0.025

incomingTrackIdsList=[]
outgoingTrackIdsList=[]
incomingCount=[]#in this list, index 0 will contain counts of objectsTrackInOut[0] object
outgoingCount=[]
for i in objectsTrackInOut:
    incomingCount.append(0)
    outgoingCount.append(0)
#******* INCOMING OUTGOING FEATURE *******


@stub.function(
    gpu=True,
    image=modal.Image.debian_slim()
    .pip_install(["torch==1.13.0", "pandas", "requests",
                  "Pillow", "opencv-python>=4.1.2", "PyYAML>=5.3.1",
                  "tqdm>=4.41.0", "torchvision>=0.10.1", "matplotlib",
                  "seaborn>=0.11.0", "easydict", "scipy>=1.4.1", "psutil"])
    .run_commands(["apt-get install ffmpeg libsm6 libxext6  -y"]),
    mounts=[
        modal.Mount(remote_dir="/root/deep_sort_pytorch",
                    local_dir="deep_sort_pytorch"),
                    modal.Mount(remote_dir="/root/yolov5/weights",
                    local_dir="yolov5/weights"),
        modal.Mount(remote_dir="/root/input", local_dir="input")],
)
def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(yolo_weights, device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    for frame_idx, (path, img, im0s, vid_cap,y) in enumerate(dataset):
        if frame_idx > 1000:
            break
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        ######GET COLOR OF CAR##########
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        if show_vid or save_vid:

                            try:
                                padding=5
                                croppedImage=im0[output[1]+padding:output[3]-padding,output[0]+padding:output[2]-padding]
                                heightCroppedImage, widthCroppedImage, _ = np.shape(croppedImage)
                                avg_color_per_row = np.average(croppedImage, axis=0)
                                avg_colors = np.average(avg_color_per_row, axis=0)
                                carColor=(int(avg_colors[2]),int(avg_colors[1]),int(avg_colors[0]))
                            except:
                                carColor=(0,0,255)
                            ######GET COLOR OF CAR##########

                            label = f'{id} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=carColor)#colors(c, True)


###########INCOMING OUTGOING##############################################################################

                        # im0 = cv2.circle(im0, (200,200), 10, (255,0,0), 10)

                        height,width,channels=im0.shape

                        center_x=int(bbox_left+(bbox_w)/2)
                        center_y=int(bbox_top+(bbox_h)/2)

                        if activateIncomingOutgoing:
                            if show_vid or save_vid:
                                cv2.line(im0, (0, int(incomingLineWrtHeightOrWidth*height+incomingLineThicknessWrtHeightOrWidth*height)), (width, int(incomingLineWrtHeightOrWidth*height+incomingLineThicknessWrtHeightOrWidth*height)), (255, 0, 0), thickness=2)
                                cv2.line(im0, (0, int(incomingLineWrtHeightOrWidth*height-incomingLineThicknessWrtHeightOrWidth*height)), (width, int(incomingLineWrtHeightOrWidth*height-incomingLineThicknessWrtHeightOrWidth*height)), (255, 0, 0), thickness=2)

                                cv2.line(im0, (0, int(outgoingLineWrtHeightOrWidth*height+outgoingLineThicknessWrtHeightOrWidth*height)), (width, int(outgoingLineWrtHeightOrWidth*height+outgoingLineThicknessWrtHeightOrWidth*height)), (0, 0, 255), thickness=2)
                                cv2.line(im0, (0, int(outgoingLineWrtHeightOrWidth*height-outgoingLineThicknessWrtHeightOrWidth*height)), (width, int(outgoingLineWrtHeightOrWidth*height-outgoingLineThicknessWrtHeightOrWidth*height)), (0, 0, 255), thickness=2)

                            class_name=names[c]
                            if class_name in objectsTrackInOut:
                                if center_y <= int(incomingLineWrtHeightOrWidth*height+incomingLineThicknessWrtHeightOrWidth*height) and center_y >= int(incomingLineWrtHeightOrWidth*height-incomingLineThicknessWrtHeightOrWidth*height):
                                    #Incoming zone touched
                                    objectTrackId=int(id)
                                    if objectTrackId not in incomingTrackIdsList:#Added because multiple same objectTrackId were appended to incomingTrackIdsList because of frames
                                        incomingTrackIdsList.append(objectTrackId)
                                        if objectTrackId not in outgoingTrackIdsList:
                                            index=objectsTrackInOut.index(class_name)
                                            incomingCount[index]+=1
                                            outgoingTrackIdsList.append(objectTrackId)#Added so that a person is only counted once
                                        elif objectTrackId in outgoingTrackIdsList:
                                            incomingTrackIdsList.remove(objectTrackId)
                                            outgoingTrackIdsList.remove(objectTrackId)

                            if class_name in objectsTrackInOut:
                                if center_y <= int(outgoingLineWrtHeightOrWidth*height+outgoingLineThicknessWrtHeightOrWidth*height) and center_y >= int(outgoingLineWrtHeightOrWidth*height-outgoingLineThicknessWrtHeightOrWidth*height):
                                    #Outgoing zone touched
                                    objectTrackId=int(id)
                                    if objectTrackId not in outgoingTrackIdsList:
                                        outgoingTrackIdsList.append(objectTrackId)
                                        if objectTrackId not in incomingTrackIdsList:
                                            index=objectsTrackInOut.index(class_name)
                                            outgoingCount[index]+=1
                                            incomingTrackIdsList.append(objectTrackId)#Added so that a person is only counted once
                                        elif objectTrackId in incomingTrackIdsList:
                                            incomingTrackIdsList.remove(objectTrackId)
                                            outgoingTrackIdsList.remove(objectTrackId)

            
###########INCOMING OUTGOING##############################################################################

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                               f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()


            initialHeight=20
            if activateIncomingOutgoing and (save_vid or show_vid):
                for objectName in objectsTrackInOut:
                    index=objectsTrackInOut.index(objectName)
                    cv2.putText(im0, "Incoming "+objectName+"s: " + str(incomingCount[index]), (10, initialHeight), 0, 0.8, (0, 0, 255), 2)
                    initialHeight+=30
                    cv2.putText(im0, "Outgoing "+objectName+"s: " + str(outgoingCount[index]), (10,initialHeight), 0, 0.8, (0,0,255), 2)
                    initialHeight+=30

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if vid_writer:
        vid_writer.release()
    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
        volume.write_file(save_path,open(save_path, "rb"))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    with stub.run():
        parser = argparse.ArgumentParser()
        parser.add_argument('--yolo_weights', nargs='+', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str, default='0', help='source')
        parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
        parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
        parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
        # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--evaluate', action='store_true', help='augmented inference')
        parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
        args = parser.parse_args()
        args.img_size = check_img_size(args.img_size)

        # with torch.no_grad():
        #     detect(args)
        detect(args)

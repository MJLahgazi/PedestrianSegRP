from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time


# =========================================================== YOLO V3 (Human only)=======================================================
def my_yolo(I):
    wht = 320
    confThreshold = 0.5
    nmsThreshold = 0.3
    classes = ['person']

    modelConfiguration = 'yolov3_320.cfg'
    modelWeights = 'yolov3_320.weights'

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    boxes = []

    def findObjects(outputs, img):
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                if classId == 0:
                    confidence = scores[classId]
                    if confidence > confThreshold:
                        w, h = int(det[2] * wT), int(det[3] * hT)
                        x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                        bbox.append([x, y, w, h])
                        classIds.append(classId)
                        confs.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

        for i in indices:
            # i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            boxes.append([x, y, w, h])
            cv2.rectangle(I, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(I, f'{int(confs[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    blob = cv2.dnn.blobFromImage(I, 1 / 255, (wht, wht), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()

    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    findObjects(outputs, I)
    return I, boxes


# ====================================================  SqueezeUNet =============================================================================
def read_img(img_path):
    batch = []
    height = 512
    width = 512
    img = img_path
    img = cv2.resize(img, (height, width))
    img = img / 255.
    batch.append(img)
    batch = np.array(batch)
    return batch


def optimal_s(pred, gt):
    iou = []
    S = []
    height, width, channels = pred.shape
    for s in np.arange(0.05, 0.8, 0.05):
        B1 = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        B1[B1 >= s] = 255
        B1[B1 < s] = 0
        B1 = B1.astype(np.uint8)

        ret1, thresh1 = cv2.threshold(B1, 50, 255, cv2.THRESH_BINARY)
        contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        D1 = np.zeros((height, width), dtype="uint8")
        cv2.drawContours(D1, contours1, -1, (255, 255, 255), -1)

        DD_I = D1 + gt
        DD_I[DD_I == 255] = 0
        DD_I[DD_I != 0] = 1

        DD_U = D1 + gt
        DD_U[DD_U != 0] = 1

        I_area = sum([sum(x) for x in DD_I])
        U_area = sum([sum(x) for x in DD_U])

        IOU = I_area / U_area

        iou.append(IOU)
        S.append(s)

    return S[iou.index(max(iou))]


def my_seg(I, M):
    tmp_img = I
    tmp_img_gt = M

    tmp_img_gt = cv2.cvtColor(tmp_img_gt, cv2.COLOR_BGR2GRAY)
    tmp_img_gt = tmp_img_gt.astype(np.uint8)
    height, width, channels = tmp_img.shape

    ret_gt, thresh_gt = cv2.threshold(tmp_img_gt, 50, 255, cv2.THRESH_BINARY)
    contours_gt, hierarchy_gt = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    D_gt = np.zeros((height, width), dtype="uint8")
    cv2.drawContours(D_gt, contours_gt, -1, (255, 255, 255), -1)

    img = read_img(I)

    res = model.predict(img)
    img_pred = cv2.resize(res[0], (width, height))

    s = optimal_s(img_pred, D_gt)
    B1 = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
    B1[B1 >= s] = 255
    B1[B1 < s] = 0
    B1 = B1.astype(np.uint8)

    ret1, thresh1 = cv2.threshold(B1, 50, 255, cv2.THRESH_BINARY)
    contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    D1 = np.zeros((height, width), dtype="uint8")
    cv2.drawContours(D1, contours1, -1, (255, 255, 255), -1)
    return D1, D_gt


# ==========================================================================================================================

model = load_model('Unet.h5')

file1 = open("./results/"+"data.csv", 'w', newline='')
writer1 = csv.writer(file1)

ImgDir = "test_images/"
Images = os.listdir(f"{ImgDir}images/")
Masks = os.listdir(f"{ImgDir}masks/")

iou_tmp = []
for i in range(len(Images)):
    print(i)

    img_path = f"{ImgDir}images/{Images[i]}"
    mask_path = f"{ImgDir}masks/{Masks[i]}"

    I = cv2.imread(img_path) #cv2.imread("test_images_pascal/images/person_" + id + ".image.png")
    M = cv2.imread(mask_path) #cv2.imread("test_images_pascal/masks/person_" + id + ".png")
    N = cv2.imread(img_path) #cv2.imread("test_images_pascal/images/person_" + id + ".image.png")

    start_time1 = time.time()

    sbp , D_gt= my_seg(I, M)

    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    start_time2 = time.time()

    h, w, c = I.shape
    img_out = np.zeros((h, w), dtype="uint8")
    out, b = my_yolo(I)
    for i in range(len(b)):
        x, y, w, h = b[i]

        crop = N[y:y + h, x:x + w]
        I_blur = cv2.blur(N, (45, 45))
        I_blur = cv2.blur(I_blur, (45, 45))
        I_blur[y:y + h, x:x + w] = crop

        output, tmp = my_seg(I_blur, M)
        img_out[y:y + h, x:x + w] = output[y:y + h, x:x + w]

    # cv2.imshow('Output',img_out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #========================================= Elapsed time Calculation ============================================================
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    #print("Elapsed time || UNet : {:.2f} | UNet+YOLO : {:.2f} ".format(elapsed_time1,elapsed_time2))


    #========================================= PSNR Calculation ============================================================
    tmp_img_gt = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)
    M = tmp_img_gt.astype(np.uint8)

    psnr1 = cv2.PSNR(sbp, M)
    psnr2 = cv2.PSNR(img_out, M)

    #print("PSNR ||  UNet : {:.2f} | UNet+YOLO : {:.2f}".format(psnr1, psnr2))

    #========================================= IoU Calculation ============================================================

    DD_I = sbp + D_gt
    DD_I[DD_I == 255] = 0
    DD_I[DD_I != 0] = 1

    DD_U = sbp + D_gt
    DD_U[DD_U != 0] = 1

    I_area = sum([sum(x) for x in DD_I])
    U_area = sum([sum(x) for x in DD_U])

    IOU1 = I_area / U_area

    DD_I = img_out + D_gt
    DD_I[DD_I == 255] = 0
    DD_I[DD_I != 0] = 1

    DD_U = img_out + D_gt
    DD_U[DD_U != 0] = 1

    I_area = sum([sum(x) for x in DD_I])
    U_area = sum([sum(x) for x in DD_U])

    IOU2 = I_area / U_area

    print("IOU ||  UNet : {:.2f} | UNet+YOLO : {:.2f}".format(IOU1, IOU2))

    print("{:.2f} {:.2f} {:.2f}  || {:.2f} {:.2f} {:.2f} ".format(psnr1,IOU1,elapsed_time1,psnr2,IOU2,elapsed_time2))
    writer1.writerow((f"{psnr1:.2f}", f"{IOU1:.2f}", f"{elapsed_time1:.2f}", f"{psnr2:.2f}", f"{IOU2:.2f}", f"{elapsed_time2:.2f}"))
    iou_tmp.append([IOU1,IOU2])


    #============================================= Creating the overlay masks ==============================================================

    # Load the input image and segmentation mask
    image = N

    mask_sbp = sbp
    # Create a color map for the mask
    mask_color = cv2.applyColorMap(mask_sbp, cv2.COLORMAP_JET)
    # Create a mask for the overlay
    overlay_mask = np.zeros_like(mask_color)
    overlay_mask[mask_sbp == 255] = (255, 140, 0)
    # Overlay the mask on the input image
    overlayed_image_sbp = cv2.addWeighted(image, 0.8, overlay_mask, 0.9, 0)

    mask_sbp_y = img_out
    # Create a color map for the mask
    mask_color = cv2.applyColorMap(mask_sbp_y, cv2.COLORMAP_JET)
    # Create a mask for the overlay
    overlay_mask = np.zeros_like(mask_color)
    overlay_mask[mask_sbp_y == 255] = (255, 140, 0)
    # Overlay the mask on the input image
    overlayed_image_sbp_y = cv2.addWeighted(image, 0.8, overlay_mask, 0.9, 0)

    # ================================ Show the visual results =========================================================

    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(N[:, :, ::-1], cmap='gray')
    # axarr[1].imshow(out[:, :, ::-1], cmap='gray')
    axarr[1].imshow(M, cmap='gray')
    axarr[2].imshow(overlayed_image_sbp[:, :, ::-1], cmap='gray')
    axarr[3].imshow(overlayed_image_sbp_y[:, :, ::-1], cmap='gray')

    axarr[0].title.set_text('Input')
    axarr[0].set_yticklabels([])
    axarr[0].set_xticklabels([])
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])

    # axarr[1].title.set_text('Yolo')
    # axarr[1].set_yticklabels([])
    # axarr[1].set_xticklabels([])
    # axarr[1].set_xticks([])
    # axarr[1].set_yticks([])

    axarr[1].title.set_text('Ground truth')
    axarr[1].set_yticklabels([])
    axarr[1].set_xticklabels([])
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])

    axarr[2].title.set_text('UNet: {:.2f}'.format(psnr1))
    axarr[2].set_yticklabels([])
    axarr[2].set_xticklabels([])
    axarr[2].set_xticks([])
    axarr[2].set_yticks([])

    axarr[3].title.set_text('UNet +  RP: {:.2f}'.format(psnr2))
    axarr[3].set_yticklabels([])
    axarr[3].set_xticklabels([])
    axarr[3].set_xticks([])
    axarr[3].set_yticks([])

    #plt.savefig("results/" + id, dpi=300)
    #plt.close()
    plt.show()

file1.close()

mean_col1 = sum(pair[0] for pair in iou_tmp) / len(iou_tmp)
mean_col2 = sum(pair[1] for pair in iou_tmp) / len(iou_tmp)

print("Mean IoU for UNet   :", mean_col1)
print("Mean IoU for UNet+RP:", mean_col2)

#print(iou_tmp)
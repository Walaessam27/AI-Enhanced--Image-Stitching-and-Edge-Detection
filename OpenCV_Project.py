# Wala' Essam Ashqar || ولاء عصام أشقر

import cv2
import numpy as np
from tkinter import filedialog
from tkinter import * 
import tkinter as Tk
from tkinter import messagebox, Label, Toplevel
from PIL import Image, ImageTk

original_img = None
output_img = None
imagefinal = None
grayfinal = None
 
def display_individual_images(images):
    window = Tk.Toplevel()
    window.title("Individual Images")
    
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((100, 100), Image.ANTIALIAS) 
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        label =Tk.Label(window, image=img_tk)
        label.image = img_tk  
        label.pack(side=Tk.LEFT)

def select_images():
    file_paths = filedialog.askopenfilenames(title="Select Images", filetypes=(("JPEG", "*.jpg"), ("PNG", "*.png"), ("GIF", "*.gif"), ("BMP", "*.bmp"), ("All files", "*.*")))
    if not file_paths:
        return
    
    images = [cv2.imread(path) for path in file_paths]

    display_individual_images(images)

    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        messagebox.showerror("Stitching error", "Image stitching failed.")
        return
    
    stitched_rgb = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
    stitched_pil = Image.fromarray(stitched_rgb)
    stitched_pil.thumbnail((400, 400), Image.ANTIALIAS)  
    stitched_tk = ImageTk.PhotoImage(image=stitched_pil)

    stitched_window = Tk.Toplevel()
    stitched_window.title("Stitched Panoramic Image")
    stitched_label = Tk.Label(stitched_window, image=stitched_tk)
    stitched_label.image = stitched_tk  
    output_img = stitched_tk
    stitched_label.pack()
        
def convertgrayscale():
    if hasattr(label2, 'image'):
        image_rgb = np.array(original_img)
        if image_rgb.dtype != np.uint8:
            image_rgb = np.array(image_rgb, dtype=np.uint8)
        grayscale_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        label_width, label_height = int(label2.cget("width")), int(label2.cget("height"))
        resizedgrayscale = cv2.resize(grayscale_image, (label_width, label_height))
        global grayfinal
        global output_img
        grayfinal = ImageTk.PhotoImage(Image.fromarray(resizedgrayscale))
        output_img = resizedgrayscale
        label2.config(image=grayfinal)
        label2.image = grayfinal

def saveimage():
    savepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if savepath:
        imgsave = Image.fromarray(output_img)
        imgsave.save(savepath)
        print(f"Processed image saved to {savepath}")

def canny(image, threshold1=50, threshold2=150):
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges

def apply_dog_and_morph(image, kernel_size=5):
    """Apply Difference of Gaussians (DoG) followed by a morphological operation."""
    gaussian_1 = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    gaussian_2 = cv2.GaussianBlur(image, (kernel_size * 2 - 1, kernel_size * 2 - 1), 0)
    dog = cv2.subtract(gaussian_1, gaussian_2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    morph = cv2.morphologyEx(dog, cv2.MORPH_CLOSE, kernel)
    
    return morph

def display_edge_detection(panorama):
    edge_window = Toplevel()
    edge_window.title("Edge Detection Results")

    panorama_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    
    canny_edges = canny(panorama_gray)
    display_image(canny_edges, edge_window, "Canny Edge Detection")

    dog_morph = apply_dog_and_morph(panorama_gray)
    display_image(dog_morph, edge_window, "DoG + Morph")

    kernel_size_slider = Scale(edge_window, from_=1, to_=21, orient='horizontal', label='Kernel Size',
                               command=lambda val: adjust_dog_morph(panorama_gray, val, edge_window))
    kernel_size_slider.pack()

def display_image(image, window, title):
    image_pil = Image.fromarray(image).convert("L")
    image_tk = ImageTk.PhotoImage(image=image_pil)
    
    frame = Frame(window)
    frame.pack()
    Label(frame, text=title).pack()
    label = Label(frame, image=image_tk)
    label.image = image_tk
    label.pack()

def adjust_dog_morph(image, kernel_size, window):
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    updated_dog_morph = apply_dog_and_morph(image, kernel_size)
    display_image(updated_dog_morph, window, "DoG + Morph Updated")  

def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    classes = open("coco.names").read().strip().split("\n")
    return net, classes, output_layers

def detect_objects(img, net, outputLayers):  
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = scores.argmax()
            conf = scores[class_id]
            if conf > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = center_x - w / 2
                y = center_y - h /2
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def start_detection(image_path):
    model, classes, output_layers = load_yolo()
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    for i in range(len(boxes)):
        if classes[class_ids[i]] == "person":
            x, y, w, h = boxes[i]
            cv2.rectangle(image, (round(x), round(y)), (round(x+w), round(y+h)), (0, 255, 0), 2)
            cv2.putText(image, f"{classes[class_ids[i]]} {int(confs[i]*100)}%", (round(x), round(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # Display in a Tkinter window
    window = Toplevel()
    window.title("AI-based Human Detection")
    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    Label(window, image=imgtk).pack()
    window.mainloop()

def applyfilter(x):
    global output_img
    filteredimage = cv2.filter2D(output_img, -1, x)
    output_img = filteredimage
    filteredfinal = ImageTk.PhotoImage(Image.fromarray(filteredimage))
    label2.config(image=filteredfinal)
    label2.image = filteredfinal

def pointdetect():
    x = np.array([[-1,-1,-1],
                  [-1, 8,-1],
                  [-1,-1,-1]], dtype=np.float32)
    x = x / (x.sum() + 1)
    applyfilter(x)

def Hline():
    x = np.array([[-1,-1,-1],
                  [ 2, 2, 2],
                  [-1,-1,-1]], dtype=np.float32)
    x = x / (x.sum() + 1)
    applyfilter(x)

def Vline():
    x = np.array([[-1, 2,-1],
                  [-1, 2,-1],
                  [-1, 2,-1]], dtype=np.float32)    
    x = x / (x.sum() + 1)
    applyfilter(x)

def line45m():
    x = np.array([[2, -1, -1],
                  [-1, 2, -1],
                  [-1, -1, 2]], dtype=np.float32)
    x = x / (x.sum() + 1)
    applyfilter(x)
    
def line45p():
    x = np.array([[-1,-1,  2],
                  [-1, 2, -1],
                  [ 2,-1, -1]], dtype=np.float32)
    x = x / (x.sum() + 1)
    applyfilter(x)

def Hedge():
     x = np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def Vedge():
     x = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def edge45p():
     x = np.array([[-2,-1, 0],
                   [-1, 0, 1],
                   [ 0, 1, 2]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def edge45m():
     x = np.array([[ 0, 1, 2],
                   [-1, 0, 1],
                   [-2,-1, 0]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def laplace():
     x = np.array([[ 0,-1, 0],
                   [-1, 4,-1],
                   [ 0,-1, 0]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def log():
     x = np.array([[ 0, 0,-1, 0, 0],
                   [ 0,-1,-2,-1, 0],
                   [-1,-2,16,-2,-1],
                   [ 0,-1,-2,-1, 0],
                   [ 0, 0,-1, 0, 0]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def zerocrossing():
    global output_img
    if output_img is not None:
        if len(output_img.shape) == 3:
            outputgray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        else:
            outputgray = output_img
        zerocrossings = np.zeros_like(outputgray, dtype=np.uint8)
        for i in range(1, outputgray.shape[0] - 1):
            for j in range(1, outputgray.shape[1] - 1):
                n = [outputgray[i - 1, j], outputgray[i + 1, j],
                              outputgray[i, j - 1], outputgray[i, j + 1]]
                if np.any(np.diff(np.sign(n))):
                    zerocrossings[i, j] = 255
        output_img = zerocrossings
        zerocrossingfinal = ImageTk.PhotoImage(Image.fromarray(zerocrossings))
        label2.config(image=zerocrossingfinal)
        label2.image = zerocrossingfinal
    else:
        print("Please open an image first.")

def threshold(threshold_value):
    global output_img
    if output_img is not None:
        if len(output_img.shape) == 3:
            outputgray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        else:
            outputgray = output_img
        _, thresholded = cv2.threshold(outputgray, threshold_value, 255, cv2.THRESH_BINARY)
        output_img = thresholded
        thresholdedfinal = ImageTk.PhotoImage(Image.fromarray(thresholded))
        label2.config(image=thresholdedfinal)
        label2.image = thresholdedfinal
    else:
        print("Please open an image first.")

def adaptivethreshold(size, c):
    global output_img
    if output_img is not None:
        if len(output_img.shape) == 3:
            outputgray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        else:
            outputgray = output_img
        adaptivethresholded = cv2.adaptiveThreshold(
            outputgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, size, c)
        output_img = adaptivethresholded
        adaptivethresholdedfinal = ImageTk.PhotoImage(Image.fromarray(adaptivethresholded))
        label2.config(image=adaptivethresholdedfinal)
        label2.image = adaptivethresholdedfinal
    else:
        print("Please open an image first.")

root = Tk.Tk()
root.geometry('1500x800')
root.title('App')

imagef  = Tk.Frame(root,highlightbackground='black',highlightthickness=1)
label2 = Label(imagef,width='450', height= '800',highlightbackground='black',highlightthickness=1)

labelsf  = Tk.Frame(root,highlightbackground='black',highlightthickness=1)
outputl = Label(labelsf,text='output image',width='50',highlightbackground='black',highlightthickness=1, height= '100',font=('bold',12))
note1 = Label(labelsf,text='Please be sure that the image in grayscale before everytime you change the filter.',width='68', height= '1',font=('bold',8))

optionsf  = Tk.Frame(root, bg= '#c3c3c3',highlightbackground='black',highlightthickness=2)
btn1 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Open images",command=select_images)
btn2 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Grayscale", command=convertgrayscale)
btn3 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Point detection",command=pointdetect)
btn4 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Horizontal edge detection",command=Hedge)
btn5 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Horizontal line detection",command=Hline)
btn6 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Vertical edge detection",command=Vedge)
btn7 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Vertical line detection",command=Vline)
btn8 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="+45 line detection",command=line45p)
btn9 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="-45 line detection", command=line45m)
btn10 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="+45 edge detection",command=edge45p)
btn11 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="-45 edge detection",command=edge45m)
btn12 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Laplacian filter",command=laplace)
btn13 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Laplacian of Gaussian",command=log)
btn14 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Zero crossing",command=zerocrossing)
btn15 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Threshold",command=lambda: threshold(128))
btn16 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Adaptive threshold",command=lambda:adaptivethreshold(11, 2))
btn17 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="AI-based Human Edge", command=start_detection(output_img))
btn18 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Canny edge detection",command=display_edge_detection(output_img))
btn19 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Save", command=saveimage)

optionsf.pack(side= LEFT)
optionsf.pack_propagate(False)
optionsf.configure(width=200, height=800) 

labelsf.pack(side=LEFT)
labelsf.pack_propagate(False)
labelsf.configure(width=150, height=800) 
outputl.pack()

imagef.pack(side=BOTTOM)
imagef.pack_propagate(False)
imagef.configure(width=900, height=700) 
label2.pack(side = RIGHT)
note1.pack()
note1.place(x=480,y=64)

btn1.pack(pady=4)
btn2.pack(pady=3)
btn3.pack(pady=3)
btn4.pack(pady=3)
btn5.pack(pady=3)
btn6.pack(pady=3)
btn7.pack(pady=3)
btn8.pack(pady=3)
btn9.pack(pady=3)
btn10.pack(pady=3)
btn11.pack(pady=3)
btn12.pack(pady=3)
btn13.pack(pady=3)
btn14.pack(pady=3)
btn15.pack(pady=3)
btn16.pack(pady=3)
btn17.pack(pady=3)
btn18.pack(pady=3)
btn19.pack(pady=3)


root.mainloop() 

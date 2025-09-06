import random
import cv2
import imageio
import torch
import numpy as np
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
#from skimage.future import graph #old version  from skimage import graph           # 
from skimage import graph
import skimage.feature as feature
from skimage import data, io, color
from torchvision.datasets import CIFAR10
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import find_contours, regionprops, label
NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

NUM_FEATURES = 5 #104 #159 #5
NUM_CLASSES = 10

def be_np(PIL_image):
    # load the image and convert it to a floating point data type
    image = np.asarray(PIL_image)
    return image

def get_histogram(image, mask):
    # cv2.cvtColor(image, cv2.COLOR_Lab2RGB)
    hist = cv2.calcHist([image], [0,1,2], mask.astype(np.uint8), [2, 2, 2], [0, 256, 0, 256, 0, 256]) 
    return hist.reshape(2*2*2,)

#https://stackoverflow.com/questions/40703086/python-taking-the-glcm-of-a-non-rectangular-region
def texture_features(greyscale_image, mask, canny):
    inverse_mask = 255 - mask
    region = np.where(inverse_mask==0, greyscale_image, 256) #256 =ignore
    outside_region = np.where(inverse_mask==255, greyscale_image, 256)

    in_region_factor = (np.sum(mask)/255)
    outside_region_factor = np.sum(inverse_mask)/255
    # cv2.imwrite(f"Canny.png", canny)
    # cv2.imwrite(f"sobel.png", sobelxy)

    lbp = feature.local_binary_pattern(greyscale_image, 24, 3, 'uniform')

    in_lbp_sum = np.sum(np.where(mask==255, lbp, 0))
    in_lbp_mean = in_lbp_sum/in_region_factor

    out_lbp_sum = np.sum(np.where(inverse_mask==255, lbp, 0))
    out_lbp_mean = out_lbp_sum/outside_region_factor if outside_region_factor>=1 else out_lbp_sum

    in_edge_sum = np.sum(np.where(mask==255, canny, 0))
    in_edge_mean = in_edge_sum/in_region_factor

    out_edge_sum = np.sum(np.where(inverse_mask==255, canny, 0))
    out_edge_mean = out_edge_sum/outside_region_factor if outside_region_factor>=1 else out_edge_sum

    in_glcm = feature.graycomatrix(region, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], levels=257)
    out_glcm = feature.graycomatrix(outside_region, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], levels=257)

    in_filt_glcm = in_glcm[:256, :256, :, :] #ignore the channel 256 because the value 256 is the flag to ignore pixels outside the mask
    out_filt_glcm = out_glcm[:256, :256, :, :]

    contrast = feature.graycoprops(in_filt_glcm, 'contrast')
    dissimilarity = feature.graycoprops(in_filt_glcm, 'dissimilarity')
    homogeneity = feature.graycoprops(in_filt_glcm, 'homogeneity')
    energy = feature.graycoprops(in_filt_glcm, 'energy')
    correlation = feature.graycoprops(in_filt_glcm, 'correlation')
    asm = feature.graycoprops(in_filt_glcm, 'ASM')

    contrast_out = feature.graycoprops(out_filt_glcm, 'contrast')
    dissimilarity_out = feature.graycoprops(out_filt_glcm, 'dissimilarity')
    homogeneity_out = feature.graycoprops(out_filt_glcm, 'homogeneity')
    energy_out = feature.graycoprops(out_filt_glcm, 'energy')
    correlation_out = feature.graycoprops(out_filt_glcm, 'correlation')
    asm_out = feature.graycoprops(out_filt_glcm, 'ASM')

    in_features = np.concatenate((contrast, dissimilarity, homogeneity, energy,
                                  correlation, asm, np.array([in_lbp_sum, in_lbp_mean, in_edge_sum, in_edge_mean]).reshape(1,4)),
                                  axis=1)
    
    out_features = np.concatenate((contrast_out, dissimilarity_out, homogeneity_out, energy_out,
                                  correlation_out, asm_out, np.array([out_lbp_sum, out_lbp_mean, out_edge_sum, out_edge_mean]).reshape(1,4)), 
                                  axis=1)
    out = np.concatenate((in_features,out_features),axis=1)
    # out = in_features
    if(np.asarray(out).shape != (1,68)):
        # print("")
        raise ValueError("More features than expected !!!!!")
    return out.reshape(68,)

def cv_mask(mask):
    # ret, mask2 = cv2.threshold(mask.astype(np.uint8), 254, 255, 0)
    # scikit_border = mask_to_border2(mask, n=0)
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    h, w = mask.shape
    border = np.zeros((h,w))
    for contour in contours:
        for c in contour:
            x = c[0][1]
            y = c[0][0]
            border[x][y]=255
    # plt.imshow(cv2.drawContours(mask.astype(np.uint8), contours, 0, (0,255,0), 3))
    # cv2.imshow(border.astype(np.uint8))
    # cv2.imshow(scikit_border.astype(np.uint8))
    # border+=(255-mask)
    return border, contours

def extract_prop_features(prop, cnt, image, mask):
    features = []
    # M = cv2.moments(cnt)
    # cv_area = cv2.contourArea(cnt)
    # cv_hull = cv2.convexHull(cnt)
    # cv_hull_area = cv2.contourArea(cv_hull)
    # cv_solidity = float(cv_area)/cv_hull_area
    # cv_mean_val = cv2.mean(image,mask = mask)
    # x0, y0 = prop.centroid
    orientation = prop.orientation
    x1 = prop.bbox[1]-1 if prop.bbox[1]>=32 else prop.bbox[1]
    y1 = prop.bbox[0]-1 if prop.bbox[0]>=32 else prop.bbox[0]
    x2 = prop.bbox[3]-1 if prop.bbox[3]>=32 else prop.bbox[3]
    y2 = prop.bbox[2]-1 if prop.bbox[2]>=32 else prop.bbox[2]
    bbox_area = prop.area_bbox
    area = prop.area
    solidity = prop.solidity
    eccentricity = prop.eccentricity
    euler_number = prop.euler_number
    convex_area = prop.area_convex
    solidity = prop.solidity
    perimeter = prop.perimeter
    mean_intensity = prop.intensity_mean
    moments_hu = prop.moments_hu
            #elipse measures
    # elipse_x1 = x0 + math.cos(orientation) * 0.5 * prop.axis_minor_length
    # elipse_y1 = y0 - math.sin(orientation) * 0.5 * prop.axis_minor_length
    # elipse_x2 = x0 - math.sin(orientation) * 0.5 * prop.axis_major_length
    # elipse_y2 = y0 - math.cos(orientation) * 0.5 * prop.axis_major_length
    # elipse_features.append(sorted([2*math.sqrt((elipse_x1-x0)**2 + (elipse_y1-y0)**2), 
    #                             2*math.sqrt((elipse_x2-x0)**2 + (elipse_y2-y0)**2)]))
            
    features.extend([x1, x2, y1, y2, bbox_area, area, perimeter, eccentricity, orientation,
                        convex_area, euler_number, solidity])
    features.extend(mean_intensity.tolist())
    features.extend(moments_hu.tolist())
    # features.extend(elipse_features[0])

    return features

#https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
#https://www.youtube.com/watch?v=RmLDL7AVXUc
#https://scikit-image.org/docs/stable/api/skimage.measure.html#regionprops
def mask_to_bbox(mask, image, mask_n, debug=False):  
    border, cv_contour = cv_mask(mask)#mask_to_border2(mask, mask_n)
    lbl = label(border)
    props = regionprops(lbl, intensity_image=image)
    area = cv2.contourArea(cv_contour[0])
    total_area=0
    cv_perimeters = []
    prop_perimeters = []
    features=[]
    if(len(cv_contour)>=2):
        for i in range (len(cv_contour)):
            cv_perimeters.append(int(cv2.arcLength(cv_contour[i],True)))
        
    for prop in props:
        if((prop.area == np.sum(border)/255)):
            features = extract_prop_features(prop, cv_contour[0], image, mask)
        total_area+=prop.area
        prop_perimeters.append(int(prop.perimeter))


    if(len(features) == 0):
        if(total_area ==  np.sum(border)/255):
            # if(max(prop_perimeters) == max(cv_perimeters)):
            features = extract_prop_features(props[prop_perimeters.index(max(prop_perimeters))], cv_contour[0], image, mask)
        else:
            x,y,w,h = cv2.boundingRect(cv_contour[0])
            for prop in props:
                if((prop.bbox[1] == x) and (prop.bbox[0]==y) and (prop.bbox[3] == (x+w)) and (prop.bbox[2] == (y+h)) ):
                     features = extract_prop_features(prop)
    if(debug):
        if(len(features)==22):
            x=cv2.rectangle(cv2.cvtColor(image, cv2.COLOR_Lab2BGR), (features[0], features[2]), (features[1], features[3]), (255, 0, 0), 1)
            border_to_save = np.expand_dims(border, axis = -1)
            border_to_save = np.concatenate([border_to_save, border_to_save, border_to_save], axis=-1)
            mask_to_save = np.expand_dims(mask, axis = -1)
            mask_to_save = np.concatenate([mask_to_save, mask_to_save, mask_to_save], axis=-1)
            cat_images = np.concatenate([border_to_save, x, mask_to_save], axis=1)
            cv2.imwrite(f"bboxes_result/debug{mask_n}.png", cat_images)
        else:
            for prop in props:
                x1 = prop.bbox[1]-1 if prop.bbox[1]>=32 else prop.bbox[1]
                y1 = prop.bbox[0]-1 if prop.bbox[0]>=32 else prop.bbox[0]
                x2 = prop.bbox[3]-1 if prop.bbox[3]>=32 else prop.bbox[3]
                y2 = prop.bbox[2]-1 if prop.bbox[2]>=32 else prop.bbox[2]
                features.extend([x1, x2, y1, y2])
            
            x=cv2.rectangle(cv2.cvtColor(image, cv2.COLOR_Lab2BGR), (features[0], features[2]), (features[1], features[3]), (255, 0, 0), 1)
            cv2.imwrite(f"debug_contour/1_bboxes{mask_n}.png", x)
            x=cv2.rectangle(cv2.cvtColor(image, cv2.COLOR_Lab2BGR), (features[4], features[6]), (features[5], features[7]), (255, 0, 0), 1)
            cv2.imwrite(f"debug_contour/2_bboxes{mask_n}.png", x)
            cv2.imwrite(f"debug_contour/contour{mask_n}.png", border)
            cv2.imwrite(f"debug_contour/mask{mask_n}.png", mask)

    if(np.asarray(features).shape != (22,)):
        # print("")
        raise ValueError("More region features than expected !!!!!") #165
    return np.asarray(features)

def RAG(image,n_nodes=6):
    NUM_FEATURES=103

    super_pixel = slic(image, n_segments=n_nodes, slic_zero=True, channel_axis=2)
    label_img=super_pixel

    asegments = np.array(label_img)
    g = graph.rag_mean_color(image, label_img)
    
    image_features=image
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(greyscale_image, (3,3), sigmaX=0, sigmaY=0) 
    # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
    num_nodes = np.max(label_img) #número de superpixels
    nodes = { #inicializa a lista de vértices
        node: {
            "rgb_list": [],
            "pos_list": [],
            "histogram": np.zeros((8,)),
            "props_features": np.zeros((22,)),
            "texture_features": np.zeros((68,)),
            "regions": [],
        } for node in range(num_nodes)
    }

    height = image.shape[0] #altura da imagem   
    width = image.shape[1]  #largura da imagem
    for y in range(height):
        for x in range(width):
            node = asegments[y,x]-1
            rgb = image_features[y,x,:]
            pos = np.array([float(x),float(y)])
            nodes[node]["rgb_list"].append(rgb) #adiciona a informação de cor referente a cada pixel do superpixel
            nodes[node]["pos_list"].append(pos) #adiciona a informação de posição referente a cada pixel do superpixel
    
    G = nx.Graph()     
    mask_n = 0                      
    for node in nodes:
        # if (node!=0):
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        # rgb
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0) #média de RGB
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0) #média da posição dos pixels pertecentes ao superpixel
        nodes[node]["histogram"] = get_histogram(image_features, np.where(asegments==node+1, 255, 0))
        nodes[node]["props_features"] = mask_to_bbox(np.where(asegments==node+1, 255, 0), image_features, mask_n, False)
        nodes[node]["texture_features"] = texture_features(greyscale_image, np.where(asegments==node+1, 255, 0), canny)
        mask_n+=1
            
        features = np.concatenate(
        [
            np.reshape(rgb_mean, -1),   #3 features (1 para cada canal)
            nodes[node]["histogram"],
            np.reshape(pos_mean, -1),   #2 features (1 para cada eixo)
            nodes[node]["texture_features"],
            nodes[node]["props_features"],
        ]
        )
        G.add_node(node, features = list(features))
    #end
    
    n = len(G.nodes)
    h = np.zeros([n,NUM_FEATURES]).astype(NP_TORCH_FLOAT_DTYPE)
    for j in G.nodes:
        h[j,:] = G.nodes[j]["features"]
    del G
    edge_list = list(g.edges()) 
    n_edges = len(list(g.edges()))
    edges = np.zeros([(2*n_edges),2]).astype(NP_TORCH_LONG_DTYPE)
    edge_features = np.zeros((edges.shape[0],1)).astype(NP_TORCH_FLOAT_DTYPE)
    i=0
    for e,(s,t) in enumerate(edge_list): 
        dist=np.linalg.norm(h[s-1]-h[t-1])
        # neighbors_s = [x - 1 for x in list(g.adj[s].keys())]
        # neighbors_t = [x - 1 for x in list(g.adj[t].keys())]

        # suport_s = np.ones((len(neighbors_s),h.shape[1]))
        # suport_t = np.ones((len(neighbors_t),h.shape[1]))

        # aux_s = suport_s*h[s-1]
        # dist_s = np.linalg.norm(np.take(h, neighbors_s, 0)-aux_s, axis=1)
        # mean_s = np.mean(dist_s)

        # aux_t = suport_t*h[t-1]
        # dist_t = np.linalg.norm(np.take(h, neighbors_t, 0)-aux_t, axis=1)
        # mean_t = np.mean(dist_t)

        edges[i,0] = s-1
        edges[i,1] = t-1
        edge_features[i] = dist
        i=i+1

        edges[i,0] = t-1
        edges[i,1] = s-1
        edge_features[i] = dist
        i=i+1
    
    return h, edges.T, edge_features, h[:,11:13], super_pixel

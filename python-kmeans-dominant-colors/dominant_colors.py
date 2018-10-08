import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("images/batman.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
image = image.reshape((image.shape[0] * image.shape[1], 3))


from sklearn.cluster import KMeans
cst = KMeans(n_clusters=5)
cst.fit(image)


def color_histogram(cst):
    
    centers = np.arange(0, len(np.unique(cst.cluster_centers_)))
    (hist, _) = np.histogram(cst.labels_, bins = centers)

    hist = hist.astype("float")
    hist /= hist.sum()

	# return the histogram
    return hist


def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    
    for (percent, color) in zip(hist, centroids):
        endX = startX + percent*300
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        
        startX = endX
    
    return bar


hist = color_histogram(cst)
bar = plot_colors(hist, cst.cluster_centers_)


plt.imshow(bar)







    


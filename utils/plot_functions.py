import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

###############
## Plotting Functions
###############
cmap = plt.cm.jet  # define the colormap
cmaplist = [(0.5, 0.5, 0.5, 1.0), (0, 0.8, 0.5, 1.0), (1.0, 0.5, 0, 1.0), (0.5, 0.2, 0.5, 1.0), (1.0, 1.0, 1.0, 1.0)]

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds = np.linspace(0, 5, 6)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


def plot_image(img):
    img_plot = plt.imshow(img)
    plt.axis('off')
    plt.show()


def plot_label(img, show_colorbar=False):
    img_plot = plt.imshow(img, cmap=cmap, norm=norm, interpolation='none' )
    if show_colorbar:
        bar = plt.colorbar(img_plot)
        bar.set_label('ColorBar')
    plt.axis('off')
    plt.show()


def plot_all_images(images):
    fig, rows = plt.subplots(nrows=(int)(np.ceil(len(images) / 4)), ncols=4, figsize=(15, 80))
    for idx, img in enumerate(images):
        rows[idx // 4][idx % 4].imshow(img)
        rows[idx // 4][idx % 4].axis('off')
    idx += 1
    if idx % 4 != 0:
        while (idx % 4 != 0):
            rows[idx // 4][idx % 4].axis('off')
            idx += 1
    plt.show()


def plot_all_labels(labels):
    fig, rows = plt.subplots(nrows=(int)(np.ceil(len(labels) / 4)), ncols=4, figsize=(15, 80))
    for idx, lbl in enumerate(labels):
        rows[idx // 4][idx % 4].imshow(lbl, cmap=cmap, interpolation='none')
        rows[idx // 4][idx % 4].axis('off')
    idx += 1
    if idx % 4 != 0:
        while (idx % 4 != 0):
            rows[idx // 4][idx % 4].axis('off')
            idx += 1
    plt.show()


def plot_sample(model, index, X, y, num_classes, name_set="", threshold_binary=0.5):
    if index > len(X) or len(X) != len(y):
        print("Dimension error!")
        return

    pred = model.predict(X[index:index + 1])
    if num_classes > 1:
        print("Classes predicted:", np.unique(np.argmax(pred[0], axis=-1)))

        fig, rows = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        rows[0][0].imshow(np.argmax(pred[0], axis=-1), cmap=cmap, norm=norm, interpolation='none')
        rows[0][0].axis('off')
        rows[0][0].title.set_text(f"Prediction {name_set}")
        rows[0][1].set_visible(False)

        rows[1][0].imshow(X[index][0])
        if num_classes > 1:
            rows[1][1].imshow(np.argmax(y[index][0], axis=-1), cmap=cmap, norm=norm, interpolation='none')
        else:
            rows[1][1].imshow(y[index][0], cmap=cmap, norm=norm, interpolation='none')

        rows[1][0].axis('off')
        rows[1][1].axis('off')
        rows[1][0].title.set_text(f"Input Image {name_set}")
        rows[1][1].title.set_text(f"Ground Truth")

    else:
        rounded = pred[0].copy()
        rounded[rounded > threshold_binary] = 1.0
        rounded[rounded <= threshold_binary] = 0.0

        fig, rows = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        rows[0][0].imshow(pred[0], cmap=cmap, norm=norm, interpolation='none')
        rows[0][1].imshow(rounded)
        rows[0][0].axis('off')
        rows[0][1].axis('off')
        rows[0][0].title.set_text(f"Unrounded Prediction {name_set}")
        rows[0][1].title.set_text(f"Rounded Prediction {name_set}")

        rows[1][0].imshow(X[index][0])

        if num_classes > 1:
            rows[1][1].imshow(np.argmax(y[index][0], axis=-1), cmap=cmap, norm=norm, interpolation='none')
        else:
            rows[1][1].imshow(y[index][0], cmap=cmap, norm=norm, interpolation='none')

        rows[1][0].axis('off')
        rows[1][1].axis('off')
        rows[1][0].title.set_text(f"Input Image {name_set}")
        rows[1][1].title.set_text(f"Ground Truth")

        plt.show()


def plot_multi_samples(model, X, y, num_classes, name_set="", threshold_binary=0.5):
    for i in range(len(X)):
        pred = model.predict(X[i:i + 1], verbose=0)
        if num_classes > 1:
            fig, rows = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
            rows[0].imshow(np.argmax(pred[0], axis=-1), cmap=cmap, norm=norm, interpolation='none')
            rows[0].axis('off')
            rows[0].title.set_text(f"Prediction {name_set}")
            rows[1].imshow(X[i][0])
            if num_classes > 1:
                rows[2].imshow(np.argmax(y[i][0], axis=-1), cmap=cmap, norm=norm, interpolation='none')
            else:
                rows[2].imshow(y[i][0], cmap=cmap, norm=norm, interpolation='none')

            rows[1].axis('off')
            rows[2].axis('off')
            rows[1].title.set_text(f"Input Image {name_set}")
            rows[2].title.set_text(f"Ground Truth")

        else:
            rounded = pred[0].copy()
            rounded[rounded > threshold_binary] = 1.0
            rounded[rounded <= threshold_binary] = 0.0

            fig, rows = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
            rows[0].imshow(pred[0], cmap=cmap, norm=norm, interpolation='none')
            rows[1].imshow(rounded)
            rows[0].axis('off')
            rows[1].axis('off')
            rows[0].title.set_text(f"Unrounded Prediction {name_set}")
            rows[1].title.set_text(f"Rounded Prediction {name_set}")

            rows[2].imshow(X[i][0])

            if num_classes > 1:
                rows[4].imshow(np.argmax(y[i][0], axis=-1), cmap=cmap, norm=norm, interpolation='none')
            else:
                rows[4].imshow(y[i][0], cmap=cmap, norm=norm, interpolation='none')

            rows[2].axis('off')
            rows[3].axis('off')
            rows[2].title.set_text(f"Input Image {name_set}")
            rows[3].title.set_text(f"Ground Truth")

        plt.show()
        


def plot_iou_sample(model, index, X, y, class_idx, name_set="", threshold_binary=0.5):
    if index > len(X) or len(X) != len(y):
        print("Dimension error!")
        return

    gt = y[index][0] 
    pred = model.predict(X[index:index + 1])
    pred2 = np.argmax(pred[0], axis=-1)

    print("Classes predicted:", np.unique(np.argmax(pred[0], axis=-1)))
    one_class_pred = np.where(pred2 == class_idx, 1, 0) 
    one_class_gt = gt[:,:,class_idx]
    
    # Create binary masks for intersection and union
    intersection = np.logical_and(one_class_gt, one_class_pred)
    union = np.logical_or(one_class_gt, one_class_pred)

    # Compute metrics
    iou = intersection.sum() / float(union.sum()) if union.sum() > 0 else 0
    dice = 2 * intersection.sum() / float(one_class_gt.sum() + one_class_pred.sum())
    
    fig, rows = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
    rows[0][0].imshow(one_class_pred, cmap=cmap, norm=norm, interpolation='none')
    rows[0][0].axis('off')
    rows[0][0].title.set_text(f"Prediction {name_set}")
    rows[0][1].title.set_text(f"Ground Truth")
    rows[0][1].imshow(one_class_gt, cmap=cmap, norm=norm, interpolation='none')
    rows[0][1].axis('off')

    rows[1][0].imshow(intersection*2, cmap=cmap, norm=norm, interpolation='none')
    rows[1][0].title.set_text(f"Intersection {name_set}")
    rows[1][1].imshow(union*3, cmap=cmap, norm=norm, interpolation='none')
    rows[1][1].title.set_text(f"Union {name_set}")
    rows[1][0].axis('off')
    rows[1][1].axis('off')


    rows[2][0].imshow(X[index][0])
    rows[2][0].title.set_text(f"Input Image {name_set}")
    rows[2][0].axis('off')
    rows[2][1].set_visible(False)
    rows[2][1].axis('off')
    
    plt.suptitle(f"Class {class_idx}, IoU: {iou:.2f}, Dice: {dice:.2f}", fontsize=16)

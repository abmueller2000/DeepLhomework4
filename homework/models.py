import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    # Convert heatmap to 4D tensor
    heatmap_4d = heatmap[None, None]
    
    # Apply max pooling
    pooled = F.max_pool2d(heatmap_4d, max_pool_ks, stride=1, padding=max_pool_ks//2)
    
    # Convert back to 2D
    pooled = pooled[0, 0]
    
    # Find local maxima
    peaks = (heatmap == pooled) & (heatmap > min_score)
    
    # Get coordinates and scores of local maxima
    y, x = torch.nonzero(peaks, as_tuple=True)
    scores = heatmap[peaks]
    
    # Sort scores in descending order and take top max_det detections
    _, idx = scores.sort(descending=True)
    idx = idx[:max_det]
    
    # Get final coordinates and scores
    final_y = y[idx]
    final_x = x[idx]
    final_scores = scores[idx]
    
    # Convert to list of tuples
    result = [(s.item(), x.item(), y.item()) for s, x, y in zip(final_scores, final_x, final_y)]
    
    return result


class Detector(torch.nn.Module):
    class Block(nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, bias=False)
            self.c2 = nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = nn.BatchNorm2d(n_output)
            self.b2 = nn.BatchNorm2d(n_output)
            self.b3 = nn.BatchNorm2d(n_output)
            self.skip = nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=3, kernel_size=3):
        super().__init__()
        L = []
        c = 3
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))
            c = l
        self.network = nn.Sequential(*L)
        self.heatmap_layer = nn.Conv2d(c, n_output_channels, kernel_size=1)

    def forward(self, x):
        z = self.network(x)
        heatmap = self.heatmap_layer(z)
        return heatmap

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
        """
        with torch.no_grad():
            heatmap = self.forward(image.unsqueeze(0))
            heatmap = torch.sigmoid(heatmap).squeeze(0)
        
        detections = []
        for i in range(heatmap.shape[0]):
            peaks = extract_peak(heatmap[i])
            detections.append([(score, cx, cy, 0, 0) for score, cx, cy in peaks])
        
        return detections


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
